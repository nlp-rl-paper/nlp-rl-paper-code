import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ActorCriticModelTextCNN import ActorCriticModel as ActorCriticModel
from PlotGenerator import PlotGenerator
from ppo_utils import algo, utils
from ppo_utils.algo import gail
from ppo_utils.arguments import get_args
from ppo_utils.envs import make_vec_envs
from ppo_utils.model import Policy
from ppo_utils.storage import RolloutStorage
from evaluation import evaluate


def get_free_gpu():
    import time
    time.sleep(2)
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def main():
    resoluton = (200,50)
    args = get_args()

    #torch.manual_seed(args.seed)
    #torch.cuda.manual_seed_all(args.seed)

    pg = PlotGenerator("ppo",args.scenario, args.arch, False, 1, args.seed,args.n_patches)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)

    #choose available gpu
    gpu_id = str(get_free_gpu())
    device = torch.device("cuda:"+ str(gpu_id) if args.cuda else "cpu")
    print("chosen gpu: " + gpu_id)

    envs = make_vec_envs(args.env_name, args.rep_type , resoluton ,args.seed, args.scenario ,args.num_processes,
                         args.gamma, args.log_dir, device, False, patch_count = args.n_patches,reverse_green=args.reverse_green)


    model_path = os.path.join("trained_models","ppo","ppo_TextCNN__patches=5_defend_the_center_22_09_21_25_fr=1_seed=" + str(args.seed) + ".pt")

    if args.load_model == False:
        actor_critic = ActorCriticModel(
            output_len=args.button_number,
            word_embedding_dimension=50,
            sentence_max_size=args.max_sentence_len,
            hidden_units=16,
            textcnn_filter_count=args.filter_count
        ).to(device)
    else:
        print("Loading model from: ", model_path)
        actor_critic = torch.load(model_path)[0]

    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            1,
            200,
            50,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))

        gail_train_loader = torch.utils.data.DataLoader(
            gail.ExpertDataset(
                file_name, num_trajectories=4, subsample_frequency=20),
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    print("num_updates: " + str(num_updates))
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                inputs = rollouts.obs[step].reshape([-1,1,200,50])
                #inputs = rollouts.obs[step]
                value, action, action_log_prob = actor_critic.act(inputs,rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, action, action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1].reshape([-1,1,200,50]),
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step].reshape([-1,1,200,50]), rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            #TODO: UNCOMMENT WHEN DONE WITH TRANSFER
            # torch.save([
            #     actor_critic,
            #     getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            # ], os.path.join(save_path, pg.generate_plot_name() + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
            pg.update_interactions_data(total_num_steps)
            pg.update_reward_data(np.mean(episode_rewards),np.mean(episode_rewards))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

    pg.dump_data()
    pg.plot_reward_progress()


if __name__ == "__main__":
    main()
