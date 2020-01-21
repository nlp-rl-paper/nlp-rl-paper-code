import subprocess
from PlotGenerator import final_plot


if __name__ == "__main__":
    #init and run all experiments in parralal
    scenarios = ["defend_the_line"]

    seeds = [35, 45, 59, 12, 5]
    processes = []

    num_processes = 8
    button_number = 3
    num_env_steps = 100000
    num_steps = 128
    num_mini_batch = 16
    entropy_coef = 1
    clip_param = 0.1
    lr = 0.00025
    value_loss_coef = 0.5

    frame_repeat = 4

    num_updates = int(num_env_steps) // num_steps // num_processes


    for scenario in scenarios:
        for seed in seeds:
            command = ["python", "ppo_main_nlp.py",
                       "--scenario", scenario,
                       "--rep-type", "nlp",
                       "--seed", str(seed),
                       "--env-name", "doom",
                       "--button_number", str(button_number),
                       "--algo", "ppo",
                       "--lr", str(lr),
                       "--value-loss-coef", str(value_loss_coef),
                       "--num-processes", str(num_processes),
                       "--num-env-steps", str(num_env_steps),
                       "--num-steps", str(num_steps),
                       "--num-mini-batch", str(num_mini_batch),
                       "--log-interval", str(1),
                       "--entropy-coef", str(0.01),
                       "--n-channels", str(1),
                       "--n-patches",str(5),
                       "--load-model",
                       "--reverse-green"
                       ]
            p = subprocess.Popen(command,shell=False)
            processes.append(p)
        for p in processes:
            p.wait()
