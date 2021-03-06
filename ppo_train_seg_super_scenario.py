import subprocess
from PlotGenerator import final_plot


if __name__ == "__main__":
    #init and run all experiments in parralal
    scenarios = ["super_scenario"]

    seeds = [45, 59, 35]
    processes = []

    num_processes = 32
    button_number = 7
    num_env_steps = 1000000
    num_steps = 256
    num_mini_batch = 16
    filter_count = 64
    entropy_coef = 1
    clip_param = 0.1
    lr = 0.00025
    value_loss_coef = 0.5

    num_updates = int(num_env_steps) // num_steps // num_processes

    for scenario in scenarios:
        for seed in seeds:
            command = ["python", "ppo_main_seg.py",
                        "--scenario",scenario,
                        "--rep-type","seg",
                        "--arch","CONVNET_SEG",
                        "--seed",str(seed),
                        "--env-name","doom",
                        "--button_number",str(button_number),
                        "--algo","ppo",
                        "--lr",str(lr),
                        "--value-loss-coef",str(value_loss_coef),
                        "--num-processes",str(num_processes),
                        "--num-env-steps",str(num_env_steps),
                        "--filter-count",str(filter_count),
                        "--num-steps",str(num_steps),
                        "--num-mini-batch",str(num_mini_batch),
                        "--log-interval",str(1),
                        "--entropy-coef",str(0.01),
                        "--n-channels",str(1)
                       ]
            p = subprocess.Popen(command,shell=False)
            processes.append(p)
        for p in processes:
            p.wait()