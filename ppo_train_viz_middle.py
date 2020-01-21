import subprocess
from PlotGenerator import final_plot


if __name__ == "__main__":
    #init and run all experiments in parralal
    #scenarios = ["basic_middle", "defend_the_line_middle","defend_the_center_middle","health_gathering_middle","take_cover_middle"]
    scenarios = ["defend_the_center_middle","defend_the_line_middle"]

    seeds = [5, 12]
    processes = []

    num_processes = 32
    filter_count = 64
    button_number = 3
    num_env_steps = 1000000
    #num_env_steps = 10
    num_steps = 128
    num_mini_batch = 16
    entropy_coef = 1
    clip_param = 0.1
    lr = 0.00025
    value_loss_coef = 0.5

    num_updates = int(num_env_steps) // num_steps // num_processes

    for scenario in scenarios:
        if scenario == "super_scenario":
            button_number = 7
        else:
            button_number = 3
        for seed in seeds:
            command = ["python", "ppo_main_viz.py",
                        "--scenario",scenario,
                        "--rep-type","viz",
                        "--arch","CONVNET",
                        "--seed",str(seed),
                        "--env-name","doom",
                        "--button_number",str(button_number),
                        "--algo","ppo",
                        "--lr",str(lr),
                        "--value-loss-coef",str(value_loss_coef),
                        "--num-processes",str(num_processes),
                        "--num-env-steps",str(num_env_steps),
                        "--num-steps",str(num_steps),
                        "--num-mini-batch",str(num_mini_batch),
                        "--log-interval",str(1),
                        "--entropy-coef",str(0.01),
                        "--n-channels",str(3),
                        "--filter-count",str(filter_count)
                       ]
            p = subprocess.Popen(command,shell=False)
            processes.append(p)
        for p in processes:
            p.wait()

