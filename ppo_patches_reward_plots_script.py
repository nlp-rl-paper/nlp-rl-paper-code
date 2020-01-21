import subprocess
from analysis_plots import patches_nlp_reward_plot


if __name__ == "__main__":
    #init and run all experiments in parralal
    #scenarios = ["basic", "take_cover", "defend_the_line","defend_the_center","health_gathering"]
    #scenarios = ["defend_the_line"]
    #scenarios = ["health_gathering_extreme", "take_cover_extreme", "defend_the_line_extreme", "basic_extreme","defend_the_center_extreme"]
    scenario = "super_scenario"
    #scenarios = ["health_gathering", "take_cover", "defend_the_line"]

    algos = ["dqn","ppo"]
    #algos = ["ppo"]

    patches_list = [3, 5, 11, 31]

    seeds = [35,45,59,12,5]

    steps_per_epoch = 250
    epochs = 100
    frame_repeat = 4

    for algo in algos:
        patches_nlp_reward_plot(algo, scenario, steps_per_epoch, 5 ,epochs,True,patches_list)