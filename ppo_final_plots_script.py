import subprocess
from analysis_plots import final_plot


if __name__ == "__main__":
    #init and run all experiments in parralal
    # #scenarios = ["basic", "take_cover", "defend_the_line","defend_the_center","health_gathering"]
    # #scenarios = ["defend_the_line"]
    # #scenarios = ["health_gathering_extreme", "take_cover_extreme", "defend_the_line_extreme", "basic_extreme","defend_the_center_extreme"]
    # scenarios = ["health_gathering_middle", "take_cover_middle", "defend_the_line_middle", "basic_middle","defend_the_center_middle"]
    # #scenarios = ["health_gathering", "take_cover", "defend_the_line"]

    # scenarios0 = ["basic", "take_cover", "defend_the_line", "defend_the_center", "health_gathering"]
    # scenarios1 = ["health_gathering_extreme", "take_cover_extreme", "defend_the_line_extreme", "basic_extreme",
    #               "defend_the_center_extreme"]
    # scenarios2 = ["health_gathering_middle", "take_cover_middle", "defend_the_line_middle", "basic_middle",
    #               "defend_the_center_middle"]
    # scenarios = scenarios0 + scenarios1 + scenarios2 + ["super_scenario"]

    #scenarios = ["defend_the_center_extreme","defend_the_line_extreme","super_scenario"]

    #scenarios = ["defend_the_line", "defend_the_center"]
    scenarios = ["defend_the_line","defend_the_center", "super_scenario"]

    algo = "ppo"

    seeds = [35,45,59,12,5]
    processes = []
    steps_per_epoch = 250
    epochs = 100
    frame_repeat = 4


    for scenario in scenarios:
        final_plot(algo, scenario, steps_per_epoch, 4,epochs,True)