import subprocess
from PlotGenerator import final_plot


if __name__ == "__main__":
    #init and run all experiments in parralal
    #scenarios = ["basic", "take_cover", "defend_the_line","defend_the_center","health_gathering"]
    #scenarios = ["health_gathering_extreme", "take_cover_extreme", "defend_the_line_extreme", "basic_extreme","defend_the_center_extreme"]
    #scenarios = ["health_gathering", "take_cover", "defend_the_line"]
    scenarios = ["super_scenario"]

    seeds = [35,45,59,12,5]
    processes = []

    steps_per_epoch = 500
    epochs = 150
    frame_repeat = 4


    for scenario in scenarios:
        final_plot(scenario, steps_per_epoch, len(seeds),True)