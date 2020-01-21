import subprocess
from analysis_plots import noise_converge_plot


if __name__ == "__main__":
    #init and run all experiments in parralal
    #scenarios = ["basic", "take_cover", "defend_the_line","defend_the_center","health_gathering"]
    #scenarios = ["health_gathering_extreme", "take_cover_extreme", "defend_the_line_extreme", "basic_extreme","defend_the_center_extreme"]
    #scenarios = ["defend_the_center_extreme"]
    #scenarios = ["health_gathering_middle", "take_cover_middle", "defend_the_line_middle", "basic_middle","defend_the_center_middle"]
    scenarios = ["defend_the_center"]

    algos = ["dqn","ppo"]
    rep_types = ["viz","seg","nlp"]



    seeds = [35,45,59,12,5]
    processes = []

    steps_per_epoch = 250
    epochs = 100
    frame_repeat = 4


    for algo in algos:
        for rep in rep_types:
            for scenario in scenarios:
                noise_converge_plot(algo,rep, scenario, steps_per_epoch, len(seeds),epochs,True)