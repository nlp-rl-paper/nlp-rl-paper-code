import subprocess
from PlotGenerator import final_plot


if __name__ == "__main__":
    #init and run all experiments in parralal
    #scenarios = [ "basic","health_gathering", "take_cover", "defend_the_line","defend_the_center"]
    scenarios = ["basic_extreme","take_cover_extreme", "defend_the_line_extreme","defend_the_center_extreme","health_gathering_extreme"]

    seeds = [35, 45, 59, 12, 5]
    processes = []

    steps_per_epoch = 500
    epochs = 100

    frame_repeat = 4

    for scenario in scenarios:
        for seed in seeds:
            if "health_gathering" in scenario:
                frame_repeat = 12
            else:
                frame_repeat = 5
            command = ["python", "dqn_vision.py",
                       "--SCENARIO",scenario,
                       "--REP_TYPE","viz",
                       "--ARCH","CONVNET",
                       "--BATCH_SIZE","500",
                       "--N_CHANNELS","3",
                       "--EPOCHS",str(epochs),
                       "--SEED",str(seed),
                       "--LEARNING_RATE", "0.00025",
                       "--FRAME_REPEAT", str(frame_repeat),
                       "--LEARNING_STEPS_PER_EPOCH", str(steps_per_epoch),
                       "--SEED",str(seed)
                       ]
            p = subprocess.Popen(command,shell=False)
            processes.append(p)
        for p in processes:
            p.wait()

    for scenario in scenarios:
        final_plot(scenario, steps_per_epoch, len(seeds))