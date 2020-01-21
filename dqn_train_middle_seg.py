import subprocess
from PlotGenerator import final_plot


if __name__ == "__main__":
    #init and run all experiments in parralal
    scenarios = ["health_gathering_middle", "take_cover_middle", "defend_the_line_middle", "basic_middle","defend_the_center_middle"]
    #scenarios = ["basic_extreme","take_cover_extreme", "defend_the_line_extreme","defend_the_center_extreme","health_gathering_extreme"]

    seeds = [35, 45, 59, 12, 5]
    processes = []

    steps_per_epoch = 250
    epochs = 100
    learning_rate = 0.00025
    batch_size = 100
    frame_repeat = 4

    for scenario in scenarios:
        for seed in seeds:
            if "health_gathering" in scenario:
                frame_repeat = 12
            else:
                frame_repeat = 5
            command = ["python", "dqn_segmentation.py",
                       "--SCENARIO",scenario,
                       "--REP_TYPE","seg",
                       "--ARCH","CONVNET_SEG",
                       "--N_CHANNELS","1",
                       "--EPOCHS",str(epochs),
                       "--SEED",str(seed),
                       "--BATCH_SIZE",str(batch_size),
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