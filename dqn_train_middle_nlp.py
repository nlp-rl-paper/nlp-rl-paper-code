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
            command = ["python", "dqn_nlp.py",
                       "--SCENARIO",scenario,
                       "--REP_TYPE","nlp",
                       "--SEED",str(seed),
                       "--BATCH_SIZE","100",
                       "--ARCH","TextCNN",
                       "--SENTANCE_LEN","200",
                       "--LEARNING_STEPS_PER_EPOCH",str(steps_per_epoch),
                       "--HIDDEN_UNITS","16",
                       "--FILTER_COUNT","12",
                       "--LEARNING_RATE","0.00025",
                       "--EPOCHS",str(epochs),
                       "--FRAME_REPEAT",str(frame_repeat)
                       ]
            p = subprocess.Popen(command,shell=False)
            processes.append(p)
        for p in processes:
            p.wait()

    for scenario in scenarios:
        final_plot(scenario, steps_per_epoch, len(seeds))