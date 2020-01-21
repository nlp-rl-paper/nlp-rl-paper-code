import subprocess
from PlotGenerator import final_plot


if __name__ == "__main__":
    #init and run all experiments in parralal
    scenarios = ["defend_the_line"]

    seeds = [35]#, 45, 59, 12, 5]
    processes = []

    steps_per_epoch = 500
    epochs = 10

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
                       "--FRAME_REPEAT",str(frame_repeat),
                       "--LOAD_MODEL",
                       "--REVERSE_GREEN"
                       ]
            p = subprocess.Popen(command,shell=False)
            processes.append(p)
        for p in processes:
            p.wait()
