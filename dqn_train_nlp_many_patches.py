import subprocess
import time
from PlotGenerator import final_plot


if __name__ == "__main__":
    #init and run all experiments in parralal
    scenarios = [
                 "super_scenario"
                 ]

    patches = [5,11,31,51,3]
    seeds = [35, 45, 59, 12, 5]

    processes = []

    steps_per_epoch = 500
    epochs = 200

    frame_repeat = 4

    for patch in patches:
        for scenario in scenarios:
            if "super_scenario" in scenario:
                button_number = 7
            else:
                button_number = 3
            for seed in seeds:
                command = ["python", "dqn_nlp.py",
                           "--SCENARIO", scenario,
                           "--REP_TYPE", "nlp",
                           "--SEED", str(seed),
                           "--BATCH_SIZE", "100",
                           "--ARCH", "TextCNN",
                           "--SENTANCE_LEN", "200",
                           "--LEARNING_STEPS_PER_EPOCH", str(steps_per_epoch),
                           "--HIDDEN_UNITS", "16",
                           "--FILTER_COUNT", "12",
                           "--LEARNING_RATE", "0.00025",
                           "--EPOCHS", str(epochs),
                           "--FRAME_REPEAT", str(frame_repeat),
                           "--N_PATCHES",str(patch)
                           ]
                p = subprocess.Popen(command,shell=False)
                time.sleep(1)
                processes.append(p)
            # for p in processes:
            #     p.wait()
            # del processes
            # processes = []
