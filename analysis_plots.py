import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

#returns 5 last runs
def get_dump_list(scenario,algo,rep_type,seed_count):
    try:
        dir_list = os.listdir(os.path.join("data",algo + "_" + scenario + "_" + rep_type))
        if len(dir_list) > 0:
            dir_list = [os.path.join("data",algo + "_" + scenario + "_" + rep_type, file) for file in dir_list]
            dir_list.sort(key=lambda x: os.path.getmtime(x),reverse=True)
            return dir_list[:int(seed_count)]
        return None
    except:
        return None

def final_plot(algo, scenario="basic",steps_per_epoch=250,seed_count=5,epochs=100,smooth=False,patches=-1):

    numpy_dumps_viz = []
    numpy_dumps_nlp = []
    numpy_dumps_vec = []
    numpy_dumps_seg = []
    numpy_dumps_ran = []

    nlp_dump_list = get_dump_list(scenario, algo, "nlp",5)
    viz_dump_list = get_dump_list(scenario, algo, "viz",seed_count)
    vec_dump_list = get_dump_list(scenario, algo, "vec",seed_count)
    seg_dump_list = get_dump_list(scenario, algo, "seg",seed_count)
    ran_dump_list = get_dump_list(scenario, algo, "ran", seed_count)

    print(viz_dump_list)
    print(seg_dump_list)
    print(nlp_dump_list)

    COLORS = ['#66624b', '#41c1dd', '#3e9651', '#e0771a', '#cc2529', '#b2a745', '#660066']

    N = 20
    f, axarr = plt.subplots(1, 1, sharex=False, squeeze=False)

    max_len = 0

    # nlp
    if nlp_dump_list is not None:
        min_vec_len = min([len(np.load(dump_file)) for dump_file in nlp_dump_list])
        max_vec_len = max([len(np.load(dump_file)) for dump_file in nlp_dump_list])
        if max_vec_len > 0:
            for dump_file in nlp_dump_list:
                vec = np.load(dump_file)
                if len(vec) > 0:
                    numpy_dumps_nlp.append(vec[:min_vec_len])
            ys = np.array(numpy_dumps_nlp)
            #print(len(np.mean(ys, axis=0)))
            ymean = np.mean(ys, axis=0)
            if smooth:
                ymean = moving_average(ymean, N)
            ystd = np.std(ys, axis=0)
            ystderr = ystd / np.sqrt(len(ys))
            ystderr = ystderr[:-N+1]
            results_x_axis = steps_per_epoch * np.arange(0, len(ymean))
            l, = axarr[0][0].plot(results_x_axis, ymean, color=COLORS[4], label="Natural Language")
            axarr[0][0].fill_between(results_x_axis, ymean - ystderr, ymean + ystderr, color=COLORS[4], alpha=.4)
            max_len = len(ymean)


    #vec
    # if vec_dump_list is not None:
    #     min_vec_len = min([len(np.load(dump_file)) for dump_file in vec_dump_list])
    #     max_vec_len = max([len(np.load(dump_file)) for dump_file in vec_dump_list])
    #     if max_vec_len > 0:
    #         for dump_file in vec_dump_list:
    #             vec = np.load(dump_file)
    #             if len(vec) > 0:
    #                 numpy_dumps_vec.append(vec[:min_vec_len])
    #
    #         ys = numpy_dumps_vec
    #         ymean = np.mean(ys, axis=0)
    #         if smooth:
    #             ymean = moving_average(ymean, N)
    #         ystd = np.std(ys, axis=0)
    #         ystderr = ystd / np.sqrt(len(ys))
    #         ystderr = ystderr[:-N+1]
    #         results_x_axis = steps_per_epoch * np.arange(0, len(ymean))
    #         l, = axarr[0][0].plot(results_x_axis, ymean, color=COLORS[1], label="Feature Vector")
    #         axarr[0][0].fill_between(results_x_axis, ymean - ystderr, ymean + ystderr, color=COLORS[1], alpha=.4)


    # seg
    if seg_dump_list is not None:
        min_vec_len = min([len(np.load(dump_file)) for dump_file in seg_dump_list])
        max_vec_len = max([len(np.load(dump_file)) for dump_file in seg_dump_list])
        if max_vec_len > 0:
            for dump_file in seg_dump_list:
                vec = np.load(dump_file)
                if len(vec) > 0:
                    numpy_dumps_seg.append(vec[:min_vec_len])

            ys = numpy_dumps_seg
            ymean = np.mean(ys, axis=0)
            if smooth:
                ymean = moving_average(ymean, N)
            ystd = np.std(ys, axis=0)
            ystderr = ystd / np.sqrt(len(ys))
            ystderr = ystderr[:-N+1]
            results_x_axis = steps_per_epoch * np.arange(0, len(ymean))
            l, = axarr[0][0].plot(results_x_axis, ymean, color=COLORS[0], label="Semantic Segmentation")
            axarr[0][0].fill_between(results_x_axis, ymean - ystderr, ymean + ystderr, color=COLORS[0], alpha=.4)


    #viz
    if viz_dump_list is not None:
        min_vec_len = min([len(np.load(dump_file)) for dump_file in viz_dump_list])
        max_vec_len = max([len(np.load(dump_file)) for dump_file in viz_dump_list])
        if max_vec_len > 0:
            for dump_file in viz_dump_list:
                vec = np.load(dump_file)
                if len(vec) > 0:
                    numpy_dumps_viz.append(vec[:min_vec_len])

            ys = numpy_dumps_viz
            ymean = np.mean(ys, axis=0)
            if smooth:
                ymean = moving_average(ymean, N)
            ystd = np.std(ys, axis=0)
            ystderr = ystd / np.sqrt(len(ys))
            ystderr = ystderr[:-N+1]
            results_x_axis = steps_per_epoch * np.arange(0, len(ymean))
            l, = axarr[0][0].plot(results_x_axis, ymean, color=COLORS[5], label="Raw Image")
            axarr[0][0].fill_between(results_x_axis, ymean - ystderr, ymean + ystderr, color=COLORS[5], alpha=.4)

    if viz_dump_list is not None or seg_dump_list is not None or vec_dump_list is not None or nlp_dump_list is not None:
        plot_title = algo.upper() + "\n"
        plot_title += scenario.replace("_"," ").capitalize()
        # if "middle" in plot_title:
        #     plot_title = plot_title.replace("middle","")
        #     plot_title += "\nlight nuisance"
        # elif "extreme" in plot_title:
        #     plot_title = plot_title.replace("extreme", "")
        #     plot_title += "\nheavy nuisance"
        # else:
        #     plot_title += "\nno nuisance"

        plt.rcParams.update({'font.size': 15})
        plt.grid()
        if scenario == "defend_the_line" and algo =="ppo":
            plt.legend(loc="upper left",prop={
                "size": 13,
            })
        if algo == "dqn":
            plt.xlim(0,20000)
        elif algo == "ppo":
            plt.xlim(0,55000)
        plt.xlabel("Interactions", fontdict={
            "size": 15
        })
        plt.ylabel("Reward", fontdict={
            "size": 15
        })
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        if patches > -1:
            #plt.title("algo: " + algo + ": " + " scenario: " + scenario + " patches: " + str(patches))
            plt.title(plot_title)
            plt.savefig(os.path.join("plots", algo + "_final_plot_" + scenario + "_patches" + str(
                patches) + "_" + datetime.now().strftime("%d_%m_%H_%M")))
        else:
            #plt.title("algo: " + algo + ": " + "\nscenario: " + scenario)
            plt.title(plot_title)
            plt.savefig(
               os.path.join("plots", algo + "_final_plot_" + scenario + "_" + datetime.now().strftime("%d_%m_%H_%M")), pad_inches=0.1)
        #plt.tight_layout()
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.show(block=False)


def noise_converge_plot(algo,rep_type,scenario,steps_per_epoch,seed_count=5,epochs=100,smooth=False):
    numpy_dumps_vanilla = []
    numpy_dumps_middle  = []
    numpy_dumps_extreme = []

    vanilla_dump_list = get_dump_list(scenario, algo, rep_type, seed_count)
    middle_dump_list  = get_dump_list(scenario + "_middle" , algo, rep_type, seed_count)
    extreme_dump_list = get_dump_list(scenario + "_extreme", algo, rep_type, seed_count)

    COLORS = ['#66624b', '#41c1dd', '#3e9651', '#e0771a', '#cc2529', '#b2a745', '#660066']
    N = 5
    f, axarr = plt.subplots(1, 1, sharex=False, squeeze=False)
    max_len = 0

    #vanilla
    if vanilla_dump_list is not None:
        min_vec_len = min([len(np.load(dump_file)) for dump_file in vanilla_dump_list])
        max_vec_len = max([len(np.load(dump_file)) for dump_file in vanilla_dump_list])
        for dump_file in vanilla_dump_list:
            vec = np.load(dump_file)
            if len(vec) > 0:
                numpy_dumps_vanilla.append(vec[:min_vec_len])
        if max_vec_len > 0:
            ys = np.array(numpy_dumps_vanilla)
            #print(len(np.mean(ys, axis=0)))
            ymean = np.mean(ys, axis=0)
            if smooth:
                ymean = moving_average(ymean, N)
            ystd = np.std(ys, axis=0)
            ystderr = ystd / np.sqrt(len(ys))
            ystderr = ystderr[:-N+1]
            results_x_axis = steps_per_epoch * np.arange(0, len(ymean))
            l, = axarr[0][0].plot(results_x_axis, ymean, color="blue", label="no nuisance")
            axarr[0][0].fill_between(results_x_axis, ymean - ystderr, ymean + ystderr, color="blue", alpha=.4)
            max_len = len(ymean)

    #middle
    if middle_dump_list is not None:
        min_vec_len = min([len(np.load(dump_file)) for dump_file in middle_dump_list])
        max_vec_len = max([len(np.load(dump_file)) for dump_file in middle_dump_list])
        for dump_file in middle_dump_list:
            vec = np.load(dump_file)
            if len(vec) > 0:
                numpy_dumps_middle.append(vec[:min_vec_len])
        if max_vec_len > 0:
            ys = np.array(numpy_dumps_middle)
            #print(len(np.mean(ys, axis=0)))
            ymean = np.mean(ys, axis=0)
            if smooth:
                ymean = moving_average(ymean, N)
            ystd = np.std(ys, axis=0)
            ystderr = ystd / np.sqrt(len(ys))
            ystderr = ystderr[:-N+1]
            results_x_axis = steps_per_epoch * np.arange(0, len(ymean))
            l, = axarr[0][0].plot(results_x_axis, ymean, color="orange", label="light nuisance")
            axarr[0][0].fill_between(results_x_axis, ymean - ystderr, ymean + ystderr, color="orange", alpha=.4)
            max_len = len(ymean)

        #extreme
        if extreme_dump_list is not None:
            min_vec_len = min([len(np.load(dump_file)) for dump_file in extreme_dump_list])
            max_vec_len = max([len(np.load(dump_file)) for dump_file in extreme_dump_list])
            for dump_file in extreme_dump_list:
                vec = np.load(dump_file)
                if len(vec) > 0:
                    numpy_dumps_extreme.append(vec[:min_vec_len])
            if max_vec_len > 0:
                ys = np.array(numpy_dumps_extreme)
                # print(len(np.mean(ys, axis=0)))
                ymean = np.mean(ys, axis=0)
                if smooth:
                    ymean = moving_average(ymean, N)
                ystd = np.std(ys, axis=0)
                ystderr = ystd / np.sqrt(len(ys))
                ystderr = ystderr[:-N + 1]
                results_x_axis = steps_per_epoch * np.arange(0, len(ymean))
                l, = axarr[0][0].plot(results_x_axis, ymean, color="red", label="heavy nuisance")
                axarr[0][0].fill_between(results_x_axis, ymean - ystderr, ymean + ystderr, color="red", alpha=.4)
                max_len = len(ymean)

        font_size = 17
        plt.rcParams.update({'font.size': 15})
        #plt.legend(loc="lower right")
        if algo == "ppo" and rep_type == "nlp":
            plt.legend(loc="upper left", prop={
                "size": 13,
            })
        if rep_type == "seg":
            rep_str = "Semantic Segmentation"
        elif rep_type == "nlp":
            rep_str = "Natural Language"
        else:
            rep_str = "Raw Image"
        #plt.title(algo.upper() + "\n" + rep_str + "\n" + scenario.replace("_"," ").capitalize(), fontdict={"size":13})
        plt.title(rep_str, fontdict={"size": font_size})
        plt.grid()
        if algo == "dqn":
            #plt.xlim(0, epochs * steps_per_epoch)
            plt.xlim(0, 23000)
        else:
            plt.xlim(0, 55000)
        #plt.ylim(-1,14)
        plt.xlabel("Interactions", fontdict={'size':font_size})
        plt.ylabel("Reward", fontdict={'size':font_size})

        plt.savefig(os.path.join("plots", algo + "_noise_robustness_plot_" + rep_type + scenario + "_" + datetime.now().strftime("%d_%m_%H_%M")), bbox_inches="tight")
        plt.show(block=False)


def patches_nlp_reward_plot(algo,scenario,steps_per_epoch,seed_count=5,epochs=100,smooth=False,patches_list=[3,5,11,31]):

    dumps_list  =  [[] for _ in range(len(patches_list))]
    numpy_dumps =  []

    COLORS = ['#66624b', '#41c1dd', '#3e9651', '#e0771a', '#cc2529', '#b2a745', '#660066']

    if algo == "ppo":
        nlp_dump_lists = get_dump_list(scenario , algo, "nlp", len(patches_list)*seed_count)
    else:
        nlp_dump_lists = get_dump_list(scenario, algo, "nlp", len(patches_list) * seed_count * 5 / 3)

    for dump_file in nlp_dump_lists:
        for index,i in enumerate(patches_list):
            if "_patches=" + str(i) in dump_file:
                dumps_list[index].append(dump_file)
    N = 50
    f, axarr = plt.subplots(1, 1, sharex=False, squeeze=False)
    max_len = 0

    #vanilla
    for index, i in enumerate(patches_list):
        if dumps_list[index] != []:
            print(i)
            min_vec_len = min([len(np.load(dump_file)) for dump_file in dumps_list[index]])
            max_vec_len = max([len(np.load(dump_file)) for dump_file in dumps_list[index]])
            for dump_file in dumps_list[index]:
                vec = np.load(dump_file)
                if len(vec) > 0:
                    numpy_dumps.append(vec[:min_vec_len])
            if max_vec_len > 0:
                ys = np.array(numpy_dumps)
                #print(len(np.mean(ys, axis=0)))
                ymean = np.mean(ys, axis=0)
                if smooth:
                    ymean = moving_average(ymean, N)
                ystd = np.std(ys, axis=0)
                ystderr = ystd / np.sqrt(len(ys))
                ystderr = ystderr[:-N+1]
                results_x_axis = steps_per_epoch * np.arange(0, len(ymean))
                l, = axarr[0][0].plot(results_x_axis, ymean, color=COLORS[index], label= str(i) + " patches")
                axarr[0][0].fill_between(results_x_axis, ymean - ystderr, ymean + ystderr, color=COLORS[index], alpha=.4)
                numpy_dumps = []

    font_size = 17
    plt.rcParams.update({'font.size': 15})
    plt.title(algo.upper(),fontdict={"size": font_size})
    plt.grid()
    if algo == "dqn":
        plt.xlim(0,20000)
    elif algo == "ppo":
        plt.legend(loc="upper left")
        plt.xlim(0,55000)
    plt.xlabel("Interactions", fontdict={"size":font_size})
    plt.ylabel("Reward", fontdict={"size":font_size})
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    #plt.yscale("log")
    plt.savefig(
        os.path.join("plots", algo + "_patches_results_plot_nlp" + scenario + "_" + datetime.now().strftime("%d_%m_%H_%M")))
    plt.show(block=False)
