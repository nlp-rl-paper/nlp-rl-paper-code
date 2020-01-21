import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

class PlotGenerator:
    def __init__(self, algo ,scenario : str,model_type : str, block : bool, frame_repeat, seed : int, patches_count=-1):
        self.total_train_episodes_vec = []
        self.mean_train_reward_per_epoch_vec = []
        self.mean_test_reward_per_epoch_vec = []
        self.total_interactions_per_epoch = []
        self.algo = algo
        self.scenario = scenario
        self.model_type = model_type
        self.block = block
        self.seed = int(seed)
        self.fig : plt.figure = None
        self.frame_repeat = str(frame_repeat)
        self.patches_count = patches_count

    def update_reward_data(self,train_reward,test_reward):
        self.mean_train_reward_per_epoch_vec.append(train_reward)
        self.mean_test_reward_per_epoch_vec.append(test_reward)


    def update_episodes_data(self,episode_count):
        self.total_train_episodes_vec.append(episode_count)


    def update_interactions_data(self,episode_count):
        self.total_interactions_per_epoch.append(episode_count)


    def generate_plot_name(self):
        if self.model_type == "TextCNN":
            return self.algo + "_" + self.model_type + "_" + "_patches=" + str(self.patches_count) + "_"+ self.scenario + "_" + datetime.now().strftime("%d_%m_%H_%M") + "_fr=" + self.frame_repeat + "_seed=" + str(self.seed)
        return self.algo + "_" + self.model_type + "_" + self.scenario + "_" + datetime.now().strftime("%d_%m_%H_%M") + "_fr=" + self.frame_repeat + "_seed=" + str(self.seed)

    def plot_reward_progress(self):
        data_folder = self.algo + "_" + self.scenario
        if self.model_type == "TextCNN":
            data_folder += "_nlp"
        elif self.model_type == "CONVNET":
            data_folder += "_viz"
        elif self.model_type == "random":
            data_folder += "_ran"
        elif self.model_type == "CONVNET_SEG":
            data_folder += "_seg"
        else:
            data_folder += "_vec"
        self.fig = plt.figure()
        plt.title(self.generate_plot_name())
        plt.plot(self.total_interactions_per_epoch,self.mean_train_reward_per_epoch_vec)
        plt.plot(self.total_interactions_per_epoch,self.mean_test_reward_per_epoch_vec)
        plt.legend(["mean train reward", "mean test reward"])
        plt.xlabel("learning steps")
        plt.ylabel("reward")
        plt.show(block=self.block)
        fig_name = self.generate_plot_name() + "_convergence.png"
        fig_name = os.path.join("plots",data_folder,fig_name)
        self.fig.savefig(fig_name)


    def plot_episodes_prograss(self):
        plt.figure()
        plt.title(self.generate_plot_name())
        plt.plot(self.total_train_episodes_vec)
        plt.legend("total_train_episodes")
        plt.xlabel("epoch")
        plt.ylabel("episodes")
        plt.show(block=self.block)
        fig_name = self.generate_plot_name() + "_convergence.png"
        plt.imsave(fig_name)



    def dump_data(self):
        dump_name = self.generate_plot_name() + "_mean_test_data"
        data_folder = self.scenario
        if self.model_type == "TextCNN":
            data_folder += "_nlp"
        elif self.model_type == "CONVNET":
            data_folder += "_viz"
        elif self.model_type == "random":
            data_folder += "_ran"
        elif self.model_type == "CONVNET_SEG":
            data_folder += "_seg"
        else:
            data_folder += "_vec"
        dump_file_name = os.path.join("data",self.algo + "_" + data_folder,dump_name)
        np.array(self.mean_test_reward_per_epoch_vec).dump(dump_file_name)



#returns 5 last runs
def get_dump_list(scenario,algo,rep_type,seed_count):
    try:
        dir_list = os.listdir(os.path.join("data",algo + "_" + scenario + "_" + rep_type))
        if len(dir_list) > 0:
            dir_list = [os.path.join("data",algo + "_" + scenario + "_" + rep_type, file) for file in dir_list]
            dir_list.sort(key=lambda x: os.path.getmtime(x),reverse=True)
            return dir_list[:seed_count]
        return None
    except:
        return None

def final_plot(algo, scenario="basic",steps_per_epoch=250,seed_count=5,epochs=100,smooth=False):
    print(scenario)

    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    numpy_dumps_viz = []
    numpy_dumps_nlp = []
    numpy_dumps_vec = []
    numpy_dumps_seg = []
    numpy_dumps_ran = []

    nlp_dump_list = get_dump_list(scenario, algo, "nlp",seed_count)
    viz_dump_list = get_dump_list(scenario, algo, "viz",seed_count)
    vec_dump_list = get_dump_list(scenario, algo, "vec",seed_count)
    seg_dump_list = get_dump_list(scenario, algo, "seg",seed_count)
    ran_dump_list = get_dump_list(scenario, algo, "ran", seed_count)

    COLORS = ['#66624b', '#41c1dd', '#3e9651', '#e0771a', '#cc2529', '#b2a745', '#660066']

    N = 20
    f, axarr = plt.subplots(1, 1, sharex=False, squeeze=False)

    max_len = 0

    # nlp
    if nlp_dump_list is not None:
        min_vec_len = min([len(np.load(dump_file)) for dump_file in nlp_dump_list])
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
        l, = axarr[0][0].plot(results_x_axis, ymean, color=COLORS[4], label="nlp")
        axarr[0][0].fill_between(results_x_axis, ymean - ystderr, ymean + ystderr, color=COLORS[4], alpha=.4)
        max_len = len(ymean)


    # ran
    # if ran_dump_list is not None:
    #     min_vec_len = min([len(np.load(dump_file)) for dump_file in ran_dump_list])
    #     for dump_file in ran_dump_list:
    #         vec = np.load(dump_file)
    #         if len(vec) > 0:
    #             numpy_dumps_ran.append(vec[:min_vec_len])
    #     ys = numpy_dumps_ran
    #     ymean = np.mean(ys, axis=0)
    #     ymean = ymean[:max_len]
    #     if smooth:
    #         ymean = moving_average(ymean,N)
    #     ystd = np.std(ys, axis=0)[:max_len]
    #     ystderr = ystd / np.sqrt(len(ys))
    #     ystderr = ystderr[:-N+1]
    #     results_x_axis = steps_per_epoch * np.arange(0, len(ymean))
    #     l, = axarr[0][0].plot(results_x_axis, ymean, color=COLORS[0], label="random")
    #     axarr[0][0].fill_between(results_x_axis, ymean - ystderr, ymean + ystderr, color=COLORS[0], alpha=.4)




    #vec
    if vec_dump_list is not None:
        min_vec_len = min([len(np.load(dump_file)) for dump_file in vec_dump_list])
        for dump_file in vec_dump_list:
            vec = np.load(dump_file)
            if len(vec) > 0:
                numpy_dumps_vec.append(vec[:min_vec_len])

        ys = numpy_dumps_vec
        ymean = np.mean(ys, axis=0)
        if smooth:
            ymean = moving_average(ymean, N)
        ystd = np.std(ys, axis=0)
        ystderr = ystd / np.sqrt(len(ys))
        ystderr = ystderr[:-N+1]
        results_x_axis = steps_per_epoch * np.arange(0, len(ymean))
        l, = axarr[0][0].plot(results_x_axis, ymean, color=COLORS[1], label="vector")
        axarr[0][0].fill_between(results_x_axis, ymean - ystderr, ymean + ystderr, color=COLORS[1], alpha=.4)


    # seg
    if seg_dump_list is not None:
        min_vec_len = min([len(np.load(dump_file)) for dump_file in seg_dump_list])
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
        l, = axarr[0][0].plot(results_x_axis, ymean, color=COLORS[3], label="segments")
        axarr[0][0].fill_between(results_x_axis, ymean - ystderr, ymean + ystderr, color=COLORS[3], alpha=.4)



    #viz
    if viz_dump_list is not None:
        min_vec_len = min([len(np.load(dump_file)) for dump_file in viz_dump_list])
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
        l, = axarr[0][0].plot(results_x_axis, ymean, color=COLORS[5], label="vision")
        axarr[0][0].fill_between(results_x_axis, ymean - ystderr, ymean + ystderr, color=COLORS[5], alpha=.4)


    plt.legend()
    plt.title(algo + ": " + str(scenario))
    plt.grid()
    if algo == "dqn":
        plt.xlim(0,epochs*steps_per_epoch)
    plt.xlabel("Interactions")
    plt.ylabel("Reward")
    plt.savefig(os.path.join("plots",algo + "_final_plot_" + scenario + "_" + datetime.now().strftime("%d_%m_%H_%M")))
    plt.show(block=False)


