import numpy as np
import matplotlib.pyplot as plt


###### plot results ##########
def plot_results(outputs, targets, inputs, i, filename=''):

    x = np.arange(inputs.shape[1]+outputs.shape[1])
    split = inputs.shape[1]-1

    output = outputs[i]
    target = targets[i]
    input_x = inputs[i]

    input_ori = np.concatenate((input_x, output), axis=0)
    input_tar = np.concatenate((input_x, target), axis=0)

    f, ax = plt.subplots(1, 1)
    ax.plot(x, input_ori, color='green', label='predicted')
    ax.plot(x, input_tar, color='red', label='target')
    ax.axvline(split, color='cyan')
    ax.legend()
    plt.show()
    ax.set(ylim=(0, 1))

    if filename != '':
        plt.savefig(filename)
        plt.close()

import pandas as pd
import seaborn as sns

#df = pd.read_csv('./data/aLSTM.csv', index_col=0)
#df_plot = df.pivot("crop", "step", "mse")
#ax = sns.heatmap(df_plot)