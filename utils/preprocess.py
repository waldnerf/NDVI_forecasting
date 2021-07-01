import os
import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler

sns.set_style("darkgrid")
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def prepare_data(data_path, test_year, ts=19, ts_length=36, n_skip=2, n_discard=0, savefig=False):
    df = pd.read_csv(data_path)
    df = df.loc[df['variable_name'] == 'NDVI', :].copy()

    Xs = []
    ys = []
    afis = []
    years = []
    aucs = []
    auc_dist = []
    soss = []
    eoss = []
    for i, row in df.iterrows():
        if row[6] < 27:
            sos = row[6] + 9  # TS starts on OCt 1
        else:
            sos = row[6] - 27  # TS starts on OCt 1
        sos = sos + ((n_skip - n_discard) * ts_length)
        if row[7] < 27:
            eos = row[7] + 9
        else:
            eos = row[7] - 27
        eos = eos + ((n_skip - n_discard) * ts_length)
        if sos > eos:
            sos = sos - ts_length
        # print(f"row[6]:   {row[6]}  row[7]:  {row[7]} sos: {sos}    eos: {eos} ")

        data = row[13:]
        si = 0
        auci = []
        for ei in range((n_skip * ts_length) + ts_length, data.shape[0], ts_length):
            datai = data[si:ei]
            si += ts_length
            Xs.append(datai[(n_discard * ts_length):(n_skip * ts_length) + ts].values)
            ys.append(datai[(n_skip * ts_length) + ts::].values)

            afis.append(row.afi)
            yeari = int(datai.index[-1][0:4])
            years.append(yeari)

            # compute cumulative NDVI
            ndvi = np.concatenate([datai[(n_discard * ts_length):(n_skip * ts_length) + ts].values,
                                   datai[(n_skip * ts_length) + ts::].values], axis=0)
            aucs.append(np.sum(ndvi[sos:eos + 1]))
            auci.append(np.sum(ndvi[sos:eos + 1]))
            eoss.append(eos)
            soss.append(sos)

        for k in range(0, len(auci)):
            auc_dist.append(np.delete(auci, k))

        if (savefig is True) & (i % 10 == 0):
            fig, ax = plt.subplots()
            ax.plot(ndvi)
            ax.vlines(x=sos, ymin=0, ymax=1, linewidth=2, color='g')
            ax.vlines(x=eos, ymin=0, ymax=1, linewidth=2, color='r')
            fig.savefig(f'./figures/ts_{i}.png')
            plt.close()

    Xs = np.stack(Xs, axis=0)
    ys = np.stack(ys, axis=0)
    afis = np.stack(afis, axis=0)
    years = np.stack(years, axis=0)
    aucs = np.stack(aucs, axis=0)
    auc_dist = np.stack(auc_dist, axis=0)
    eoss = np.stack(eoss, axis=0)
    soss = np.stack(soss, axis=0)

    # --- Split data by year
    np.random.seed(0)
    calval_years = np.setdiff1d(np.unique(years), test_year)
    val_years = np.random.choice(calval_years, size=2, replace=False)
    idx_train = ~np.isin(years, np.concatenate([np.array([test_year]), val_years]))
    idx_val = np.isin(years, val_years)
    idx_test = np.isin(years, test_year)

    X_train = Xs[idx_train, ]
    X_val = Xs[idx_val, ]
    X_test = Xs[idx_test, ]

    y_train = ys[idx_train, ]
    y_val = ys[idx_val, ]
    y_test = ys[idx_test, ]

    afi_train = afis[idx_train,]
    afi_val = afis[idx_val,]
    afi_test = afis[idx_test,]

    aucs_test = aucs[idx_test, ]
    auc_dist_test = auc_dist[idx_test, ]
    sos_test = soss[idx_test,]
    eos_test = eoss[idx_test,]


    return {'X': X_train, 'y': y_train, 'afi': afi_train}, \
           {'X': X_val, 'y': y_val, 'afi': afi_val}, \
           {'X': X_test, 'y': y_test, 'afi': afi_test, 'auc': aucs_test, 'auc_dist': auc_dist_test,
            'sos': sos_test, 'eos': eos_test}


if __name__ == "__main__":
    data_path = './data/afi_bins_1000_random_Algeria.csv'
    train_step = 19  # how many NDVI data points for training, start from 19, max 37

    data_train, data_val, data_test = prepare_data(data_path, test_year=2005,
                                                   ts=train_step, ts_length=36, n_skip=2, n_discard=0)

    print(data_train['X'].shape)
    print(data_train['y'].shape)
    print(data_train['afi'].shape)

    print(data_val['X'].shape)
    print(data_val['y'].shape)
    print(data_val['afi'].shape)

    print(data_test['X'].shape)
    print(data_test['y'].shape)
    print(data_test['afi'].shape)
    print(data_test['auc'].shape)
    print(data_test['auc_dist'].shape)

    # print(train_mean_rain)
    # print(train_std_rain)
    # print(train_mean_temp)
    # print(train_std_temp)

    # #######plot one sample###########

    # x = np.arange(train_step)

    # print(train_inputs[0,:,1])

    # f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1)
    # ax1.plot(x, train_inputs[0,:,0],label='NDVI')
    # ax2.plot(x, train_inputs[0,:,1],label='Scaled Rain')
    # ax3.plot(x, train_inputs[0,:,2],label='Scaled MaxT')
    # ax4.plot(x, train_inputs[0,:,3],label='Scaled MinT')
    # ax5.plot(x, train_inputs[0,:,4],label='Temp_SIN')
    # ax6.plot(x, train_inputs[0,:,5],label='Temp_COS')

    # ax1.set_xticks(x)
    # ax2.set_xticks(x)
    # ax3.set_xticks(x)
    # ax4.set_xticks(x)
    # ax5.set_xticks(x)
    # ax6.set_xticks(x)

    # ax1.legend()
    # ax2.legend()
    # ax3.legend()
    # ax4.legend()
    # ax5.legend()
    # ax6.legend()

    # plt.show()
