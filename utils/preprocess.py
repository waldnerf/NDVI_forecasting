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
    """
    Extract training, validation and test data from database
    :param data_path: Path to input csv file
    :param test_year: Year to set aside for testing
    :param ts: lentgh of time series of the current season to use of forecasting
    :param ts_length: Length of full time series for forecasting
    :param n_skip: Number of years to skip
    :param n_discard: Number of years to discard
    :param savefig: Save figures
    :return:
    """
    # read data and keep only selected variables
    df = pd.read_csv(data_path)
    df = df.loc[df['variable_name'] == 'NDVI', :].copy()

    # Prepare output data
    Xs = []  # Predictors
    ys = []  # Target variables
    afis = []  # AFI from ASAP
    years = []  # Year
    aucs = []  # Area under the curve at the end of the season
    auc_dist = []  # Distribution of end-of-season areas under the curve
    aucs_incomplete = []  # Area under the curve in season
    auc_dist_incomplete = []  # Distribution of in-season areas under the curve
    soss = []  # Start of season
    eoss = []  # End of season
    for i, row in df.iterrows():
        # Retrieve start and end of season
        # These are constant for all calendar years
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
        auci = []  # End-of-season area under the curve for particular years
        auci_incomplete = []  # In-season area under the curve for particular years

        # Loop through all season in data
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
            aucs.append(np.sum(ndvi[sos:eos]))
            auci.append(np.sum(ndvi[sos:eos]))
            if sos+ts < eos:
                aucs_incomplete.append(np.sum(ndvi[sos:(sos+ts)]))
                auci_incomplete.append(np.sum(ndvi[sos:(sos+ts)]))
            else:
                aucs_incomplete.append(np.sum(ndvi[sos:(eos)]))
                auci_incomplete.append(np.sum(ndvi[sos:(eos)]))
            eoss.append(eos)
            soss.append(sos)

        for k in range(0, len(auci)):
            auc_dist.append(np.delete(auci, k))
            auc_dist_incomplete.append(np.delete(auci_incomplete, k))

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
    aucs_incomplete = np.stack(aucs_incomplete, axis=0)
    auc_dist_incomplete = np.stack(auc_dist_incomplete, axis=0)
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
    aucs_incomplete_test = aucs_incomplete[idx_test, ]
    auc_dist_incomplete_test = auc_dist_incomplete[idx_test, ]
    sos_test = soss[idx_test, ]
    eos_test = eoss[idx_test, ]


    return {'X': X_train, 'y': y_train, 'afi': afi_train}, \
           {'X': X_val, 'y': y_val, 'afi': afi_val}, \
           {'X': X_test, 'y': y_test, 'afi': afi_test,
            'auc': aucs_test, 'auc_dist': auc_dist_test,
            'auc_incomplete': aucs_incomplete_test, 'auc_dist_incomplete': auc_dist_incomplete_test,
            'sos': sos_test, 'eos': eos_test}


if __name__ == "__main__":
    data_path = './data/afi_bins_1000_random_Algeria.csv'
    train_step = 10  # how many NDVI data points for training, start from 19, max 37

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
    print(data_test['auc_incomplete'].shape)
    print(data_test['auc_dist_incomplete'].shape)

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
