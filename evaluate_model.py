import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

df = pd.read_csv('./data/forecasts_vRNN.csv', index_col=0)
df['forecast_length'] = 360 - 10 * df['step']

df_summary = df.groupby('forecast_length', as_index=False).agg(mse_m=pd.NamedAgg(column="mse", aggfunc="mean"),
                                                    recall_normal_m=pd.NamedAgg(column="recall_normal", aggfunc="mean"),
                                                    recall_critical_m=pd.NamedAgg(column="recall_critical",
                                                                                  aggfunc="mean"),
                                                    precision_normal_m=pd.NamedAgg(column="precision_normal",
                                                                                   aggfunc="mean"),
                                                    precision_critical_m=pd.NamedAgg(column="precision_critical",
                                                                                     aggfunc="mean"),
                                                    mse_sd=pd.NamedAgg(column="mse", aggfunc="std"),
                                                    recall_normal_sd=pd.NamedAgg(column="recall_normal", aggfunc="std"),
                                                    recall_critical_sd=pd.NamedAgg(column="recall_critical",
                                                                                   aggfunc="std"),
                                                    precision_normal_sd=pd.NamedAgg(column="precision_normal",
                                                                                    aggfunc="std"),
                                                    precision_critical_sd=pd.NamedAgg(column="precision_critical",
                                                                                      aggfunc="std")
                                                    )

_figsize = (15, 5)

variables = ['RMSE', 'Recall', 'Precision']
x_label = 'Forecast length (days)'

fig, axs = plt.subplots(1, len(variables), figsize=_figsize)

ax = axs[0]
plt.sca(ax)
plt.plot(df_summary['forecast_length'], np.sqrt(df_summary['mse_m']), color='#D1E8E2')
plt.fill_between(df_summary['forecast_length'],
                 np.sqrt(df_summary['mse_m'] + df_summary['mse_sd']),
                 np.sqrt(df_summary['mse_m'] - df_summary['mse_sd']), color='#D1E8E2', alpha=0.5)
plt.xlim([df_summary['forecast_length'].max(), df_summary['forecast_length'].min()])
plt.title(variables[0])
plt.xlabel(x_label)
plt.ylabel('RMSE')

ax = axs[1]
plt.sca(ax)
plt.plot(df_summary['forecast_length'], df_summary['recall_normal_m'], color='#2C3531', label='Non-critical')
plt.fill_between(df_summary['forecast_length'],
                 df_summary['recall_normal_m'] + df_summary['recall_normal_sd'],
                 df_summary['recall_normal_m'] - df_summary['recall_normal_sd'], color='#2C3531', alpha=0.5)
plt.plot(df_summary['forecast_length'], df_summary['recall_critical_m'], color='#116466', label='Critical')
plt.fill_between(df_summary['forecast_length'],
                 df_summary['recall_critical_m'] + df_summary['recall_critical_sd'],
                 df_summary['recall_critical_m'] - df_summary['recall_critical_sd'], color='#116466', alpha=0.5)
plt.ylim([0, 1.05])
plt.xlim([df_summary['forecast_length'].max(), df_summary['forecast_length'].min()])
plt.legend()
plt.xlabel(x_label)
plt.ylabel('Recall')
plt.title(variables[1])

ax = axs[2]
plt.sca(ax)
plt.plot(df_summary['forecast_length'], df_summary['precision_normal_m'], color='#2C3531', label='Non-critical')
plt.fill_between(df_summary['forecast_length'],
                 df_summary['precision_normal_m'] + df_summary['precision_normal_sd'],
                 df_summary['precision_normal_m'] - df_summary['precision_normal_sd'], color='#2C3531', alpha=0.5)
plt.plot(df_summary['forecast_length'], df_summary['precision_critical_m'], color='#116466', label='Critical')
plt.fill_between(df_summary['forecast_length'],
                 df_summary['precision_critical_m'] + df_summary['precision_critical_sd'],
                 df_summary['precision_critical_m'] - df_summary['precision_critical_sd'], color='#116466', alpha=0.5)
plt.xlim([df_summary['forecast_length'].max(), df_summary['forecast_length'].min()])
plt.ylim([0, 1.05])
plt.legend()
plt.xlabel(x_label)
plt.ylabel('Precision')
plt.title(variables[2])
plt.show()

cmaps = ['Greens', 'Blues', 'Purples', 'Reds']
for col in range(len(cmaps)):
    ax = axs[col]
    plt.sca(ax)
    pcm = ax.imshow(np.flipud(hist[:, :, col]), cmap=cmaps[col])
    fig.colorbar(pcm, ax=ax)
    plt.title(variables[col])
plt.suptitle(title)
plt.tight_layout()

def f(x):
    prec_scores = precision_score(x['anomaly_true'], x['anomaly_pred'], average=None)
    recall_scores = recall_score(x['anomaly_true'], x['anomaly_pred'], average=None)

    d = {}
    d['mse'] = x['mse'].mean()
    d['recall_normal_f'] = recall_scores[0]
    d['recall_critical_f'] = recall_scores[1]
    d['precision_normal_f'] = prec_scores[0]
    d['precision_critical_f'] = prec_scores[1]

    prec_scores = precision_score(x['anomaly_true'], x['anomaly_nopred'], average=None)
    recall_scores = recall_score(x['anomaly_true'], x['anomaly_nopred'], average=None)
    d['recall_normal_nf'] = recall_scores[0]
    d['recall_critical_nf'] = recall_scores[1]
    d['precision_normal_nf'] = prec_scores[0]
    d['precision_critical_nf'] = prec_scores[1]

    return pd.Series(d, index=['mse',
                               'recall_normal_f', 'recall_critical_f', 'precision_normal_f', 'precision_critical_f',
                               'recall_normal_nf', 'recall_critical_nf', 'precision_normal_nf', 'precision_critical_nf'
                               ])



df = pd.read_csv('./data/season_accuracy_vRNN.csv')
df['progress'] = np.round((df['tos'] - df['sos']) / (df['eos'] - df['sos']), 1)
df.loc[df.tos > df.eos, 'progress'] = -1
df.loc[df.tos < df.sos, 'progress'] = -1
df = df.loc[df.progress != -1, :].copy()

df_summary = df.groupby('progress').apply(f)
df_summary.reset_index(inplace=True)

_figsize = (15, 5)

variables = ['RMSE', 'Recall', 'Precision']
x_label = 'Season progress (%)'

fig, axs = plt.subplots(1, len(variables), figsize=_figsize)

ax = axs[0]
plt.sca(ax)
plt.plot(df_summary['progress'], np.sqrt(df_summary['mse']), color='#2C3531')
plt.xlim([0, 1])
plt.title(variables[0])
plt.xlabel(x_label)
plt.ylabel('RMSE')

ax = axs[1]
plt.sca(ax)
plt.plot(df_summary['progress'], df_summary['recall_normal_f'], color='#2C3531', label='Non-critical')
plt.plot(df_summary['progress'], df_summary['recall_critical_f'], color='#116466', label='Critical')
plt.plot(df_summary['progress'], df_summary['recall_normal_nf'], '--', color='#2C3531', label='No forecast, Non-critical')
plt.plot(df_summary['progress'], df_summary['recall_critical_nf'], '--', color='#116466', label='No forecast, Critical')

plt.ylim([0, 1.05])
plt.xlim([0, 1])
plt.legend()
plt.xlabel(x_label)
plt.ylabel('Recall')
plt.title(variables[1])

ax = axs[2]
plt.sca(ax)
plt.plot(df_summary['progress'], df_summary['precision_normal_f'], color='#2C3531', label='Non-critical')
plt.plot(df_summary['progress'], df_summary['precision_critical_f'], color='#116466', label='Critical')
plt.plot(df_summary['progress'], df_summary['precision_normal_nf'], '--', color='#2C3531', label='No forecast, Non-critical')
plt.plot(df_summary['progress'], df_summary['precision_critical_nf'], '--', color='#116466', label='No forecast, Critical')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.legend()
plt.xlabel(x_label)
plt.ylabel('Precision')
plt.title(variables[2])
plt.show()
