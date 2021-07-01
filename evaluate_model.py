import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
