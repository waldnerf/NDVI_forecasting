from sklearn.metrics import classification_report, f1_score, cohen_kappa_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from catalyst.utils import set_global_seed
from catalyst import dl
from catalyst.dl import AccuracyCallback, AUCCallback, EarlyStoppingCallback, CriterionCallback

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import random
import math
import os
import time
import statistics

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision

from collections import OrderedDict
import os
import argparse
import time
import numpy as np
import pandas as pd
from itertools import product
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

sns.set_style("darkgrid")
mpl.use('tkagg')
np.set_printoptions(threshold=np.inf)
pd.options.display.width = 0

# reproduce
SEED = 9527
set_global_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utils.utils import *
from utils.plot import *
from nn.models.asLTSM import *
import utils.constants as cst
from nn.models.vanilla_rnn import *
from nn.models.convForecastNet import *
import utils.constants as cst
from utils.preprocess import *
from utils.anomaly import *


class CustomRunner_vRNN(dl.Runner):
    def handle_batch(self, batch):
        x, y = batch
        x = torch.unsqueeze(x, 2)
        y = torch.unsqueeze(y, 2)
        # x = x.permute(0, 2, 1)
        # y = y.permute(0, 2, 1)
        # s = s.permute(1, 0, 2)

        if self.is_train_loader:
            outputs = self.model(x)  # , 0.5)
        else:
            outputs = self.model(x)  # , 0)

        loss = F.mse_loss(outputs, y)
        # output_numpy = output.cpu().data.numpy()
        # y_numpy = y.cpu().data.numpy()

        # self.batch_metrics = {
        #     "loss": loss
        # }

        self.batch = {
            "outputs": outputs,
            "preds": y,
        }


class CustomRunner_ForecastNet(dl.Runner):
    def handle_batch(self, batch):
        x, y = batch
        x = torch.unsqueeze(x, 2)
        y = torch.unsqueeze(y, 2)
        x = x.permute(1, 0, 2)
        in_seq_length, batch_size, input_dim = x.shape
        x_modified = torch.reshape(x, (batch_size, -1))
        y = y.permute(1, 0, 2)

        if self.is_train_loader:
            outputs = self.model(x_modified, y, 0.5)
        else:
            outputs = self.model(x_modified)

        loss = F.mse_loss(outputs, y)
        # output_numpy = output.cpu().data.numpy()
        # y_numpy = y.cpu().data.numpy()

        # self.batch_metrics = {
        #     "loss": loss
        # }

        self.batch = {
            "outputs": outputs,
            "preds": y,
        }


if __name__ == "__main__":
    # DataLoader definition
    # model hyperparameters
    INPUT_DIM = 7
    OUTPUT_DIM = 1
    SUPPORT_DIM = INPUT_DIM - OUTPUT_DIM
    ENC_HID_DIM = 64
    DEC_HID_DIM = 64
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    ECN_Layers = 2
    DEC_Layers = 2
    LR = 0.005  # learning rate
    EPOCHS = 20
    BATCH_SIZE = 128
    BATCH_SIZE_INF = 1
    FIG_FLAG = False
    MONITOR = True
    MODEL = 'vRNN' #'ForecastNet'  #'vRNN'
    params = cst.params[MODEL]
    
    out_filename = f'./data/forecasts_{MODEL}.csv'
    season_filename = f'./data/season_accuracy_{MODEL}.csv'
    df_out = pd.DataFrame(list(product(range(3, 36, 3), range(2004, 2017))), columns=['step', 'year'])

    df_out['mse'] = np.nan
    df_out['recall_normal_f'] = np.nan
    df_out['recall_critical_f'] = np.nan
    df_out['precision_normal_f'] = np.nan
    df_out['precision_critical_f'] = np.nan
    df_out['recall_normal_nf'] = np.nan
    df_out['recall_critical_nf'] = np.nan
    df_out['precision_normal_nf'] = np.nan
    df_out['precision_critical_nf'] = np.nan

    data_path = './data/afi_bins_2000_random_Algeria.csv'

    season_list = []
    for index, row in df_out.iterrows():
        train_step = int(row['step'])
        test_year = int(row['year'])
        logdir = f'./log/log_{MODEL}_{train_step}'

        ########### read data ###############
        data_train, data_val, data_test = prepare_data(data_path, test_year=int(test_year), ts=train_step,
                                                       ts_length=36, n_skip=2, n_discard=0)

        #### Numpy to Tensor ########
        train_inputs = numpy_to_tensor(data_train['X'].astype(np.float32), torch.FloatTensor)
        train_target_single = numpy_to_tensor(data_train['y'].astype(np.float32), torch.FloatTensor)

        val_inputs = numpy_to_tensor(data_val['X'].astype(np.float32), torch.FloatTensor)
        val_target_single = numpy_to_tensor(data_val['y'].astype(np.float32), torch.FloatTensor)

        test_inputs = numpy_to_tensor(data_test['X'].astype(np.float32), torch.FloatTensor)
        test_target_single = numpy_to_tensor(data_test['y'].astype(np.float32), torch.FloatTensor)

        print('train_inputs: {}'.format(train_inputs.shape))
        print('train_target_single: {}'.format(train_target_single.shape))

        print('val_inputs: {}'.format(val_inputs.shape))
        print('val_target_single: {}'.format(val_target_single.shape))

        print('test_inputs: {}'.format(test_inputs.shape))
        print('test_target_single: {}'.format(test_target_single.shape))

        #########################################
        train_dataset = TensorDataset(train_inputs, train_target_single)
        valid_dataset = TensorDataset(val_inputs, val_target_single)
        test_dataset = TensorDataset(test_inputs, test_target_single)

        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                               drop_last=True, shuffle=True,
                                               num_workers=0)

        valid_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                                               drop_last=True, shuffle=False,
                                               num_workers=0)

        # Catalyst loader:

        loaders = OrderedDict()
        loaders["train"] = train_dl
        loaders["valid"] = valid_dl

        # model, criterion, optimizer, scheduler
        # model, criterion, optimizer, scheduler
        if MODEL == 'vRNN':
            #encoder = EncoderRNN(params['INPUT_DIM'], params['ENC_HID_DIM'], params['ECN_Layers'])
            #decoder = DecoderRNN(params['INPUT_DIM'], params['DEC_HID_DIM'], params['DEC_Layers'],
            #                    params['FC_Units'], params['OUTPUT_DIM'])
            model = VanillaRNN(params['INPUT_DIM'], params['ENC_HID_DIM'], params['ENC_Layers'],
                               params['DEC_HID_DIM'], params['DEC_Layers'],
                               params['FC_Units'], params['OUTPUT_DIM'], train_target_single.shape[1],
                               'GRU', 0.2, True, device)
            runner = CustomRunner_vRNN()
        elif MODEL == 'ForecastNet':
            model = ForecastNetConvModel2(input_dim=params['INPUT_DIM'], hidden_dim=params['ENC_HID_DIM'],
                                          output_dim=params['OUTPUT_DIM'], in_seq_length=train_inputs.shape[1],
                                          out_seq_length=train_target_single.shape[1], device=device)
            runner = CustomRunner_ForecastNet()

        model.apply(init_weights)

        print(model)
        print(f'The model has {count_parameters(model):,} trainable parameters')
        
        # Magic
        if MONITOR:
            wandb.watch(model)
        
        # AdamW
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        criterion = nn.MSELoss()  # DilateLoss(device=device) #

        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=EPOCHS,
            loaders=loaders,
            logdir=logdir,
            verbose=True,
            timeit=True,
            callbacks=[
                dl.CriterionCallback(metric_key="loss", input_key="outputs", target_key="preds"),
                dl.OptimizerCallback(metric_key="loss"),
                dl.SchedulerCallback(),
                dl.CheckpointCallback(
                    logdir=logdir,
                    loader_key="valid", metric_key="loss", minimize=True, save_n_best=1),
            ],
            load_best_on_end=True
        )

        ###################### inference ################
        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE_INF,
                                              drop_last=False, shuffle=False,
                                              num_workers=0)

        loss_list = []
        predicted_y = []
        target_y = []
        input_x = []

        anomaly_obs = []
        anomaly_preds = []
        anomaly_nopreds = []

        sos_list = []
        eos_list = []
        tos_list = []

        # load model and prediction
        # model.load_state_dict(torch.load(
        #    './model/best_full_newdata.pth')["model_state_dict"])

        for i, (x, y) in enumerate(test_dl, 0):
            if MODEL == 'vRNN':
                x = torch.unsqueeze(x, 2)
                y = torch.unsqueeze(y, 2)
                outputs = model(x)  # , y, 0)
            elif MODEL == 'LSTM':
                x = torch.unsqueeze(x, 2)
                y = torch.unsqueeze(y, 2)
                x = x.permute(1, 0, 2)
                y = y.permute(1, 0, 2)
                outputs = model(x, teacher_forcing_ratio=0)
            elif MODEL == 'ForecastNet':
                x = torch.unsqueeze(x, 2)
                y = torch.unsqueeze(y, 2)
                x = x.permute(1, 0, 2)
                in_seq_length, batch_size, input_dim = x.shape
                x_modified = torch.reshape(x, (batch_size, -1))
                y = y.permute(1, 0, 2)

                outputs = model(x_modified)

            original_x = x[:, :, 0]

            loss = F.mse_loss(outputs, y)

            x_numpy = original_x.cpu().data.numpy()
            output_numpy = outputs.cpu().data.numpy()
            y_numpy = y.cpu().data.numpy()

            # Store the loss from the final iteration
            input_x.append(x_numpy)
            predicted_y.append(output_numpy)
            target_y.append(y_numpy)  # Store the corresponding anomaly label
            loss_list.append(loss.item())

            # Compute area under the curve and assess forecast
            ndvi_seq = np.concatenate((np.squeeze(x_numpy), np.squeeze(output_numpy)), axis=0)
            auc = compute_auc(ndvi_seq, data_test['sos'][i], data_test['eos'][i])
            pred_outlook = classify_season(data_test['auc_dist'][i, :], auc)
            nopred_outlook = classify_season(data_test['auc_dist_incomplete'][i, :], data_test['auc_incomplete'][i])
            act_outlook = classify_season(data_test['auc_dist'][i, :], data_test['auc'][i])
            anomaly_obs.append(act_outlook)
            anomaly_preds.append(pred_outlook)
            anomaly_nopreds.append(nopred_outlook)
            sos_list.append(data_test['sos'][i])
            eos_list.append(data_test['eos'][i])
            tos_list.append(np.squeeze(x_numpy).shape[0])

        print(f'| Test MSE Loss: {statistics.mean(loss_list):.4f} ')

        # Format and save results
        df_seas = pd.DataFrame({'sos': sos_list, 'eos': eos_list, 'tos': tos_list,
                                'anomaly_true': anomaly_obs, 'anomaly_pred': anomaly_preds,
                                'anomaly_nopred': anomaly_nopreds, 'mse': loss_list})
        season_list.append(df_seas)

        # Anomaly classification -- With NDVI forecasts
        cm = confusion_matrix(anomaly_obs, anomaly_preds)
        prec_scores = precision_score(anomaly_obs, anomaly_nopreds, average=None)
        recall_scores = recall_score(anomaly_obs, anomaly_preds, average=None)

        df_out.loc[df_out.index == index, 'mse'] = statistics.mean(loss_list)
        df_out.loc[df_out.index == index, 'recall_normal_f'] = recall_scores[0]
        df_out.loc[df_out.index == index, 'recall_critical_f'] = recall_scores[1]
        df_out.loc[df_out.index == index, 'precision_normal_f'] = prec_scores[0]
        df_out.loc[df_out.index == index, 'precision_critical_f'] = prec_scores[1]

        # Anomaly classification -- Without NDVI forecasts
        cm = confusion_matrix(anomaly_obs, anomaly_nopreds)
        prec_scores = precision_score(anomaly_obs, anomaly_nopreds, average=None)
        recall_scores = recall_score(anomaly_obs, anomaly_nopreds, average=None)

        df_out.loc[df_out.index == index, 'recall_normal_nf'] = recall_scores[0]
        df_out.loc[df_out.index == index, 'recall_critical_nf'] = recall_scores[1]
        df_out.loc[df_out.index == index, 'precision_normal_nf'] = prec_scores[0]
        df_out.loc[df_out.index == index, 'precision_critical_nf'] = prec_scores[1]

        ####### plot some test resutls ##########
        inputs_x = np.squeeze(np.array(input_x))  # Input NDVI
        predictions = np.squeeze(np.array(predicted_y))  # predicted NDVI
        targets = np.squeeze(np.array(target_y))  # target NDVI

        if FIG_FLAG:
            for i in range(0, 20):
                plot_results(predictions, targets, inputs_x, i, filename=f'./figures/{MODEL}_{train_step}_{i}.png')
                plt.close()
    df_out.to_csv(out_filename)
    df_seasonal = pd.concat(season_list)
    df_seasonal.to_csv(season_filename, index=False)
