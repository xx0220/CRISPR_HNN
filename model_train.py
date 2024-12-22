import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from model import CRISPR_HNN
import tensorflow as tf
from scipy.stats import spearmanr, pearsonr
import os
import random
import scipy as sp
import pickle
import time


def set_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data_kf(x1, y):
    train_test_data = []
    kf = ShuffleSplit(n_splits=5, test_size=0.2, random_state=33)
    for train_index, test_index in kf.split(x1):
        x1_train, x1_test = x1[train_index], x1[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_test_data.append((x1_train, x1_test, y_train, y_test))
    return train_test_data


if __name__ == '__main__':
    set_random_seeds(2024)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    dataList = ['ESP', 'HELA', 'HF', 'HL60', 'Sniper-Cas9', 'SpCas9-NG', 'xCas']  # 数据集名称
    for dataname in dataList:
        batch_size = 16
        epochs = 200
        input_path = f'./datasets/{dataname}.pkl'
        with open(input_path, 'rb') as f:

            X_onehot, y = pickle.load(f)

        X_onehot = X_onehot.reshape(len(X_onehot), 23, 4)
        data_list = load_data_kf(X_onehot, y)
        k = 1
        results_df = pd.DataFrame(columns=['Fold', 'SCC', 'PCC'])

        for data in data_list:
            print('k-fold:', k)
            x1_train, x1_test, y_train, y_test = data[0], data[1], data[2], data[3]
            model = CRISPR-HNN()
            modelname = model.name
            print(modelname)

            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)

            adamax = tf.keras.optimizers.Adamax(learning_rate=0.0001)
            model.compile(loss='mean_absolute_error', optimizer=adamax, metrics=['mae'])
            print(x1_train[0])
            history = model.fit(x1_train, y_train, batch_size=batch_size,
                                epochs=epochs, validation_split=0.1,
                                shuffle=False, callbacks=[early_stop, lr_schedule])

            y_test_pred = model.predict(x1_test)
            y_test_pred = np.array(y_test_pred, dtype='float64').squeeze()
            y_test = np.array(y_test, dtype='float64').squeeze()
            y_test_pred = np.squeeze(y_test_pred)

            y_test = np.nan_to_num(y_test, nan=np.nanmean(y_test), posinf=np.max(y_test[np.isfinite(y_test)]),
                                   neginf=np.min(y_test[np.isfinite(y_test)]))
            y_test_pred = np.nan_to_num(y_test_pred, nan=np.nanmean(y_test_pred),
                                        posinf=np.max(y_test_pred[np.isfinite(y_test_pred)]),
                                        neginf=np.min(y_test_pred[np.isfinite(y_test_pred)]))

            y_test_pred = y_test_pred.flatten()

            scc = sp.stats.spearmanr(y_test, y_test_pred)[0]
            pcc = sp.stats.pearsonr(y_test, y_test_pred)[0]
            results_df = results_df.append({'Fold': k, 'SCC': scc, 'PCC': pcc}, ignore_index=True)

            print(f'Fold {k}: SCC = {scc:.4f}, PCC = {pcc:.4f}')

            k += 1
        average_scc = round(results_df['SCC'].mean(), 4)
        average_pcc = round(results_df['PCC'].mean(), 4)
        current_time = time.strftime('%Y_%m_%d_%H_%M_%S')
        results_df = results_df.append({'Fold': 'Average', 'SCC': average_scc, 'PCC': average_pcc}, ignore_index=True)
        results_df.to_csv(f'./Log/CRISPR_HNN/{current_time}_{batch_size}CRISPR_HNN{dataname}.csv')
        print(results_df)
