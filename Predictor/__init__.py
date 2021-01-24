
from GlobalData import GlobalData as gd

import copy
import pandas as pd

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


class Predictor:
    def __init__(self, cols):
        self.model = self.build_model([len(cols)-1])

    def get_df_stats(self, x):
        x_stats = x.describe()
        x_stats = x_stats.transpose()
        return x_stats

    def normalize_data(self, x, x_s, cols):
        if not x_s.empty:
            tmp = copy.deepcopy(x)
            tmp_stats = self.get_df_stats(x_s)
            for col in cols:
                tmp[col] = (tmp[col] - tmp_stats['mean'][col]) / \
                    tmp_stats['std'][col]
            return tmp
        return x

    def filter_data(self, x, to_normalize_cols):
        x = x.drop_duplicates()
        x = x.apply(pd.to_numeric, errors='coerce', axis=1)
        tmp = self.normalize_data(x, x, to_normalize_cols)
        return tmp

    def build_model(self, input_shape):
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=input_shape),
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['mae', 'mse'])
        return model

    def train_model(self, dataset, patience=20, verbose=True):
        X = self.filter_data(dataset, gd.to_normalize_cols)
        y = X.pop('H')

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience)

        early_history = self.model.fit(
            X,
            y,
            epochs=1000,
            validation_split=0.2,
            verbose=verbose,
            callbacks=[early_stop]
        )

        return early_history

    def save_model(self):
        self.model.save('data/depth'+str(gd.heuristic_depth)+'/model')

    def load_model(self):
        self.model = keras.models.load_model(
            'data/depth'+str(gd.heuristic_depth)+'/model')

    def predict(self, data):
        return self.model.predict(data)
