from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
import sklearn
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.layers import (
    LSTM,
    SimpleRNN,
    GRU,
    Bidirectional,
    BatchNormalization,
    Conv1D,
    MaxPooling1D,
    Reshape,
    GlobalAveragePooling1D,
)
import sklearn.preprocessing
from sklearn import metrics

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from preprocess import combine_data


y = combine_data["class"]
X = combine_data.drop("class", axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

oos_pred = []
oversample = RandomOverSampler(sampling_strategy="minority")
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# 模型定义
batch_size = 64
model = Sequential()
model.add(
    Conv1D(
        64,
        kernel_size=64,
        padding="same",
        activation="relu",
        input_shape=(X_train.shape[1], 1),
    )
)
model.add(MaxPooling1D(pool_size=10))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Reshape((128, 1), input_shape=(128,)))
model.add(MaxPooling1D(pool_size=10))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(128, return_sequences=False)))
model.add(Dropout(0, 5))
model.add(Dense(10))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


for train_index, test_index in kfold.split(X_train, y_train):
    train_X, test_X = X_train.iloc[train_index], X_train.iloc[test_index]
    train_y, test_y = y_train.iloc[train_index], y_train.iloc[test_index]

    train_X_over, train_y_over = oversample.fit_resample(train_X, train_y)

    x_columns_train = combine_data.columns.drop("class")
    x_train_array = train_X_over[x_columns_train].values
    x_train_1 = np.reshape(
        x_train_array, (x_train_array.shape[0], x_train_array.shape[1], 1)
    )

    dummies = pd.get_dummies(train_y_over)
    outcomes = dummies.columns
    num_classes = len(outcomes)
    y_train_1 = dummies.values

    x_columns_test = combine_data.columns.drop("class")
    x_test_array = test_X[x_columns_test].values
    x_test_2 = np.reshape(
        x_test_array, (x_test_array.shape[0], x_test_array.shape[1], 1)
    )

    dummies_test = pd.get_dummies(test_y)
    outcomes_test = dummies_test.columns
    num_classes = len(outcomes_test)
    y_test_2 = dummies_test.values
    model.fit(
        x_train_1,
        y_train_1,
        validation_data=(x_test_2, y_test_2),
        epochs=9,
        batch_size=batch_size,
        verbose=1,
    )

    pred = model.predict(x_test_2)
    pred = np.argmax(pred, axis=1)
    y_eval = np.argmax(y_test_2, axis=1)
    score = metrics.accuracy_score(y_eval, pred)
    oos_pred.append(score)
    print("Validation score: ", score)
