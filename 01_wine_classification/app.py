import pandas as pd
import tensorflow as tf
import numpy as np


class App:
    def __init__(self):
        train = pd.read_csv("./data/train.csv")
        test = pd.read_csv("./data/test.csv")
        sub = pd.read_csv("./submission.csv")

        train['type'] = train['type'].map({'white':0, 'red':1}).astype(int)
        test['type'] = test['type'].map({'white':0, 'red':1}).astype(int)

        train_x = train.drop(['index', 'quality'], axis = 1)
        train_y = train['quality']
        test_x = test.drop('index', axis = 1)

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(
            optimizer='adam',
            loss=tf.losses.MSLE,
        )

        model.fit(
            train_x,
            train_y,
            batch_size=32,
            epochs=10,
        )

        y_pred = model.predict(test_x)
        sub['quality'] = np.int32(y_pred)
        sub.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    app = App()

