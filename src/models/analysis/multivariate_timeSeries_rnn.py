from genericpath import exists
import warnings
import numpy as np
import pandas as pd
import pandas_datareader as pdr

# from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import minmax_scale
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import tensorflow.keras.backend as K
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime, date
from pathlib import Path
import os
from os.path import exists

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = [13, 7]
plt.rcParams["figure.dpi"] = 100
sns.set_style("whitegrid")
np.random.seed(42)
# os.environ["NUMEXPR_MAX_THREADS"] = "24"
# os.environ["NUMEXPR_NUM_THREADS"] = "12"

saveDate = str(datetime.now())[:10]
saveMonth = str(saveDate)[:7]
results_path = Path(f"data/variates/multivariate/{saveMonth}/{saveDate}/")
if not results_path.exists():
    results_path.mkdir(parents=True)


#       *       *       *       *                                          > model: [ Multivariate TimeSeries RNN ] *


class Multivariate_TimeSeries_RNN(object):
    def __init__(self):
        self.tics = ["UMCSENT", "IPGMFN"]

    def multivariate(self):
        df = pdr.DataReader(
            name=self.tics,
            data_source="fred",
            start="2010-01-01",
            end=datetime.now(),
        ).dropna()
        df.columns = ["Consumer_Sentiment", "Industrial_Production"]
        st.line_chart(df)
        plt.close()

        df_transformed = pd.DataFrame(
            {
                "Industrial_Production": np.log(df.Industrial_Production).diff(12),
                "Consumer_Sentiment": df.Consumer_Sentiment.diff(12),
            }
        ).dropna()
        df_transformed = df_transformed.apply(minmax_scale)

        def create_multivariate_rnn_data(data, window_size):
            y = data[window_size:]
            n = data.shape[0]
            X = np.stack(
                [data[i:j] for i, j in enumerate(range(window_size, n))], axis=0
            )
            return X, y

        K.clear_session()

        window_size = 18
        test_size = 24
        X, y = create_multivariate_rnn_data(df_transformed, window_size=window_size)
        train_size = X.shape[0] - test_size
        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]
        n_features = output_size = 2
        lstm_units = 12
        dense_units = 6

        rnn = Sequential(
            [
                LSTM(
                    units=lstm_units,
                    dropout=0.1,
                    recurrent_dropout=0.1,
                    input_shape=(window_size, n_features),
                    name="LSTM",
                    return_sequences=False,
                ),
                Dense(dense_units, name="FC"),
                Dense(output_size, name="Output"),
            ]
        )

        optimizer1 = keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name="Adam",
        )

        rnn.compile(optimizer=optimizer1, loss="mae", metrics=["mae", "acc"])
        lstm_path = (results_path / "lstm.h5").as_posix()

        checkpointer = ModelCheckpoint(
            filepath=lstm_path,
            verbose=1,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        )

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        result = rnn.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=20,
            shuffle=False,
            validation_data=(X_test, y_test),
            # callbacks=[early_stopping, checkpointer],
            verbose=1,
        )

        y_pred = pd.DataFrame(
            rnn.predict(X_test), columns=y_test.columns, index=y_test.index
        )

        # test_mae = mean_absolute_error(y_pred, y_test)
        # st.markdown(f"**Mean Absolute Error = {round(float(test_mae * 100),2)}%**")

        st.info(
            "**Log Return is one of three methods for calculating returns. \
                It assumes returns are compounded continuously rather than across sub-periods**"
        )

        fig, axes = plt.subplots(nrows=2, ncols=1)
        pd.DataFrame(result.history).rename(
            columns={"loss": "Training", "val_loss": "Validation"}
        )
        for i, col in enumerate(y_test.columns):
            y_train.loc["2010":, col].plot(ax=axes[i], label="Training")
            y_test[col].plot(ax=axes[i], label="Actual")
            y_pred[col].plot(ax=axes[i], label="Prediction")
            axes[i].set_xlabel("")
        fontP = FontProperties()
        fontP.set_size("small")

        axes[0].set_ylim(-0.2, 1.3)
        axes[0].set_xlim(date(2016, 1, 1))
        axes[0].fill_between(x=y_test.index, y1=0.0, y2=1.4, color="grey", alpha=0.3)
        axes[0].set_title("Industrial Production", fontsize=25, fontweight="bold")
        axes[0].set_xlabel("Time (years)", fontsize=15, fontweight="bold")
        axes[0].set_ylabel("Log Return", fontsize=15, fontweight="bold")
        for label in axes[0].get_xticklabels() + axes[0].get_yticklabels():
            label.set_fontsize(15)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", prop=fontP)
        axes[0].grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
        sns.despine()
        plt.tight_layout()

        axes[1].set_ylim(-0.2, 1.3)
        axes[1].set_xlim(date(2016, 1, 1))
        axes[1].fill_between(x=y_test.index, y1=0.0, y2=1.4, color="grey", alpha=0.3)
        axes[1].set_title("Consumer Sentiment", fontsize=25, fontweight="bold")
        axes[1].set_xlabel("Time (years)", fontsize=15, fontweight="bold")
        axes[1].set_ylabel("Log Return", fontsize=15, fontweight="bold")
        for label in axes[1].get_xticklabels() + axes[1].get_yticklabels():
            label.set_fontsize(15)
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", prop=fontP)
        axes[1].grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
        plt.xlim(date(2016, 1, 1))
        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)

        return


#       *       *       *       *                                                                 > stage: [ TEST ]


if __name__ == "__main__":
    Multivariate_TimeSeries_RNN().multivariate()
