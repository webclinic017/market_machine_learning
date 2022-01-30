import warnings
from datetime import datetime, date
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import streamlit as st
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from yahooquery import Ticker

warnings.filterwarnings("ignore")
today = str(datetime.now())[:10]
mpl.use("Agg")
plt.style.use("ggplot")
sm, med, lg = 10, 15, 20
plt.rc("font", size=sm)  # controls default text sizes
plt.rc("axes", titlesize=med)  # fontsize of the axes title
plt.rc("axes", labelsize=med)  # fontsize of the x & y labels
plt.rc("xtick", labelsize=sm)  # fontsize of the tick labels
plt.rc("ytick", labelsize=sm)  # fontsize of the tick labels
plt.rc("legend", fontsize=sm)  # legend fontsize
plt.rc("figure", titlesize=lg)  # fontsize of the figure title
plt.rc("axes", linewidth=2)  # linewidth of plot lines
plt.rcParams["figure.figsize"] = [18, 10]
plt.rcParams["figure.dpi"] = 150

today_stamp = str(datetime.now())[:10]
savePlot = Path(f"data/variates/univariate/{today_stamp}/")
if not savePlot.exists():
    savePlot.mkdir(parents=True)

# *     *         *     *     *     *     *     *     *     *     *     *     *     *     *     *


class The_Univariate_TS_Reg(object):
    def __init__(self, stock_symbol):
        self.ticker = stock_symbol
        self.saver = stock_symbol

        def company_longName(symbol):
            d = Ticker(symbol).quote_type
            return list(d.values())[0]["longName"]

        self.company = company_longName(self.ticker)

    def runs(self):
        st.header(f"{self.company} [{self.ticker}] - Univariate Analysis & Forecast")
        if self.ticker:
            sp500 = yf.download(self.ticker, period="5y", interval="1d")
            sp500 = pd.DataFrame(sp500["Adj Close"])
            sp500.columns = [self.saver]
            sp500.fillna(0.0, inplace=True)
            scaler = MinMaxScaler()
            sp500_scaled = pd.Series(
                scaler.fit_transform(sp500).squeeze(), index=sp500.index
            )

            def create_univariate_rnn_data(data, window_size):
                n = len(data)
                y = data[window_size:]
                data = data.values.reshape(-1, 1)  # make 2D
                X = np.hstack(
                    tuple(
                        [
                            data[i : n - j, :]
                            for i, j in enumerate(range(window_size, 0, -1))
                        ]
                    )
                )
                return pd.DataFrame(X, index=y.index), y

            window_size = 63
            X, y = create_univariate_rnn_data(sp500_scaled, window_size)
            X_train = X[:"2020"].values.reshape(-1, window_size, 1)
            y_train = y[:"2020"]

            # keep the last year for testing
            X_test = X["2020":].values.reshape(-1, window_size, 1)
            y_test = y["2020":]
            n_obs, window_size, n_features = X_train.shape

            rnn = Sequential(
                [
                    LSTM(units=10, input_shape=(window_size, n_features), name="LSTM"),
                    Dense(n_features, name="Output"),
                ]
            )
            print(rnn.summary())

            optimizer1 = keras.optimizers.Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=False,
                name="Adam",
            )
            # rnn.compile(optimizer='RMSProp', loss='mae')
            # rnn.compile(optimizer="Adam", loss="mae")
            # rnn.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])
            rnn.compile(optimizer=optimizer1, loss="mae", metrics=["mae", "acc"])

            rnn_path = (savePlot / f"univariate_rnn_{self.ticker}.h5").as_posix()

            checkpointer = ModelCheckpoint(
                filepath=rnn_path,
                verbose=1,
                monitor="val_loss",
                mode="min",
                save_best_only=True,
            )
            early_stopping = EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )
            lstm_training = rnn.fit(
                x=X_train,
                y=y_train,
                batch_size=40,
                epochs=100,
                verbose=1,
                callbacks=[early_stopping, checkpointer],
                validation_split=0.0,
                validation_data=(X_test, y_test),
                shuffle=True,
                class_weight=None,
                sample_weight=None,
                initial_epoch=0,
                steps_per_epoch=None,
                validation_steps=None,
                validation_batch_size=None,
                validation_freq=1,
                max_queue_size=10,
                workers=1,
                # use_multiprocessing=True,
            )

            fig, ax = plt.subplots()
            loss_history = pd.DataFrame(lstm_training.history).pow(0.5)
            loss_history.index += 1
            best_rmse = loss_history.val_loss.min()
            best_epoch = loss_history.val_loss.idxmin()
            title = f"5-Epoch Rolling RMSE (Best Validation RMSE: {best_rmse:.4%})"
            loss_history.columns = ["Training RMSE", "Validation RMSE"]
            loss_history.rolling(5).mean().plot(logy=True, lw=2, title=title, ax=ax)
            ax.axvline(best_epoch, ls="--", lw=1, c="k")
            sns.despine()
            fig.tight_layout()
            # fig.savefig( savePlot / f"univariate_timeSeries_rnn_1-error-{self.ticker}.png",dpi=250,)
            st.pyplot(fig)
            plt.close(fig)

            train_rmse_scaled = np.sqrt(rnn.evaluate(X_train, y_train, verbose=1))
            test_rmse_scaled = np.sqrt(rnn.evaluate(X_test, y_test, verbose=1))

            print(train_rmse_scaled)
            print(test_rmse_scaled)

            train_predict_scaled = rnn.predict(X_train)
            test_predict_scaled = rnn.predict(X_test)

            train_ic = spearmanr(y_train, train_predict_scaled)[0]
            test_ic = spearmanr(y_test, test_predict_scaled)[0]

            train_predict = pd.Series(
                scaler.inverse_transform(train_predict_scaled).squeeze(),
                index=y_train.index,
            )
            test_predict = pd.Series(
                scaler.inverse_transform(test_predict_scaled).squeeze(),
                index=y_test.index,
            )

            y_train_rescaled = scaler.inverse_transform(y_train.to_frame()).squeeze()
            y_test_rescaled = scaler.inverse_transform(y_test.to_frame()).squeeze()

            train_rmse = np.sqrt(mean_squared_error(train_predict, y_train_rescaled))
            test_rmse = np.sqrt(mean_squared_error(test_predict, y_test_rescaled))

            sp500["Model-Train"] = train_predict
            sp500["True-Predictions"] = test_predict
            sp500 = sp500.join(
                train_predict.to_frame("predictions")
                .assign(data="Train")
                .append(test_predict.to_frame("predictions").assign(data="Test"))
            )

            fig = plt.figure()
            ax1 = plt.subplot(221)
            sp500.loc["2018":, self.saver].plot(lw=3, ax=ax1, c="k", alpha=0.6)
            sp500.loc["2018":, ["True-Predictions", "Model-Train"]].plot(
                lw=2, ax=ax1, ls="--"
            )
            ax1.set_title(
                f"Univariate TimeSeries RNN Model (In & Out Of Sample) ~ {self.saver}"
            )
            ax1.set_ylabel("Stock Price")
            with sns.axes_style("white"):
                ax3 = plt.subplot(223)
                sns.scatterplot(
                    x=self.saver, y="predictions", data=sp500, hue="data", ax=ax3
                )
                ax3.text(
                    x=0.02, y=0.95, s=f"Test IC ={test_ic:.2%}", transform=ax3.transAxes
                )
                ax3.text(
                    x=0.02,
                    y=0.90,
                    s=f"Train IC={train_ic:.2%}",
                    transform=ax3.transAxes,
                )
                ax3.set_title("Correlation Plot ~ ")
                ax3.legend(loc="lower right")
                ax2 = plt.subplot(222)
                ax4 = plt.subplot(224, sharex=ax2, sharey=ax2)
                sns.distplot(train_predict.squeeze() - y_train_rescaled, ax=ax2)
                ax2.set_title("Train Error")
                ax2.text(
                    x=0.03,
                    y=0.92,
                    s=f"Train RMSE ={train_rmse:.4f}",
                    transform=ax2.transAxes,
                )
                ax2.set_ylabel("val_loss - Train_Tally")
                ax2.set_xlabel("Root_Mean_Squared_Error - Train_Tally")
                sns.distplot(test_predict.squeeze() - y_test_rescaled, ax=ax4)
                ax4.set_title("Test Error")
                ax4.text(
                    x=0.03,
                    y=0.92,
                    s=f"Test RMSE ={test_rmse:.4f}",
                    transform=ax4.transAxes,
                )
                ax4.set_ylabel("val_loss - Test_Tally (less train) = Final")
                ax4.set_xlabel("Root_Mean_Squared_Error - Test_Tally >>> Final")
            sns.despine()
            plt.title(f"Univariate Model of {self.company} ({self.saver})")
            plt.grid(True)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            fig, ax = plt.subplots()
            ax = sp500.loc["2019":, self.saver].plot(
                lw=2, c="k", alpha=0.6, label=f"{self.saver} Stock Price"
            )
            sp500.loc["2020":, ["True-Predictions", "Model-Train"]].plot(
                ax=ax, lw=1, style=[":", "--"]
            )
            ax.vlines(
                ["2017-07-01", "2018-12-25"],
                0,
                1,
                transform=ax.get_xaxis_transform(),
                colors="r",
                lw=2,
                ls="--",
                label="Model-Train",
            )
            ax.vlines(
                ["2019-01-01", "2019-12-25"],
                0,
                1,
                transform=ax.get_xaxis_transform(),
                colors="y",
                lw=2,
                ls="--",
                label="Model-Validation",
            )
            ax.vlines(
                ["2020-01-01", "2020-12-25"],
                0,
                1,
                transform=ax.get_xaxis_transform(),
                colors="g",
                lw=2,
                ls="--",
                label="Model-Predict [In Sample]",
            )
            ax.vlines(
                ["2021-01-01", "2021-06-30"],
                0,
                1,
                transform=ax.get_xaxis_transform(),
                colors="b",
                lw=2,
                ls="--",
                label="Model-Predict [Out-Of Sample]",
            )
            ax.set_title(
                f"Univariate TimeSeries RNN Model (In & Out Of Sample) ~{self.company} ({self.saver})",
                fontsize=30,
                fontweight="bold",
            )
            ax.set_ylabel("Stock Price ($)", fontsize=20, fontweight="bold")
            ax.set_xlabel("Date", fontsize=20, fontweight="bold")
            plt.grid(True)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(15)
            ax.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.3)
            ax.legend(loc="best", prop={"size": 16})
            plt.xlim(date(2020, 12, 1))
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            return


#     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *


if __name__ == "__main__":
    The_Univariate_TS_Reg("TSLA").runs()
