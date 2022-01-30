import pandas as pd
from pandas.io.pickle import read_pickle
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
import yfinance as yf
from sklearn.tree import export_graphviz
import streamlit as st
from datetime import datetime


pd.set_option("display.max_rows", 50)
plt.style.use(["seaborn-darkgrid", "seaborn-poster"])
plt.rcParams["figure.figsize"] = [13, 6.5]
plt.style.use("seaborn")
sm, med, lg = 10, 15, 25
plt.rc("font", size=sm)  # controls default text sizes
plt.rc("axes", titlesize=med)  # fontsize of the axes title
plt.rc("axes", labelsize=med)  # fontsize of the x & y labels
plt.rc("xtick", labelsize=sm)  # fontsize of the tick labels
plt.rc("ytick", labelsize=sm)  # fontsize of the tick labels
plt.rc("legend", fontsize=sm)  # legend fontsize
plt.rc("figure", titlesize=lg)  # fontsize of the figure title
plt.rc("axes", linewidth=2)  # linewidth of plot lines
plt.rcParams["figure.figsize"] = [15, 6.5]
plt.rcParams["figure.dpi"] = 134

# from .web_plotRoc import plot_roc



class The_Random_Forest(object):


    def __init__(self, symbols):
        self.tickers = symbols
        self.mkt_index = "^GSPC"


    def collect_data(self):
        self.component_hist = yf.download(self.tickers, period="1y")
        self.index_hist = yf.download(self.mkt_index, period="1y")["Adj Close"]


    def clean_data(self):
        self.collect_data()
        self.component_df = pd.DataFrame(self.component_hist["Adj Close"].copy())
        self.component_df['SP500'] = self.index_hist
        weights = np.arange(1, 16)
        self.component_df["wma15"] = (
            self.component_df['SP500']
            .rolling(15)
            .apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
        )
        self.component_df["GL"] = (self.component_df['SP500'] >= self.component_df["wma15"])
        self.component_df = self.component_df.drop(('SP500'), axis=1)
        self.component_df = self.component_df.drop(("wma15"), axis=1)
        self.component_df.fillna(0.0, inplace=True)


    def score(self):
        self.clean_data()
        self.y = self.component_df.pop("GL").values
        self.X = self.component_df.values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y)
        self.rf = RandomForestClassifier()
        self.rf.fit(self.X_train, self.y_train)

        st.write("\n * score:", self.rf.score(self.X_test, self.y_test))
        self.y_predict = self.rf.predict(self.X_test)
        st.write("\n * precision:", precision_score(self.y_test, self.y_predict))
        st.write(" * recall:", recall_score(self.y_test, self.y_predict))
        rf = RandomForestClassifier(n_estimators=30, oob_score=True)
        rf.fit(self.X_train, self.y_train)
        st.write("\n * accuracy score:", rf.score(self.X_test, self.y_test))
        st.write(" * out of bag score:", rf.oob_score_)
        return self.X, self.y, self.X_train, self.X_test, self.y_train, self.y_test


    def feature_importance(self):
        (
            self.X,
            self.y,
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = self.score()
        feature_importances = np.argsort(self.rf.feature_importances_)

        present = pd.DataFrame()
        present["tickers"] = list(self.component_df.columns[feature_importances[-1::-1]])
        n = len(self.component_df.columns)
        # importances = forest_fit.feature_importances_[:n]
        importances = self.rf.feature_importances_[:n]
        std = np.std([tree.feature_importances_ for tree in self.rf.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        features = list(self.component_df.columns[indices])

        st.subheader("** Feature ranking:**")
        for f in range(n):
            st.write("%d. %s (%f)" % (f + 1, features[f], importances[indices[f]]))

        fig, ax = plt.subplots()
        st.subheader("**Feature Importance Plot**")
        ax.bar(range(n), importances[indices], yerr=std[indices], color="r", align="center")
        ax.set_xticks(range(n))
        ax.set_xticklabels(features, rotation=60)
        ax.set_xlim([-1, n])
        ax.set_xlabel("importance")
        ax.set_title("Feature Importances")
        plt.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.13)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


    def trees(self):
        self.feature_importance()
        num_trees = range(5, 100, 5)
        accuracies = []
        for n in num_trees:
            tot = 0
            for i in range(5):
                rf = RandomForestClassifier(n_estimators=n)
                rf.fit(self.X_train, self.y_train)
                tot += rf.score(self.X_test, self.y_test)
            accuracies.append(tot / 5)
        tree_prediction = rf.predict(self.X_test)
        lil_lst = {}
        for nt in range(len(num_trees)):
            lil_lst[num_trees[nt]] = round(accuracies[nt] * 100, 2)
        self.maxT = max(lil_lst, key=lil_lst.get)
        maxA = max(accuracies)

        fig, ax = plt.subplots()
        st.subheader("**Number of Trees [Accuracy] Plot**")
        st.write(f"** - Max Trees = {self.maxT}**")
        ax.plot(num_trees, accuracies)
        ax.plot(self.maxT, maxA, color="green", marker="X", ms=13)
        ax.set_xlabel("Number of Trees")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Num Trees")
        plt.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.13)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        tree = rf.estimators_[10]
        export_graphviz(
            tree,
            feature_names=self.tickers,
            rounded=True,
            precision=1,
        )

    # Use dot file to create a graph
    # (graph, ) = pydot.graph_from_dot_file(f'/home/gordon/one/report/portfolio_{today}/I_wideView/tree.png')
    # Write graph to a png file
    # graph.write_png(savePlot / f"rf_tree.png")


    def features(self):
        self.trees()
        num_features = range(1, len(self.component_df.columns) + 1)
        accuracies = []
        for n in num_features:
            tot = 0
            for i in range(5):
                rf = RandomForestClassifier(max_features=n)
                rf.fit(self.X_train, self.y_train)
                tot += rf.score(self.X_test, self.y_test)
            accuracies.append(tot / 5)
        feature_dict = {}
        for nt in range(len(num_features)):
            feature_dict[num_features[nt]] = round(accuracies[nt] * 100, 2)
        self.maxF = max(feature_dict, key=feature_dict.get)
        maxA2 = max(accuracies)

        fig, ax = plt.subplots()
        st.subheader("**Number of Features [Accuracy] Plot**")
        st.write(f"** - Max Features = {self.maxF}**")
        ax.plot(num_features, accuracies)
        ax.plot(self.maxF, maxA2, color="green", marker="X", ms=13)
        ax.set_xlabel("Number of Features")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Num Features")
        plt.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.13)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        return self.X, self.y


    def get_scores(self, classifier, **kwargs):
        model = classifier(**kwargs)
        model.fit(self.X_train, self.y_train)
        self.y_predict = model.predict(self.X_test)
        return (
            model.score(self.X_test, self.y_test),
            precision_score(self.y_test, self.y_predict),
            recall_score(self.y_test, self.y_predict),
        )


    def report_scores(self):
        self.features()
        lR_score, lR_precision, lR_recall = self.get_scores(LogisticRegression)
        dT_score, dT_precision, dT_recall = self.get_scores(DecisionTreeClassifier)
        rF_score, rF_precision, rF_recall = self.get_scores(
            RandomForestClassifier, n_estimators=self.maxT, max_features=self.maxF
        )
        nB_score, nB_precision, nB_recall = self.get_scores(MultinomialNB)
        model_titles = [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Naive Bayes",
        ]
        model_scores = [lR_score, dT_score, rF_score, nB_score]
        model_precision = [lR_precision, dT_precision, rF_precision, nB_precision]
        model_recall = [lR_recall, dT_recall, rF_recall, nB_recall]
        df = pd.DataFrame(model_titles, columns=["models"]).set_index("models")
        df["score"] = model_scores
        df["precision"] = model_precision
        df["recall"] = model_recall
        df["avg"] = (df["score"] + df["precision"] + df["recall"]) / 3
        df = df.sort_values("avg", ascending=False)
        st.subheader("** Model, Accuracy, Precision, Recall **")
        st.table(df)
        return df.index[0]


    def plot_roc(self, X, y, clf_class, plot_name, **kwargs):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True)
        y_prob = np.zeros((len(y), 2))
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train = y[train_index]
            clf = clf_class(**kwargs)
            clf.fit(X_train, y_train)
            # Predict probabilities, not classes
            y_prob[test_index] = clf.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y[test_index], y_prob[test_index, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, label="ROC fold %d (area = %0.2f)" % (i, roc_auc))
        mean_tpr /= n_splits
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)

        fig, ax = plt.subplots()
        plt.plot(mean_fpr, mean_tpr, "k--", label="Mean ROC (area = %0.2f)" % mean_auc, lw=2)
        plt.plot([0, 1], [0, 1], "--", color=(0.6, 0.6, 0.6), label="Random", lw=1.5)
        plt.axvline(x=0.15, color="r", ls="--", lw=2)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{plot_name} - Receiver operating characteristic")
        plt.legend(loc="lower right")
        plt.grid(True, color="k", linestyle="-", linewidth=1, alpha=0.13)
        plt.tight_layout()
        st.pyplot(fig)


    def plot_plot_roc(self):
        mod_to_run = self.report_scores()

        if mod_to_run == "Random Forest":
            st.subheader(f"** Visualize The ROC Curve [{mod_to_run}]**")
            self.plot_roc(self.X, self.y, RandomForestClassifier, "Random_Forest", n_estimators=45, max_features=5)

        if mod_to_run == "Logistic Regression":
            st.subheader(f"** Visualize The ROC Curve [{mod_to_run}]**")
            self.plot_roc(self.X, self.y, LogisticRegression, "Logistic_Regression")

        if mod_to_run == "Decision Tree":
            st.subheader(f"** Visualize The ROC Curve [{mod_to_run}]**")
            self.plot_roc(self.X, self.y, DecisionTreeClassifier, "Decision_Tree")

        if mod_to_run == "Naive Bayes":
            self.plot_roc(self.X, self.y, RandomForestClassifier, "Random_Forest", n_estimators=45, max_features=5)

            self.plot_roc(self.X, self.y, LogisticRegression, "Logistic_Regression")
            self.plot_roc(self.X, self.y, DecisionTreeClassifier, "Decision_Tree")
