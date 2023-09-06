from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from utils import cross_val_score_plot
import warnings

warnings.filterwarnings('ignore')


def decision_tree(X, y, k):
    best_y_pred = []
    best_accuracy = 0
    best_criterion = ""
    best_max_depth = 0
    best_min_samples_split = 0
    best_min_samples_leaf = 0
    best_dtc = None
    for criterion in ['gini', 'entropy']:
        for max_depth in [None, 5, 10, 20, 30]:
            for min_samples_split in [2, 5, 10, 20, 30]:
                for min_samples_leaf in [1, 2, 5, 10, 20, 30]:
                    dtc = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                                 min_samples_split=min_samples_split,
                                                 min_samples_leaf=min_samples_leaf)
                    y_pred = cross_val_predict(dtc, X, y, cv=k)
                    accuracy = accuracy_score(y, y_pred)
                    if accuracy > best_accuracy:
                        best_y_pred = y_pred
                        best_accuracy = accuracy
                        best_criterion = criterion
                        best_max_depth = max_depth
                        best_min_samples_split = min_samples_split
                        best_min_samples_leaf = min_samples_leaf
                        best_dtc = dtc
    cross_val_score_plot(cross_val_score(best_dtc, X, y, cv=k), "decision_tree", save=True, display=False)
    report = classification_report(y, best_y_pred)
    return "Decision Tree\n" + "criteria: " + best_criterion + "\nmax_depth: " + str(
        best_max_depth) + "\nmin_samples_split: " + str(best_min_samples_split) + "\nmin_samples_leaf: " + str(
        best_min_samples_leaf) + "\n" + report + "\n"


def knn(X, y, k):
    best_y_pred = []
    best_accuracy = 0
    best_n_neighbors = 0
    best_weights = ""
    best_metric = ""
    best_knn = None
    for n_neighbors in [3, 5, 7, 9, 11, 13, 15]:
        for weights in ['uniform', 'distance']:
            for metric in ['euclidean', 'manhattan', 'chebyshev']:
                knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
                y_pred = cross_val_predict(knn_model, X, y, cv=k)
                accuracy = accuracy_score(y, y_pred)
                if accuracy > best_accuracy:
                    best_y_pred = y_pred
                    best_accuracy = accuracy
                    best_n_neighbors = n_neighbors
                    best_weights = weights
                    best_metric = metric
                    best_knn = knn_model
    cross_val_score_plot(cross_val_score(best_knn, X, y, cv=k), "knn", save=True, display=False)
    report = classification_report(y, best_y_pred)
    return "KNN\n" + "n_neighbors: " + str(
        best_n_neighbors) + "\nweights: " + best_weights + "\nmetric: " + best_metric + "\n" + report + "\n"


def naive_bayes(X, y, k):
    best_y_pred = []
    best_accuracy = 0
    best_var_smoothing = 0
    best_nb = None
    for alpha in [0.1, 0.5, 1, 2, 5, 10]:
        nb_model = GaussianNB(var_smoothing=alpha)
        y_pred = cross_val_predict(nb_model, X, y, cv=k)
        accuracy = accuracy_score(y, y_pred)
        if accuracy > best_accuracy:
            best_y_pred = y_pred
            best_accuracy = accuracy
            best_var_smoothing = alpha
            best_nb = nb_model
    cross_val_score_plot(cross_val_score(best_nb, X, y, cv=k), "naive_bayes", save=True, display=False)
    report = classification_report(y, best_y_pred)
    return "Naive Bayes\n" + "var_smoothing: " + str(best_var_smoothing) + "\n" + report + "\n"


def log_regression(X, y, k):
    best_y_pred = []
    best_accuracy = 0
    best_c = 0
    best_lg = None
    for c in [0.1, 0.5, 1, 2, 5, 10]:
        lg_model = LogisticRegression(max_iter=5000, C=c)
        y_pred = cross_val_predict(lg_model, X, y, cv=k)
        accuracy = accuracy_score(y, y_pred)
        if accuracy > best_accuracy:
            best_y_pred = y_pred
            best_accuracy = accuracy
            best_c = c
            best_lg = lg_model
    cross_val_score_plot(cross_val_score(best_lg, X, y, cv=k), "log_regression", save=True, display=False)
    report = classification_report(y, best_y_pred)
    return "Logistic Regression\n" + "C: " + str(best_c) + "\n" + report + "\n"


def svm(X, y, k):
    best_y_pred = []
    best_accuracy = 0
    best_c = 0
    best_kernel = ""
    best_gamma = 0
    best_svm = None
    for c in [0.1, 0.5, 1, 2, 5, 10]:
        for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
            for gamma in ['scale', 'auto']:
                svm_model = SVC(C=c, kernel=kernel, gamma=gamma)
                y_pred = cross_val_predict(svm_model, X, y, cv=k)
                accuracy = accuracy_score(y, y_pred)
                if accuracy > best_accuracy:
                    best_y_pred = y_pred
                    best_accuracy = accuracy
                    best_c = c
                    best_kernel = kernel
                    best_gamma = gamma
                    best_svm = svm_model
    cross_val_score_plot(cross_val_score(best_svm, X, y, cv=k), "svm", save=True, display=False)
    report = classification_report(y, best_y_pred)
    return "SVM\n" + "C: " + str(best_c) + "\nkernel: " + best_kernel + "\ngamma: " + best_gamma + "\n" + report + "\n"


def random_forest(X, y, k):
    best_y_pred = []
    best_accuracy = 0
    best_n_estimators = 0
    best_max_features = ""
    best_max_depth = 0
    best_rfc = None
    for n_estimator in [100, 200, 300]:
        for max_features in ['sqrt', 'log2']:
            for max_depth in [None, 5, 10, 20, 30]:
                rfc = RandomForestClassifier(n_estimators=n_estimator, max_features=max_features,
                                             max_depth=max_depth)
                y_pred = cross_val_predict(rfc, X, y, cv=k)
                accuracy = accuracy_score(y, y_pred)
                if accuracy > best_accuracy:
                    best_y_pred = y_pred
                    best_accuracy = accuracy
                    best_n_estimators = n_estimator
                    best_max_features = max_features
                    best_max_depth = max_depth
                    best_rfc = rfc
    cross_val_score_plot(cross_val_score(best_rfc, X, y, cv=k), "random_forest", save=True, display=False)
    report = classification_report(y, best_y_pred)
    return "Random Forest\n" + "n_estimators: " + str(
        best_n_estimators) + "\nmax_features: " + best_max_features + "\nmax_depth: " + str(
        best_max_depth) + "\n" + report + "\n"


def ada_boost(X, y, k):
    best_y_pred = []
    best_accuracy = 0
    best_n_estimators = 0
    best_learning_rate = 0
    best_estimator = ""
    best_ada = None
    for n_estimator in [50, 100, 200, 300]:
        for learning_rate in [0.1, 0.5, 1, 2, 5, 10]:
            for estimator in [DecisionTreeClassifier(), GaussianNB(), LogisticRegression()]:
                ada = AdaBoostClassifier(n_estimators=n_estimator, learning_rate=learning_rate,
                                         estimator=estimator)
                y_pred = cross_val_predict(ada, X, y, cv=k)
                accuracy = accuracy_score(y, y_pred)
                if accuracy > best_accuracy:
                    best_y_pred = y_pred
                    best_accuracy = accuracy
                    best_n_estimators = n_estimator
                    best_learning_rate = learning_rate
                    best_estimator = str(estimator).split('(')[0]
                    best_ada = ada
    cross_val_score_plot(cross_val_score(best_ada, X, y, cv=k), "ada_boost", save=True, display=False)
    report = classification_report(y, best_y_pred)
    return "Ada Boost\n" + "n_estimators: " + str(best_n_estimators) + "\nlearning_rate: " + str(
        best_learning_rate) + "\nestimator: " + best_estimator + "\n" + report + "\n"


def neural_network(X, y, k):
    num_nodes = 0
    dropout_prob = 0

    def create_model():
        model = Sequential()
        model.add(Dense(num_nodes, activation='relu', input_shape=(X.shape[1],)))
        model.add(Dropout(dropout_prob))
        model.add(Dense(num_nodes / 2, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    best_y_pred = []
    best_accuracy = 0
    best_num_nodes = 0
    best_epoch = 0
    best_dropout_prob = 0
    best_model = None

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    for num_nodes in [16, 32, 64, 128]:
        for epoch in [10, 50, 100]:
            for dropout_prob in [0, 0.2, 0.5]:
                model = KerasClassifier(build_fn=create_model, epochs=epoch, verbose=0)
                y_pred = cross_val_predict(model, X, y_encoded, cv=kf)
                accuracy = accuracy_score(y_encoded, y_pred)
                if accuracy > best_accuracy:
                    best_y_pred = y_pred
                    best_accuracy = accuracy
                    best_num_nodes = num_nodes
                    best_epoch = epoch
                    best_dropout_prob = dropout_prob
                    best_model = model
    cross_val_score_plot(cross_val_score(best_model, X, y_encoded, cv=kf), "neural_network", save=True, display=False)
    report = classification_report(y_encoded, best_y_pred)

    return "Neural Network\n" + "Number of nodes: " + str(best_num_nodes) + " " + str(int(
        best_num_nodes / 2)) + " 3" + "\nEpoch: " + str(best_epoch) + "\nDropout probability: " + str(
        best_dropout_prob) + "\n" + report + "\n"
