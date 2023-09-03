import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical


def decision_tree(X_train, X_test, y_train, y_test):
    dtc = DecisionTreeClassifier().fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    return classification_report(y_test, y_pred)


def knn(X_train, X_test, y_train, y_test, n_neighbors=3):
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    return classification_report(y_test, y_pred)


def naive_bayes(X_train, X_test, y_train, y_test):
    nb_model = GaussianNB()
    nb_model = nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)
    return classification_report(y_test, y_pred)


def log_regression(X_train, X_test, y_train, y_test):
    lg_model = LogisticRegression(max_iter=5000)
    lg_model = lg_model.fit(X_train, y_train)
    y_pred = lg_model.predict(X_test)
    return classification_report(y_test, y_pred)


def svm(X_train, X_test, y_train, y_test):
    svm_model = SVC()
    svm_model = svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    return classification_report(y_test, y_pred)


def random_forest(X_train, X_test, y_train, y_test):
    rfc = RandomForestClassifier().fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    return classification_report(y_test, y_pred)


def ada_boost(X_train, X_test, y_train, y_test):
    ada = AdaBoostClassifier().fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    return classification_report(y_test, y_pred)


def neural_network(X_train, X_test, y_train, y_test):
    y_train_encoded = to_categorical(y_train)[:, 1:]
    y_test_encoded = to_categorical(y_test)[:, 1:]

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss=categorical_crossentropy, metrics=['accuracy'])
    model.fit(X_train, y_train_encoded, epochs=100, batch_size=32, validation_data=(X_test, y_test_encoded), verbose=0)

    y_pred = model.predict(X_test)
    return classification_report(y_test_encoded.argmax(axis=1), y_pred.argmax(axis=1))


def supervised(X_train, X_test, y_train, y_test, display=False, save=False):
    dt = decision_tree(X_train, X_test, y_train, y_test)
    kn = knn(X_train, X_test, y_train, y_test, n_neighbors=11)
    nb = naive_bayes(X_train, X_test, y_train, y_test)
    lr = log_regression(X_train, X_test, y_train, y_test)
    s = svm(X_train, X_test, y_train, y_test)
    rf = random_forest(X_train, X_test, y_train, y_test)
    ab = ada_boost(X_train, X_test, y_train, y_test)
    nn = neural_network(X_train, X_test, y_train, y_test)
    report = f"Decision Tree: \n{dt}\n\nKNN: \n{kn}\n\nNaive Bayes: \n{nb}\n\nLogistic Regression: \n{lr}\n\nSVM: \n{s}\n\nRandom Forest: \n{rf}\n\nAda Boost: \n{ab}\n\nNeural Network: \n{nn}"
    if display:
        print(report)
    if save:
        with open("report.txt", "w") as file:
            file.write(report)


df = pd.read_csv('dataset/new_dataset.csv', low_memory=False)
df.drop(['accident_index'], axis=1, inplace=True)

X_tr, X_te, y_tr, y_te = train_test_split(df.drop('accident_severity', axis=1), df['accident_severity'], test_size=0.2)

supervised(X_tr, X_te, y_tr, y_te, display=True, save=True)
