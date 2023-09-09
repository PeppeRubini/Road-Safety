import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

NEW_ACCIDENT_PATH = '../Dataset/accident-preprocessed.csv'
NEW_VEHICLE_PATH = '../Dataset/vehicle-preprocessed.csv'
NEW_CASUALTY_PATH = '../Dataset/casualty-preprocessed.csv'

PLOT_PATH = "../plots/"


def plot_accident_severity(accident: pd.DataFrame, save=True, show=False):
    plt.pie(accident['accident_severity'].value_counts(), labels=['Slight', 'Serious', 'Fatal'], autopct='%1.1f%%')
    plt.title("Accident Severity")
    if show:
        plt.show()
    if save:
        plt.savefig("../plots/accident_severity.png")
    plt.close()


def plot_accident_location(accident: pd.DataFrame, save=False, show=True, color=False):
    if color:
        plt.scatter(accident[accident["accident_severity"] == 1]['longitude'],
                    accident[accident["accident_severity"] == 1]['latitude'], color='red', marker=".", alpha=0.5)
        plt.scatter(accident[accident["accident_severity"] == 2]['longitude'],
                    accident[accident["accident_severity"] == 2]['latitude'], color='blue', marker=".", alpha=0.5)
        plt.scatter(accident[accident["accident_severity"] == 3]['longitude'],
                    accident[accident["accident_severity"] == 3]['latitude'], color='green', marker=".", alpha=0.5)
        plt.legend([1, 2, 3])
    else:
        plt.scatter(accident['longitude'], accident['latitude'], marker=".")

    plt.title("Accident Location")
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.xticks(range(-7, 3))
    plt.yticks(range(50, 62))
    plt.gca().set_aspect('equal', adjustable='box')
    if show:
        plt.show()
    if save:
        plt.savefig("../plots/accident_location.png")
    plt.close()


def stats(accident: pd.DataFrame, vehicle: pd.DataFrame, casualty: pd.DataFrame, save=False, display=True):
    stat_a = accident[['number_of_vehicles', 'number_of_casualties']].describe()
    stat_v = vehicle[['sex_of_driver', 'age_of_driver', 'age_of_vehicle', 'engine_capacity_cc']].describe()
    stat_c = casualty[['sex_of_casualty', 'age_of_casualty', 'casualty_severity']].describe()
    if display:
        print(stat_a)
        print(stat_v)
        print(stat_c)
    if save:
        stat_a.to_csv("../Dataset/accident_stat.csv")
        stat_v.to_csv("../Dataset/vehicle_stat.csv")
        stat_c.to_csv("../Dataset/casualty_stat.csv")


def code_label_dict(field_name: str, inverse=False) -> dict:
    df_guide = pd.read_excel('../Dataset/Road-Safety-Open-Dataset-Data-Guide.xlsx')
    d = {}
    for index, row in df_guide[df_guide['field name'] == field_name][['code/format', 'label']].iterrows():
        k = row['code/format']
        v = row['label']
        if inverse:
            d[v] = k
        d[k] = v
    return d


def change_code_to_description_df(df: pd.DataFrame, inverse=False):
    for col in df.columns:
        df[col] = df[col].replace(code_label_dict(col, inverse))


def cross_val_score_plot(score, name, save: True, display: False):
    plt.plot(range(1, 11), score)
    plt.title(f"{name} Cross Validation Score")
    plt.xlabel("k")
    plt.ylabel("score")
    if display:
        plt.show()
    if save:
        plt.savefig(PLOT_PATH + name + "_cross_val_score.png")
    plt.close()


def confusion_matrix_plot(confusion_matrix, labels, name, save: True, display: False):
    cm = ConfusionMatrixDisplay(confusion_matrix, display_labels=labels)
    cm.plot()
    if display:
        plt.show()
    if save:
        plt.savefig(PLOT_PATH + name + "_confusion_matrix.png")
    plt.close()
