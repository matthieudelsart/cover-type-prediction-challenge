import pandas as pd
import numpy as np
from utils import *


def Kaggle_score(path):
    PREDS_PATH = path
    TRUTH_PATH = "data/ground_truth.parquet"

    truth = pd.read_parquet(TRUTH_PATH)['Cover_Type'].to_numpy()
    preds = pd.read_parquet(PREDS_PATH)['Cover_Type'].to_numpy()

    accuracy = (preds == truth).mean()

    print("Accuracy:", accuracy)


def get_model_score(model, df, categorical_feature=None):
    X = df.drop(columns=["Cover_Type"], axis=1)
    y = df["Cover_Type"]
    if categorical_feature is not None:
        model.fit(X, y, categorical_feature=categorical_feature)
    else:
        model.fit(X, y)
    df_test = get_data_test()
    pred = model.predict(df_test)

    pred = clean_predictor(y_pred=pred, Id=df_test.Id)

    truth = pd.read_parquet(
        "data/ground_truth.parquet")['Cover_Type'].to_numpy()
    accuracy = (pred.Cover_Type.to_numpy() == truth).mean()
    print("Accuracy:", accuracy)
    # csv_for_submission(pred, "pred_test")
    # Kaggle_score('Output/pred_test')


if __name__ == "__main__":
    path = input("Prediction path:")
    Kaggle_score(path)
