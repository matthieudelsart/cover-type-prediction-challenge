import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def get_data_train(DATA_PATH=''):
    return pd.read_csv(DATA_PATH+"train.csv")


def get_data_test(DATA_PATH=''):
    return pd.read_csv(DATA_PATH+"test-full.csv")


def csv_for_submission(df: pd.DataFrame, name=''):
    OUTPUT_PATH = "Output/"

    col_names = df.columns.to_list()
    assert ('Id' in col_names) and (
        'Cover_Type' in col_names), "Columns names Id and Cover_Type not found"

    submission = pd.DataFrame({'Id': df.Id, 'Cover_Type': df.Cover_Type})
    if name != '':
        file_name = name + ".csv"
    else:
        cur_date = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
        file_name = f"Submission_{cur_date}.csv"
    submission.to_csv(OUTPUT_PATH + file_name, index=False)


def clean_predictor(y_pred, Id=None):
    df_test = get_data_test()
    df_train = get_data_train()

    if Id is not None:
        predictions_df = pd.DataFrame(
            {'Id': Id, 'Cover_Type': y_pred})
    else:   # We assume the prediction are sorted
        predictions_df = pd.DataFrame({'Cover_Type': y_pred})
        predictions_df['Id'] = range(1, len(df_test) + 1)

    # Removing those in df_train
    predictions_df.drop(predictions_df[predictions_df["Id"].isin(
        df_train["Id"])].index, inplace=True)

    # Adding df_train instead
    predictions_df = pd.concat(
        [df_train[['Id', 'Cover_Type']], predictions_df], axis=0, ignore_index=True)

    # Sorting by Id (just in case)
    predictions_df.sort_values("Id", inplace=True)

    return predictions_df


def IWCV(df_train=None,
         predictor=RandomForestClassifier(n_estimators=100, random_state=42),
         k_valid=10,
         verbose=0):
    """
    df_train: training data
    predictor: classifier (can be a sklearn pipeline)
    k_valid: number of cross-validations desired

    """
    COEFFS = np.array([2.63, 3.06, 0.43, 0.05, 0.24, 0.27, 0.32])

    if df_train is None:
        df_train = get_data_train()

    if "Wilderness_Area_Synth" in df_train.columns:
        df_train = df_train.drop(columns="Wilderness_Area_Synth")

    # Separate features and target
    X_train = df_train.drop('Cover_Type', axis=1)
    y_train = df_train['Cover_Type']

    class_accuracies = np.zeros((k_valid, 7))

    for i in range(k_valid):
        if verbose:
            print(f"Fold {i+1}")
        data_train, data_test, target_train, target_test = train_test_split(
            X_train, y_train, test_size=1/k_valid
        )
        predictor.fit(data_train, target_train)
        y_pred = predictor.predict(data_test)

        for label in range(1, 8):
            class_accuracies[i, label - 1] = accuracy_score(target_test[target_test == label],
                                                            y_pred[target_test == label])
        if verbose:
            print("Accuacy", class_accuracies[i, :])
    IMCV = np.mean(class_accuracies @ COEFFS) / np.sum(COEFFS)

    return IMCV, class_accuracies.mean(axis=0)
