import pandas as pd
from datetime import datetime


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
