
import pandas as pd

df_test = pd.read_csv("test-full.csv")
df_train = pd.read_csv("train.csv")


def clean_predictor(y_pred, df_test=df_test, df_train=df_train):
    predictions_df = pd.DataFrame(
        {'Id': y_pred["Id"], 'Cover_Type': y_pred["Cover_Type"]})
    # predictions_df['Id'] = range(1, len(df_test) + 1)

    # Removing those in df_train
    predictions_df.drop(predictions_df[predictions_df["Id"].isin(
        df_train["Id"])].index, inplace=True)

    # Adding df_train instead
    predictions_df = pd.concat(
        [df_train[['Id', 'Cover_Type']], predictions_df], axis=0, ignore_index=True)

    # Sorting by Id (just in case)
    predictions_df.sort_values("Id", inplace=True)

    return predictions_df
