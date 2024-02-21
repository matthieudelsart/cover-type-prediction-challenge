import pandas as pd
from datetime import datetime


def get_submission(model, X_test, X_train=None, y_train=None):
    """
    Generate a submission file based on predictions made by a machine learning model.

    Parameters:
    - model (object): The machine learning model to use for prediction. This should be a callable object that can make predictions on new data.
    - X_test (array-like): The test data for which predictions are to be made.
    - X_train (array-like, optional): Training data to fit the model if training is required. Default is None.
    - y_train (array-like, optional): Target values corresponding to `X_train` if training is required. Default is None.

    Returns:
    None

    Description:
    This function generates a submission file based on the predictions made by the specified machine learning model on the provided test data (`X_test`). If training data (`X_train` and `y_train`) are provided, the model is first trained on that data before making predictions.

    The predictions are saved to a CSV file named `Submission_<current_date>.csv`, where `<current_date>` is the current date and time in the format `YYYY-MM-DD-HH.MM.SS`.

    # Example usage:
    Assuming model, X_test, X_train, and y_train are defined elsewhere
    get_submission(LinearRegression(), X_test, X_train, y_train)
    ```

    Notes:
    - The provided `model` object must be callable, capable of making predictions on new data.
    - The function saves the submission file in the Output directory.
    """
    if (X_train is not None) and y_train is not None:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    pred = pd.DataFrame(y_pred, index=X_test.index)
    current_date = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
    pred.to_csv(f"Output/Submission_{current_date}")


def get_data_train(DATA_PATH=''):
    return pd.read_csv(DATA_PATH+"train.csv")


def get_data_test(DATA_PATH=''):
    return pd.read_csv(DATA_PATH+"test-full.csv")


def csv_for_submission(df: pd.DataFrame, name=''):
    OUTPUT_PATH = "Output/"
    if df.shape[1] == 2:
        if df.columns.to_list() != ['Id', 'Cover_Type']:
            print("Columns names not match!")

    submission = pd.DataFrame({'Id': df.Id, 'Cover_Type': df.Cover_Type})
    cur_date = datetime.now().strftime("%Y-%m-%d-%H.%M.%S")
    submission.to_csv(f"Output/Submission_{name}{cur_date}.csv", index=False)
