
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df_test = pd.read_csv("test-full.csv")
df_train = pd.read_csv("train.csv")


#################
# Train inclus dans test donc pour s'assurer qu'on garde bien les bon cover_types du train
def clean_predictor(y_pred, df_test=df_test, df_train=df_train):
    predictions_df = pd.DataFrame({'Cover_Type': y_pred})
    predictions_df['Id'] = range(1, len(df_test) + 1)
    
    # Removing those in df_train
    predictions_df.drop(predictions_df[predictions_df["Id"].isin(df_train["Id"])].index, inplace=True)
    
    # Adding df_train instead
    predictions_df = pd.concat([df_train[['Cover_Type', 'Id']], predictions_df], axis=0, ignore_index=True)

    # Sorting by Id (just in case)
    predictions_df.sort_values("Id", inplace=True)
    
    return predictions_df
#################

#################
# Importance weighted cross-validation 

coeffs = np.array([2.63, 3.06, 0.43, 0.05, 0.24, 0.27, 0.32])

def IWCV(df_train=df_train, 
         predictor=RandomForestClassifier(n_estimators=100, random_state=42), 
         k_valid=10,
         coeffs=coeffs):
    """
    df_train: training data
    predictor: classifier (can be a sklearn pipeline)
    k_valid: number of cross-validations desired
    coeffs: do not touch, empirically obtained to measure the over/under 
            representation of classes between train and test sets
    """
    
    if "Wilderness_Area_Synth" in df_train.columns:
        df_train = df_train.drop(columns="Wilderness_Area_Synth")
        
    # Separate features and target 
    X_train = df_train.drop('Cover_Type', axis=1)
    y_train = df_train['Cover_Type']
    
    class_accuracies = np.zeros((k_valid, 7))
    
    for i in range(k_valid):
        data_train, data_test, target_train, target_test = train_test_split(
            X_train, y_train, test_size = 1 / k_valid
        )
        predictor.fit(data_train, target_train)
        y_pred = predictor.predict(data_test)

        for label in range(1,8):
            class_accuracies[i, label - 1] = accuracy_score(target_test[target_test == label], 
                                                        y_pred[target_test == label])
        IMCV = np.mean(class_accuracies @ coeffs) / np.sum(coeffs)
        
    return IMCV, class_accuracies.mean(axis=0)