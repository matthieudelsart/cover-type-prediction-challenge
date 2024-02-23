
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans


df_test = pd.read_csv("test-full.csv")
df_train = pd.read_csv("train.csv")
coeffs = np.array([2.63, 3.06, 0.43, 0.05, 0.24, 0.27, 0.32])


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

### 1 - normal
def IWCV(df_train=df_train, 
         predictor=RandomForestClassifier(n_estimators=100, random_state=42), 
         k_valid=10,
         coeffs=coeffs):
    """
    Inputs:
    df_train: training data
    predictor: classifier (can be a sklearn pipeline)
    k_valid: number of cross-validations desired
            
    Outputs:
    1. IWCV - unbiased estimate of test score if assumptions are correct
    2. clean_accuracies - array of estimated accuracy per class
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

### 2 - with oversampling
def IWCV_oversample(df_train=df_train, 
         predictor=RandomForestClassifier(n_estimators=100, random_state=42), 
         k_valid=10,
         ovs_strat = {1: 30_000, 2: 30_000}) :
    
    if "Wilderness_Area_Synth" in df_train.columns:
        df_train = df_train.drop(columns="Wilderness_Area_Synth")

    # Define the oversampler
    adasyn = ADASYN(sampling_strategy=ovs_strat) ## Random state = 4 ou 1 sont les meilleurs so far à 0.8297 en CV (mais fixer seed aussi en cross_val...)

    # Separate features and target 
    X_train = df_train.drop('Cover_Type', axis=1)
    y_train = df_train['Cover_Type']
    
    class_accuracies = np.zeros((k_valid, 7))
    
    for i in range(k_valid):
        data_train, data_test, target_train, target_test = train_test_split(
            X_train, y_train, test_size = 1 / k_valid
        )
        
        # Oversampling
        X_train_synth, y_train_synth = adasyn.fit_resample(data_train, target_train)
        X_train_synth = pd.DataFrame(X_train_synth, columns=X_train.columns)
        
        predictor.fit(X_train_synth, y_train_synth)
        y_pred = predictor.predict(data_test)

        for label in range(1,8):
            class_accuracies[i, label - 1] = accuracy_score(target_test[target_test == label], 
                                                        y_pred[target_test == label])
        IMCV = np.mean(class_accuracies @ coeffs) / np.sum(coeffs)
        
    return IMCV, class_accuracies.mean(axis=0)


### 3 - with oversampling + kmeans
# Defining pipe to be able to choose number of clusters
def pipe_setter(n_clusters=3, clf=RandomForestClassifier(n_estimators=100, random_state=42)):
    
    # Initializing kmeans
    km_test = KMeans(n_clusters=n_clusters, n_init=10, init="k-means++")
    km_test.fit_predict(df_test.loc[:, "Id":"Wilderness_Area4"])

    # Setting pipe
    def _enocode_kmeans(X, kmeans=km_test):
        X = X.copy()
        X["kmean_cluster"] = kmeans.predict(X.loc[:, "Id":"Wilderness_Area4"])
        return X
    km_encoder = FunctionTransformer(_enocode_kmeans)

    cat_col = ["kmean_cluster"]
    cols = df_train.drop(columns=['Cover_Type', 'Wilderness_Area_Synth']).columns

    preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_col),
            ("others", "passthrough", cols),
        ])
    
    return make_pipeline(km_encoder, preprocessor, clf)

def IWCV_ovs_km(df_train=df_train, 
         predictor=pipe_setter(), 
         k_valid=10,
         coeffs=coeffs):
    
    ovs_strat = {1: 30_000, 2: 30_000}
    if "Wilderness_Area_Synth" in df_train.columns:
        df_train = df_train.drop(columns="Wilderness_Area_Synth")

    # Define the oversampler
    adasyn = ADASYN(sampling_strategy=ovs_strat) 

    # Separate features and target 
    X_train = df_train.drop('Cover_Type', axis=1)
    y_train = df_train['Cover_Type']
    
    class_accuracies = np.zeros((k_valid, 7))
    
    for i in range(k_valid):
        data_train, data_test, target_train, target_test = train_test_split(
            X_train, y_train, test_size = 1 / k_valid
        )
        
        # Oversampling
        X_train_synth, y_train_synth = adasyn.fit_resample(data_train, target_train)
        X_train_synth = pd.DataFrame(X_train_synth, columns=X_train.columns)
        
        predictor.fit(X_train_synth, y_train_synth)
        y_pred = predictor.predict(data_test)

        for label in range(1,8):
            class_accuracies[i, label - 1] = accuracy_score(target_test[target_test == label], 
                                                        y_pred[target_test == label])
        IMCV = np.mean(class_accuracies @ coeffs) / np.sum(coeffs)
        
    return IMCV, class_accuracies.mean(axis=0)