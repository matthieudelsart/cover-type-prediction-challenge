from final import *
import optuna

import warnings

# Filter out the specific warning you want to ignore
warnings.filterwarnings("ignore")

def objective(trial):
    
    df_test = pd.read_csv("test-full.csv")
    df_train = pd.read_csv("train.csv")
    predict_true = pd.read_parquet("ground_truth.parquet")["Cover_Type"]

    # Un-one-hot-encoding the categorical variables
    soil_types = [f"Soil_Type{i}" for i in range(1, 41)]
    wilderness_areas = [f"Wilderness_Area{i}" for i in range(1,5)]
    df_test["Wilderness_Area_Synth"] = df_test[wilderness_areas] @ range(1,5)
    df_train["Wilderness_Area_Synth"] = df_train[wilderness_areas] @ range(1,5)
    df_test["Soil_Type_Synth"] = df_test[soil_types] @ range(1,41)
    df_train["Soil_Type_Synth"] = df_train[soil_types] @ range(1,41)
    df_train = df_train.drop(columns=wilderness_areas + soil_types)
    df_test = df_test.drop(columns=wilderness_areas + soil_types)

    # Separating train 
    X_train = df_train.drop(columns=['Cover_Type'], axis=1)
    y_train = df_train['Cover_Type']
    
    #ovs1 = trial.suggest_int("ovs1", 60_000, 95_000, log=True)
    ovs1 = 60_000
    #ovs2 = trial.suggest_int("ovs2", 65_000, 100_000, log=True)
    ovs2 = 93_500
    #smote_random = trial.suggest_int("smote_random", 0, 100)
    smote_random = 84
    #pca_random = trial.suggest_int("pca_random", 0, 100)
    pca_random = 67
    #ext_random = trial.suggest_int("ext_random", 0, 100)
    ext_random = 69
    #min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    min_samples_leaf = 1
    #min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_split = 2
    #n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    n_estimators = 745

    # MODEL
    print('Training...')
    accuracy = model(X_train, y_train, df_test, predict_true, 
          ovs1=ovs1, ovs2=ovs2,
          smote_random=smote_random, pca_random=pca_random, ext_random=ext_random,
          min_samples_leaf=min_samples_leaf,
          min_samples_split=min_samples_split,
          n_estimators=n_estimators
          )
    
    # Save the results to a text file
    with open('hpt_res.txt', 'a') as f:
        f.write(f"ovs1: {ovs1}, ovs2: {ovs2}, " 
                + f"smote_random: {smote_random}, pca_random: {pca_random}, ext_random: "
                + f"{ext_random}, min_samples_leaf: {min_samples_leaf}, min_samples_split: {min_samples_split}, " 
                + f"n_estimators: {n_estimators}, "
                + f"accuracy: {accuracy}\n")
    
    print(f"Score: {accuracy}")

    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(study.best_trial)
