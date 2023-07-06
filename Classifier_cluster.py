# Classifier_cluster

import xgboost as xgb
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# CHOOSE THE FILTER
Filter = "Original" # ["Original", "SquareRoot", "Logarithm", "Exponential", "Wavelet", "Squared"]
    
# CHOOSE YOUR INPUT
#single_acquisitions = ["ADC", "DTI_eddy_FA", "FLAIR", "SWI", "DWI", "T1_contrast", "DTI_eddy_MD", "ASL", "T1", "T2"]
single_acquisitions = ["ADC", "FLAIR"]

def get_paths(acquisitions):
    environment = os.environ.get('PYRADIOMICS_ENV', 'local')  # Default to 'local' if the environment variable is not set

    suffixes = [acquisition + ".csv" for acquisition in acquisitions]
    
    if environment == 'local':
        base_directory = "/Users/Gabriel/MSc_Dissertation/pyRadiomics/"
        train_features_paths = [os.path.join(base_directory, "Training/extracted_features/" + Filter + "/training_firstorder_" + Filter.lower() + "_" + suffix) for suffix in suffixes]
        train_data_path = os.path.join(base_directory, "Training/training_data.csv")
        validation_features_paths = [os.path.join(base_directory, "Validation/extracted_features/" + Filter + "/validation_firstorder_" + Filter.lower() + "_" + suffix) for suffix in suffixes]
        validation_data_path = os.path.join(base_directory, "Validation/validation_data.csv")
        eval_dir = os.path.join(base_directory,"Model_performance", f'{Filter}_XGBoost_evaluation.csv')
    elif environment == 'cluster':
        base_directory = "/cluster/project2/UCSF_PDGM_dataset/"
        train_features_paths = [os.path.join(base_directory, "Training/extracted_features/" + Filter + "/training_firstorder_" + Filter.lower() + "_" + suffix) for suffix in suffixes]
        train_data_path = os.path.join(base_directory, "Training/training_data.csv")
        validation_features_paths = [os.path.join(base_directory, "Validation/extracted_features/" + Filter + "/validation_firstorder_" + Filter.lower() + "_" + suffix) for suffix in suffixes]
        validation_data_path = os.path.join(base_directory, "Validation/validation_data.csv")
        eval_dir = os.path.join(base_directory,"Model_performance", f'{Filter}_XGBoost_evaluation.csv')
    else:
        raise ValueError("Unknown environment: please set PYRADIOMICS_ENV to 'local' or 'cluster'")

    return base_directory, train_features_paths, train_data_path, validation_features_paths, validation_data_path, eval_dir

def process_target_columns(df):
    df = df.copy()
    grade_mapping = {2: 0, 3: 1, 4: 2}
    df["WHO CNS Grade"] = df["WHO CNS Grade"].map(grade_mapping)
    
    # Update 'IDH' values to be dichotomous 
    df.loc[df["IDH"] != "wildtype", "IDH"] = "mutant"

    # Encode IDH column in 1s and 0s
    df["IDH"] = pd.Categorical(df["IDH"], categories=["mutant", "wildtype"]).codes

    return df

def load_data(train_features_paths, train_data_path, validation_features_paths, validation_data_path, normalize=False):
    # Extract unique identifier from path
    def get_id(path):
        filename = os.path.basename(path)
        identifier = filename.replace(".csv", "")
        return identifier

    # Load data training 
    train_features_data = pd.concat([(pd.read_csv(path).add_suffix('_' + get_id(path))).rename(columns={pd.read_csv(path).add_suffix('_' + get_id(path)).columns[0]: "ID"}).astype(float).set_index("ID") for path in train_features_paths], axis=1)
    train_meta_data = pd.read_csv(train_data_path)
    train_meta_data["ID"] = train_meta_data["ID"].str.extract('(\d+)').astype(int)
    train_meta_data.set_index("ID", inplace=True)
    train_merged_data = train_meta_data[['Sex', 'Age at MRI', 'WHO CNS Grade', 'IDH']].merge(train_features_data, left_index=True, right_index=True)
    
    if normalize:
        columns_to_exclude = ['ID', 'Sex','WHO CNS Grade', 'IDH']
        train_merged_data = normalize_data(train_merged_data, columns_to_exclude)

    train_target_columns = process_target_columns(train_merged_data[["WHO CNS Grade", "IDH"]])
    X_train = train_merged_data.drop(train_target_columns, axis=1)
    X_train["Sex"] = pd.Categorical(X_train["Sex"], categories=["M", "F"]).codes 
    
    y_train = train_target_columns

    # Load data validation
    validation_features_data = pd.concat([(pd.read_csv(path).add_suffix('_' + get_id(path))).rename(columns={pd.read_csv(path).add_suffix('_' + get_id(path)).columns[0]: "ID"}).astype(float).set_index("ID") for path in validation_features_paths], axis=1)
    validation_meta_data = pd.read_csv(validation_data_path)
    validation_meta_data["ID"] = validation_meta_data["ID"].str.extract('(\d+)').astype(int)  # Add this line to convert 'ID' to int
    validation_meta_data.set_index("ID", inplace=True)
    validation_merged_data = validation_meta_data[['Sex', 'Age at MRI', 'WHO CNS Grade', 'IDH']].merge(validation_features_data, left_index=True, right_index=True)

    if normalize:
        columns_to_exclude = ['ID', 'Sex','WHO CNS Grade', 'IDH']
        validation_merged_data = normalize_data(validation_merged_data, columns_to_exclude)
    
    validation_target_columns = process_target_columns(validation_merged_data[["WHO CNS Grade", "IDH"]])
    X_validation = validation_merged_data.drop(["WHO CNS Grade", "IDH"], axis=1)
    X_validation["Sex"] = pd.Categorical(X_validation["Sex"], categories=["M", "F"]).codes 
    
    y_validation = process_target_columns(validation_merged_data[["WHO CNS Grade", "IDH"]])
    
    return X_train, y_train, X_validation, y_validation

def normalize_data(df, columns_to_exclude):
    '''Andrew Ng suggests that gradient descent generally runs faster when features are rescaled, i.e. normalized and centred
    near 0. Although this is not necessarily relevant for XGBoost, which is a decision tree based method, 
    I add the normalization option here
    IMPORTANT: feature normalization decreased model performance with xgboost for Grade and IDH in this pipeline'''
    df_copy = df.copy()
    scaler = StandardScaler()
    for column in df_copy.columns:
        if column not in columns_to_exclude:
            df_copy[column] = scaler.fit_transform(df_copy[column].values.reshape(-1,1))
    return df_copy

def train_model(X_train, y_train):
    params = {
        "objective": "multi:softmax",
        "learning_rate": 0.1,
        "max_depth": 5,
        "n_estimators": 100,
        "seed": 42,
        "num_class": 3, # has to be the value of the case with the most categories (here: WHO grades (3))
    }

    xgb_clf = xgb.XGBClassifier(**params)
    clf = MultiOutputClassifier(xgb_clf)
    
    clf.fit(X_train, y_train)
    return clf

def make_predictions(clf, X_test):
    y_pred = clf.predict(X_test)
    return y_pred

def save_evaluation(acquisition, target_columns, y_validation, y_pred, base_directory):
    # Save report of model performance to txt file
    output_directory = os.path.join(base_directory,"Model_performance")
    os.makedirs(output_directory, exist_ok=True)

    # Save report of model performance to txt file
    for i, col in enumerate(target_columns):
        report = classification_report(y_validation[col], y_pred[:, i])
        matrix = confusion_matrix(y_validation[col], y_pred[:, i])
        accuracy = balanced_accuracy_score(y_validation[col], y_pred[:, i])

        content = "Balanced accuracy for {col}: {accuracy}\n\n".format(col=col, accuracy=accuracy)
        content += "Classification Report for {col}:\n{report}\n".format(col=col, report=report)
        content += "Confusion Matrix for {col}:\n{matrix}\n".format(col=col, matrix=matrix)

        file_name = acquisition[0] + "_{}_results.txt".format(col.replace('/', '_'))
        file_path = os.path.join(output_directory, file_name)
        with open(file_path, "w") as f:
            f.write(content)
            
def main():
    results = []
    for r in range(1, len(single_acquisitions)+1):
        for acquisitions in combinations(single_acquisitions, r):
            # Get paths
            base_directory, train_features_paths, train_data_path, validation_features_paths, validation_data_path, eval_dir = get_paths(acquisitions)

            # Load data
            X_train, y_train, X_validation, y_validation = load_data(train_features_paths, train_data_path, validation_features_paths, validation_data_path)

            # Train the model
            clf = train_model(X_train, y_train)

            # Make predictions on the validation set
            y_pred = make_predictions(clf, X_validation)

            # Assess model performance
            target_columns = ["WHO CNS Grade", "IDH"]
            result = {
                "Acquisition": acquisitions,
            }
            
            for i, col in enumerate(target_columns):
                # Update the values to your results dictionary.
                result.update({
                    col + " balanced-accuracy": balanced_accuracy_score(y_validation[col], y_pred[:, i]),
                    f"{col} f1-score": f1_score(y_validation[col], y_pred[:, i], average='micro', zero_division=0)
                })

            results.append(result)

            # Save detailed evaluation in text file
            #save_evaluation(acquisitions, target_columns, y_validation, y_pred, base_directory)


    results_df = pd.DataFrame(results)
    
    
    return results_df, eval_dir

results_df, eval_dir = main()

# Save results of model performance to csv
results_df.to_csv(eval_dir, index=False)

