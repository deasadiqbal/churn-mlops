import joblib
import mlflow
import pandas as pd
from urllib.parse import urlparse

from churn_mlops import logger
from churn_mlops.constants import *
from churn_mlops.utils.common import read_yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score,recall_score, precision_score, confusion_matrix, classification_report


# function to evaluate the model
def evaluation_metrics(y_test, pred, avg_method):
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average=avg_method)
    recall = recall_score(y_test, pred, average=avg_method) 
    f1 = f1_score(y_test, pred, average=avg_method)
    cm = confusion_matrix(y_test, pred) 

    target_names = ['0', '1']
    classification_rep = classification_report(y_test, pred, target_names=target_names)

    return accuracy, precision, recall, f1, cm, classification_rep

# function to load features and target
def load_features_target(df, target):
    x = df.drop(target, axis=1)
    y = df[[target]]
    return x,y


def train_and_evaluate(config_path):
    # read the config and params
    config = read_yaml(config_path)

    #data paths and target variable
    train_data_path = config.transformed_data_config.churn_train_data_path
    test_data_path = config.transformed_data_config.churn_test_data_path
    target = config.data_config.target

    #model params
    max_depth = config.random_forest.max_depth
    n_estimators = config.random_forest.n_estimators

    # read the train and test data
    train_df = pd.read_csv(train_data_path)
    x_train, y_train = load_features_target(train_df, target)

    # read the test data
    test_df = pd.read_csv(test_data_path)
    x_test, y_test = load_features_target(test_df, target)

    # random_forest_model = joblib.load(config.model_dir)
    random_forest_model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    random_forest_model.fit(x_train, y_train)
    

    mlflow.set_tracking_uri(config.mlflow_config.mlflow_uri)
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    with mlflow.start_run(run_name=config.mlflow_config.experiment_name):
        pred = random_forest_model.predict(x_test)

        accuracy, precision, recall, f1, cm, classification_rep = evaluation_metrics(y_test, pred, "weighted")
    
        mlflow.log_param("max_depth",max_depth)
        mlflow.log_param("n_estimators", n_estimators)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(random_forest_model, "model", registered_model_name="RandomForestModel")
        else:
            mlflow.sklearn.log_model(random_forest_model, "model")

if __name__ == "__main__":
    train_and_evaluate(CONFIG_FILE_PATH)