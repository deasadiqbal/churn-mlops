import joblib
import pandas as pd
from urllib.parse import urlparse

from churn_mlops import logger
from churn_mlops.constants import *
from churn_mlops.utils.common import read_yaml

from sklearn.ensemble import RandomForestClassifier


# function to load features and target
def load_features_target(df, target):
    x = df.drop(target, axis=1)
    y = df[[target]]
    return x,y

# function to traine and evaluate the model
def train_and_evaluate(config_path):
    # read the config 
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
    test_df = pd.read_csv(test_data_path)
    x_train, y_train = load_features_target(train_df, target)
    x_test, y_test = load_features_target(test_df, target)

    model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    model.fit(x_train, y_train)

    joblib.dump(model, config.model_dir)
    logger.info(f"Model is saved at {config.model_dir}")

if __name__ == "__main__":
    train_and_evaluate(CONFIG_FILE_PATH) 