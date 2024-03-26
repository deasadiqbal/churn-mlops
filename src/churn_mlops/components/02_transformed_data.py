import pandas as pd
import numpy as np
from churn_mlops import logger
from churn_mlops.constants import *
from churn_mlops.utils.common import read_yaml
from sklearn.model_selection import train_test_split

#function to split data
def split_data(df, train_data_path, test_data_path, split_ratio, random_state):
    train, test = train_test_split(df, test_size=split_ratio, random_state=random_state)
    train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    test.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")

#function to save the split data
def split_and_save(config_path):
    config = read_yaml(config_path)

    raw_data = config.data_config.raw_data_path
    train_data_path = config.transformed_data_config.churn_train_data_path
    test_data_path = config.transformed_data_config.churn_test_data_path

    split_ratio = config.data_config.split_ratio
    rnadom_state = config.data_config.random_state

    df = pd.read_csv(raw_data)
    split_data(df, train_data_path, test_data_path, split_ratio, rnadom_state)

    logger.info(f"Data split is done and saved at {train_data_path} and {test_data_path}")

if __name__ == "__main__":
    split_and_save(CONFIG_FILE_PATH)