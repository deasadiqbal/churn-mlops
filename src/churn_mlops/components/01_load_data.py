import pandas as pd
import numpy as np
import argparse
from churn_mlops.utils.common import read_yaml
from churn_mlops.constants import *

def load_data(data_path : Path, model_var):
    df = pd.read_csv(data_path)
    df = df[model_var]
    return df

def load_raw_data(config_path):
    config = read_yaml(config_path)
    extrenal_data_path = config.data_config.external_data_path
    raw_data_path = config.data_config.raw_data_path
    model_var = config.data_config.model_var

    df = load_data(extrenal_data_path, model_var)
    df.to_csv(raw_data_path, index=False)

if __name__ == "__main__":
    load_raw_data(CONFIG_FILE_PATH)