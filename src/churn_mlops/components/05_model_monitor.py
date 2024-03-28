import pandas as pd
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, CatTargetDriftTab

from churn_mlops.constants import *
from churn_mlops.utils.common import read_yaml
from churn_mlops import logger

# fumction to monitor the model
def monitor_model(config_path):
    config = read_yaml(config_path)

    train_data_path = config.data_config.train_data_path
    new_data_path = config.data_config.new_data_path
    dashboard_path = config.model_monitor.dashboard_path
    target = config.data_config.target
    monitor_target = config.model_monitor.target_col_name

    ref_data = pd.read_csv(train_data_path)
    new_data = pd.read_csv(new_data_path)

    ref_data = ref_data.rename(columns={target: monitor_target}, inplace= False)
    new_data = new_data.rename(columns={target: monitor_target}, inplace= False)

    data_drift_dashboard = Dashboard(tabs=[DataDriftTab, CatTargetDriftTab])
    data_drift_dashboard.calculate(ref_data, new_data, column_mapping = None)
    data_drift_dashboard.save(dashboard_path)

if __name__ == "__main__":
    monitor_model(config_path=CONFIG_FILE_PATH)