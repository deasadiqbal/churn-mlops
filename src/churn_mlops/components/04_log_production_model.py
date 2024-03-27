import joblib
import mlflow
from churn_mlops import logger
from churn_mlops.constants import *
from churn_mlops.utils.common import read_yaml
from mlflow.tracking import MlflowClient
from pprint import pprint

# function to select the best model
def production_model(config_path):
    config = read_yaml(config_path)

    mlflow_tracking_uri = config.mlflow_config.mlflow_tracking_uri
    model_dir = config.model_dir
    model_name = config.mlflow_config.registered_model_name

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    runs = mlflow.search_runs(search_all_experiments=True)
    max_accuracy = max(runs["metrics.accuracy"])

    max_accuracy_run_id = list(runs[runs["metrics.accuracy"] == max_accuracy]["run_id"])[0]

    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):

        if mv.run_id == max_accuracy_run_id:
            model_version = mv.version
            logged_model = mv.source
            pprint(mv, indent=4)
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage="Production"
            )
            logger.info(f"Model with name {model_name} and version {model_version} is set to Production stage")
        else:
            model_version = mv.version
            client.transition_model_version_stage(
                name= model_name,
                version=model_version,
                stage="Staging"
            )
            logger.info(f"Model with name {model_name} and version {model_version} is set to Staging stage")

    loaded_model = mlflow.pyfunc.load_model(logged_model)
    joblib.dump(loaded_model, model_dir)
    logger.info(f"Best model is saved at {model_dir}")

if __name__ == "__main__":
    production_model(CONFIG_FILE_PATH)


