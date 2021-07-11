"""The module realizes LightGBM model hyperparameter tuning"""
import os
import shutil
import glob
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.exceptions import ConvergenceWarning

from helpers import (
    load_config,
    get_dt_str,
    seed_everything,
    NpEncoder,
)
from logger import logger
from features import get_book_features
from lgbm.helpers import kfold

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load configs
CONFIG_KFOLD = "lgbm/config_lgbm_kfold.yml"
CONFIG_FILE = "config.yml"
CONFIG_FILE_LGBM = "config_lgbm.yml"

config = load_config(CONFIG_KFOLD)

PARAMS_FODLER = config["KFOLD"]["params_folder"]
PARAMS_FILE = os.path.join(PARAMS_FODLER, "params_optimal.json")

config.update(load_config(os.path.join(PARAMS_FODLER, CONFIG_FILE)))
config.update(load_config(os.path.join(PARAMS_FODLER, CONFIG_FILE_LGBM)))

model_folder = os.path.join(
    config["MODEL_PATH"], "lgbm_kfold" + "_" + get_dt_str()
)

os.mkdir(model_folder)
shutil.copy2(r"lgbm/kfold.py", model_folder)
shutil.copy2(os.path.join(PARAMS_FODLER, CONFIG_FILE), model_folder)
shutil.copy2(PARAMS_FILE, model_folder)
shutil.copy2(CONFIG_KFOLD, model_folder)

seed_everything(config["SEED"])


# model params to provide into CV method
FIT_PARAMS_CV = {
    "learning_rate": config["learning"]["LEARNING_RATE"],
    "random_state": config["SEED"],
    "objective": "regression",
    "boosting": config["learning"]["boosting"],
    "metric": config["learning"]["LEARN_METRICS_STRING"],
    "device": config["DEVICE"],
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
    "n_jobs": config["learning"]["N_JOBS"],
    "num_boost_round": config["learning"]["N_ROUNDS"],
}


# Dataset
logger.info("Building dataset")

dataset = pd.merge(
    pd.read_parquet(os.path.join(config["DATAPATH"], "trade_train.parquet")),
    pd.read_csv(os.path.join(config["DATAPATH"], "train.csv")),
    how="inner",
    on=["stock_id", "time_id"],
)

files = glob.glob("data/book_train.parquet/*")

dataset_new = get_book_features(files)

dataset_new = pd.merge(
    dataset_new,
    dataset[["time_id", "stock_id", "target"]],
    how="inner",
    on=["time_id", "stock_id"],
)

features = ["vol", "vol2"]

X_train = dataset_new[features]
Y_train = dataset_new["target"]

del dataset
del dataset_new

if __name__ == "__main__":

    logger.info("Script starts properly")

    KFOLDER = KFold(
        n_splits=config["KFOLD"]["n_splits"],
        shuffle=True,
        random_state=config["SEED"],
    )
    # Hyperparameters tunings
    logger.info("Training models")

    with open(PARAMS_FILE, "r") as f:
        params = json.load(f)

    artifacts = kfold(
        params=params,
        X=X_train,
        Y=Y_train,
        kfolder=KFOLDER,
        fit_params=FIT_PARAMS_CV,
    )

    cvs = [x["metrics"] for x in artifacts]

    metrics = {
        "mean": np.median(cvs),
        "median": np.median(cvs),
        "std": np.std(cvs),
        "all": cvs,
    }

    # Saving results
    logger.info("Saving results")

    logger.info("Saving models")
    for i, d in enumerate(artifacts):
        d["model"].save_model(os.path.join(model_folder, f"model_{i}.txt"))

    logger.info("Saving metrics")
    with open(os.path.join(model_folder, "metrics.json"), "w") as f:
        json.dump(metrics, f, cls=NpEncoder)
