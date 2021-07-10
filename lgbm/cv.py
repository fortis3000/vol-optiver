"""The module realizes LightGBM model hyperparameter tuning"""
import os
import shutil
import glob
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import ShuffleSplit
from sklearn.exceptions import ConvergenceWarning

import hyperopt

from helpers import (
    load_config,
    get_dt_str,
    seed_everything,
    NpEncoder,
)
from logger import logger
from features import get_book_features
from lgbm.helpers import custom_cv

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load configs
CONFIG_FILE = "config.yml"
CONFIG_FILE_LGBM = "lgbm/config_lgbm.yml"

config = load_config(CONFIG_FILE)
config.update(load_config(CONFIG_FILE_LGBM))

model_folder = os.path.join(
    config["MODEL_PATH"], "lgbm_cv" + "_" + get_dt_str()
)

os.mkdir(model_folder)
shutil.copy2(r"lgbm/cv.py", model_folder)
shutil.copy2(CONFIG_FILE, model_folder)
shutil.copy2(CONFIG_FILE_LGBM, model_folder)

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

VALIDATOR = ShuffleSplit(
    n_splits=config["CV"]["n_splits"],
    train_size=config["CV"]["train_size"],
    test_size=config["CV"]["test_size"],
    random_state=config["SEED"],
)

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


def objective_custom(params):
    return np.median(
        custom_cv(
            params, X_train, Y_train, VALIDATOR, fit_params=FIT_PARAMS_CV
        )
    )


if __name__ == "__main__":

    logger.info("Script starts properly")

    max_depth = int(np.sqrt(len(features))) if len(features) > 4 else 2
    num_leaves = 2 ** max_depth if max_depth <= 10 else 2 ** 10

    # Hyperopt space
    SPACE = {
        "max_depth": hyperopt.hp.choice(
            "max_depth", range(max_depth, max_depth * 2)
        ),
        "num_leaves": hyperopt.hp.choice(
            "num_leaves", list(range(num_leaves, num_leaves * 3))
        ),
        "n_estimators": hyperopt.hp.choice("n_estimators", range(500, 2000)),
        "boosting": hyperopt.hp.choice("boosting", ["gbdt", "dart", "goss"]),
        "colsample_bytree": hyperopt.hp.uniform("colsample_bytree", 0.1, 1.0),
        "neg_bagging_fraction": hyperopt.hp.uniform(
            "neg_bagging_fraction", 0.1, 0.9
        ),
        "pos_bagging_fraction": hyperopt.hp.uniform(
            "pos_bagging_fraction", 0.1, 0.9
        ),
        "reg_alpha": hyperopt.hp.uniform("reg_alpha", 0.1, 10),
        "reg_lambda": hyperopt.hp.uniform("reg_lambda", 0.1, 20),
        "min_data_in_leaf": hyperopt.hp.choice(
            "min_data_in_leaf", range(100, 10000)
        ),
        "min_child_samples": hyperopt.hp.choice(
            "min_child_samples", range(2, 30)
        ),
        "min_child_weight": hyperopt.hp.uniform(
            "min_child_weight", 0.001, 0.9
        ),
        "min_split_gain": hyperopt.hp.uniform("min_split_gain", 0.0, 0.9),
    }

    # Cross-validator
    CV = ShuffleSplit(
        n_splits=config["CV"]["n_splits"],
        train_size=config["CV"]["train_size"],
        test_size=config["CV"]["test_size"],
        random_state=config["SEED"],
    )

    # Hyperparameters tunings
    logger.info("Hyperopt optimization")

    # visibility bug in hyperopt
    params_optimal = hyperopt.fmin(
        fn=objective_custom,
        space=SPACE,
        algo=hyperopt.tpe.suggest,
        max_evals=config["learning"]["HYPER_EVALS"],
        rstate=np.random.RandomState(config["SEED"]),
    )

    # early stopping and boosting rounds
    params_optimal.update(FIT_PARAMS_CV)

    cvs = custom_cv(
        params=params_optimal, X=X_train, Y=Y_train, validator=VALIDATOR
    )
    metrics = {
        "mean": np.median(cvs),
        "median": np.median(cvs),
        "std": np.std(cvs),
        "all": cvs,
    }

    # Saving results
    logger.info("Saving results")

    logger.info("Saving params")
    with open(os.path.join(model_folder, "params_optimal.json"), "w") as f:
        json.dump(params_optimal, f, cls=NpEncoder)

    logger.info("Saving metrics")
    with open(os.path.join(model_folder, "metrics.json"), "w") as f:
        json.dump(metrics, f, cls=NpEncoder)
