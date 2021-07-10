"""The module trains LightGBM model"""
import os
import shutil
import glob
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

import lightgbm as lgb

from helpers import load_config, get_dt_str, seed_everything, rmspe_np
from logger import logger
from features import get_book_features

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Load configs
CONFIG_FILE = "config.yml"
CONFIG_FILE_LGBM = "lgbm/config_lgbm.yml"

config = load_config(CONFIG_FILE)
config.update(load_config(CONFIG_FILE_LGBM))

model_folder = os.path.join(config["MODEL_PATH"], "lgbm" + "_" + get_dt_str())

os.mkdir(model_folder)
shutil.copy2(r"lgbm/main.py", model_folder)
shutil.copy2(CONFIG_FILE, model_folder)
shutil.copy2(CONFIG_FILE_LGBM, model_folder)

seed_everything(config["SEED"])

params = {
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


if __name__ == "__main__":

    logger.info("Script starts properly")
    logger.info("Building dataset")

    dataset = pd.merge(
        pd.read_parquet(
            os.path.join(config["DATAPATH"], "trade_train.parquet")
        ),
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

    X_train, X_test, Y_train, Y_test = train_test_split(
        dataset_new[features],
        dataset_new["target"],
        test_size=0.3,
        train_size=0.7,
        random_state=config["SEED"],
        shuffle=True,
    )

    del dataset
    del dataset_new

    train_ds = lgb.Dataset(data=X_train, label=Y_train)
    val_ds = lgb.Dataset(data=X_train, label=Y_train)

    del X_train
    del Y_train

    max_depth = int(np.sqrt(len(features)))
    num_leaves = 2 ** max_depth if max_depth <= 10 else 2 ** 10

    logger.info("Start training")

    model = lgb.train(
        params=params,
        train_set=train_ds,
        valid_sets=[val_ds],
        valid_names=["val"],
        categorical_feature="auto",
        early_stopping_rounds=100,
    )

    logger.info("Start predicting")

    preds = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(Y_test, preds),
        "mse": mean_squared_error(Y_test, preds),
        "r2": r2_score(Y_test, preds),
        "rmspe": rmspe_np(Y_test, preds),
    }

    logger.info(f"Metrics: {metrics}")

    # Saving results
    logger.info("Saving results")

    with open(os.path.join(model_folder, "metrics.json"), "w") as f:
        json.dump(metrics, f)

    logger.info("Saving model")
    model.save_model(os.path.join(model_folder, "model.txt"))
