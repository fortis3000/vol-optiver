import os
import datetime as dt
import json
import yaml

import random
import numpy as np


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def get_dt_str():
    return str(dt.datetime.now()).split(".")[0].replace(" ", "_")


def load_config(config_file):
    """Loading YAML config file and parsing to the dictionary"""
    with open(config_file, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def reduce_mem_usage(props):
    """Reduce memory spended on pandas DataFrame
    https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
    """
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            print("******************************")
            print("Column: ", col)
            print("dtype before: ", props[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn - 1, inplace=True)

            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = props[col] - asint
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if (
                        mn > np.iinfo(np.int8).min
                        and mx < np.iinfo(np.int8).max
                    ):
                        props[col] = props[col].astype(np.int8)
                    elif (
                        mn > np.iinfo(np.int16).min
                        and mx < np.iinfo(np.int16).max
                    ):
                        props[col] = props[col].astype(np.int16)
                    elif (
                        mn > np.iinfo(np.int32).min
                        and mx < np.iinfo(np.int32).max
                    ):
                        props[col] = props[col].astype(np.int32)
                    elif (
                        mn > np.iinfo(np.int64).min
                        and mx < np.iinfo(np.int64).max
                    ):
                        props[col] = props[col].astype(np.int64)

            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            print("dtype after: ", props[col].dtype)
            print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props, NAlist


def rmspe_np(y_true, y_pred):
    """Root Mean Squared Percentage Error"""

    return np.sqrt(
        np.nanmean(
            np.square((y_pred - y_true) / y_true)
        )
    )


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)