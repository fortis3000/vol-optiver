import numpy as np
import pandas as pd

from tqdm import tqdm


def get_wap(bid_price, ask_price, bid_size, ask_size):
    return (bid_price * ask_size + ask_price * bid_size) / (
        bid_size + ask_size
    )


def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff()


def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return ** 2))


def get_realized_vol(book: pd.DataFrame):
    """Mutable mothod to get past realized volatility"""

    book["wap1"] = get_wap(
        book.bid_price1, book.ask_price1, book.bid_size1, book.ask_size1
    )

    book["wap2"] = get_wap(
        book.bid_price2, book.ask_price2, book.bid_size2, book.ask_size2
    )

    book.loc[:, "log_return1"] = log_return(book["wap1"])
    book.loc[:, "log_return2"] = log_return(book["wap2"])

    # book = book[~book['log_return'].isnull()]
    # to not remove rows both on train and test set
    book = book.fillna(0)

    book = pd.merge(
        book,
        book[["time_id", "log_return1"]]
        .rename({"log_return1": "vol"}, axis=1)
        .groupby("time_id")
        .agg(realized_volatility),
        how="inner",
        on="time_id",
    )

    book = pd.merge(
        book,
        book[["time_id", "log_return2"]]
        .rename({"log_return2": "vol2"}, axis=1)
        .groupby("time_id")
        .agg(realized_volatility),
        how="inner",
        on="time_id",
    )

    return book


# Price features


def get_book_features(files):
    pieces = []

    for f in tqdm(files):
        book = pd.read_parquet(f)

        book = get_realized_vol(book)
        book["stock_id"] = int(f.split("=")[-1])

        del book["seconds_in_bucket"]
        del book["bid_price1"]
        del book["ask_price1"]
        del book["bid_size1"]
        del book["ask_size1"]
        del book["bid_price2"]
        del book["ask_price2"]
        del book["bid_size2"]
        del book["ask_size2"]
        del book["wap1"]
        del book["wap2"]
        del book["log_return1"]
        del book["log_return2"]

        dataset_new = book.groupby(["time_id", "stock_id"]).mean()

        pieces.append(dataset_new)

    dataset_new = pd.concat(pieces).reset_index()

    dataset_new["row_id"] = [
        f"{stock_id}-{time_id}"
        for stock_id, time_id in zip(
            dataset_new["stock_id"], dataset_new["time_id"]
        )
    ]

    return dataset_new
