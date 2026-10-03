import pandas as pd
import numpy as np

DATA_DIR = "../data"

def load_raw():
    train = pd.read_csv(f"{DATA_DIR}/train.csv", parse_dates=["date"])
    stores = pd.read_csv(f"{DATA_DIR}/stores.csv")
    holidays = pd.read_csv(f"{DATA_DIR}/holidays_events.csv", parse_dates=["date"])
    return train, stores, holidays

def build_features():
    train, stores, holidays = load_raw()
    df = train.copy()
    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_payday"] = ((df["date"].dt.day == 15) | df["date"].dt.is_month_end).astype(int)
    df["log_sales"] = np.log1p(df["sales"])

    hol = holidays[(holidays["locale"] == "National") & (~holidays["transferred"])]

    def flag(types):
        s = hol[hol["type"].isin(types)].drop_duplicates("date").set_index("date")
        return pd.Series(1, index=s.index)

    hol_days = (pd.DataFrame({
        "is_holiday": flag(["Holiday", "Transfer", "Additional", "Bridge"]),
        "is_event": flag(["Event"]),
    }).fillna(0).astype(int).rename_axis("date").reset_index())

    df = df.merge(hol_days, on="date", how="left")
    df = df.merge(stores, on="store_nbr", how="left")
    for c in ["is_holiday", "is_event"]:
        df[c] = df[c].fillna(0).astype(int)
    return df