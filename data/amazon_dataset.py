import os
import json
import gzip
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm


AMAZON_DATASETS = {
    "beauty":   "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/All_Beauty.json.gz",
    "sports":   "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Sports_and_Outdoors.json.gz",
    "toys":     "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Toys_and_Games.json.gz",
    "movies":   "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/Movies_and_TV.json.gz",
    "ml-1m":    None,  # fallback synthetic
}


class AmazonDataset:
    """
    Loads Amazon review data and returns interaction DataFrames.

    Each row: user_id (int), item_id (int), rating (float), timestamp (int).
    Also exposes user2idx / item2idx mappings.
    """

    def __init__(self, category: str = "beauty", data_dir: str = "data/raw", min_interactions: int = 5):
        self.category = category
        self.data_dir = Path(data_dir)
        self.min_interactions = min_interactions
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.df: pd.DataFrame = None
        self.user2idx: dict = {}
        self.item2idx: dict = {}
        self.n_users: int = 0
        self.n_items: int = 0

    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        raw = self._get_raw()
        df = self._filter_and_encode(raw)
        self.df = df
        return df

    # ------------------------------------------------------------------
    def _get_raw(self) -> pd.DataFrame:
        cache_path = self.data_dir / f"{self.category}.parquet"
        if cache_path.exists():
            print(f"[AmazonDataset] Loading cached data from {cache_path}")
            return pd.read_parquet(cache_path)

        url = AMAZON_DATASETS.get(self.category)
        if url is None:
            print("[AmazonDataset] No URL for category — generating synthetic data")
            return self._synthetic_data()

        gz_path = self.data_dir / f"{self.category}.json.gz"
        if not gz_path.exists():
            self._download(url, gz_path)

        print(f"[AmazonDataset] Parsing {gz_path} ...")
        records = []
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            for line in tqdm(f, desc="parsing"):
                try:
                    obj = json.loads(line)
                    if "reviewerID" in obj and "asin" in obj:
                        records.append({
                            "user_id": obj["reviewerID"],
                            "item_id": obj["asin"],
                            "rating":  float(obj.get("overall", 1.0)),
                            "timestamp": int(obj.get("unixReviewTime", 0)),
                        })
                except (json.JSONDecodeError, ValueError):
                    continue

        raw = pd.DataFrame(records)
        raw.to_parquet(cache_path)
        return raw

    # ------------------------------------------------------------------
    def _filter_and_encode(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        # k-core filtering
        for _ in range(10):
            user_counts = df["user_id"].value_counts()
            item_counts = df["item_id"].value_counts()
            df = df[df["user_id"].isin(user_counts[user_counts >= self.min_interactions].index)]
            df = df[df["item_id"].isin(item_counts[item_counts >= self.min_interactions].index)]
            if len(df) == len(raw):
                break
            raw = df

        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

        users = sorted(df["user_id"].unique())
        items = sorted(df["item_id"].unique())
        self.user2idx = {u: i for i, u in enumerate(users)}
        self.item2idx = {it: i for i, it in enumerate(items)}
        self.n_users = len(users)
        self.n_items = len(items)

        df["user_id"] = df["user_id"].map(self.user2idx)
        df["item_id"] = df["item_id"].map(self.item2idx)

        print(f"[AmazonDataset] {self.n_users} users | {self.n_items} items | {len(df)} interactions")
        return df

    # ------------------------------------------------------------------
    @staticmethod
    def _download(url: str, dest: Path):
        print(f"[AmazonDataset] Downloading {url} ...")
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    # ------------------------------------------------------------------
    @staticmethod
    def _synthetic_data(n_users: int = 2000, n_items: int = 500, n_interactions: int = 50000) -> pd.DataFrame:
        import numpy as np
        rng = np.random.default_rng(42)
        users = rng.integers(0, n_users, n_interactions).astype(str)
        items = rng.integers(0, n_items, n_interactions).astype(str)
        ratings = rng.integers(1, 6, n_interactions).astype(float)
        timestamps = rng.integers(1_000_000, 2_000_000, n_interactions)
        return pd.DataFrame({"user_id": users, "item_id": items, "rating": ratings, "timestamp": timestamps})