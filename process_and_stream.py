# import findspark
import os
from pathlib import Path
import tarfile
import pandas as pd
import shutil
import time, matplotlib.pyplot as plt, seaborn as sns, matplotlib.ticker as ticker
import numpy as np
from datasets import load_dataset
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import json
import dask.dataframe as dd
import tarfile
import gc
from dask.diagnostics import ProgressBar
import shutil





def extract_tar_bz2(tar_path, extract_dir):
    if not os.path.exists(tar_path):
        print(f"Error: File {tar_path} does not exist.")
        return
    if not tar_path.endswith(".tar.bz2"):
        print(f"Error: File {tar_path} is not a .tar.bz2 file.")
        return

    try:
        with tarfile.open(tar_path, "r:bz2") as tar:
            print(f"Extracting {tar_path} to {extract_dir}")
            tar.extractall(path=extract_dir)
    except Exception as e:
        print(f"Error during extraction: {e}")


def convert_to_dd(folder, category):
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.endswith(".parquet") and category.lower() in f.lower()]
    if not files:
        print("No parquet files found")
        return None
    df = dd.read_parquet(files)
    print(f"Loaded {len(files)} files into Dask DataFrame")
    return df


def clean_data_dask(category, review_path, meta_path):
    output_dir = r"C:\Users\maian\Downloads\cleaned_files"
    os.makedirs(output_dir, exist_ok=True)

    print("Reading parquet files as Dask DataFrames")
    review_df = dd.read_parquet(review_path)
    meta_df = dd.read_parquet(meta_path)

    print("Merging review and meta on 'parent_asin'")
    merged = dd.merge(review_df, meta_df, on="parent_asin", how="left")

    print("Filtering bad data")
    if "rating" in merged.columns:
        merged = merged[merged["rating"].between(1, 5)]
    if "text" in merged.columns:
        merged = merged[merged["text"].notnull() & (merged["text"].str.strip() != "")]

    print("Extracting brand")
    def fast_extract_brand(details, store):
        if isinstance(details, dict) and details.get("brand"):
            return details["brand"]
        elif isinstance(store, str) and store.strip():
            return store
        return "Unknown"

    merged["brand"] = merged.map_partitions(
        lambda df: df.apply(lambda row: fast_extract_brand(row.get("details"), row.get("store")), axis=1),
        meta=("brand", "object")
    )

    print("Computing derived columns")
    if "text" in merged.columns:
        merged["review_length"] = merged["text"].str.split().map(
            lambda x: len(x) if x else 0, meta=("review_length", "int")
        )
    if "timestamp" in merged.columns:
        merged["year"] = dd.to_datetime(merged["timestamp"], unit="ms", errors="coerce").dt.year

    print("Selecting necessary columns")
    necessary_columns = [
        "user_id", "asin", "parent_asin", "rating", "text", "verified_purchase",
        "helpful_vote", "review_length", "year", "brand", "main_category",
        "title", "average_rating", "rating_number", "price"
    ]
    merged = merged[[col for col in necessary_columns if col in merged.columns]]

    # print("Repartitioning to reduce write overhead")
    # merged = merged.repartition(npartitions=50)

    # output_file = os.path.join(output_dir, f"{category}_cleaned.parquet")
    # print(f"Saving cleaned file to {output_file}")

    # with ProgressBar():
    #     merged.to_parquet(output_file, compression="snappy", write_index=False, overwrite=True)

    print(f"Done! File cleaned.")

    return merged



def preprocess_category(review_tar_path, meta_tar_path, output_folder, category, batch_size=1000):
    temp_path = os.path.join(output_folder, "temp_extract", category)
    os.makedirs(temp_path, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    print(f"Extracting tar files for {category}...")
    extract_tar_bz2(review_tar_path, temp_path)
    extract_tar_bz2(meta_tar_path, temp_path)

    arrow_files = list(Path(temp_path).rglob("*.arrow"))
    print(f"Found {len(arrow_files)} Arrow files")

    batch_num = 0
    total_rows = 0

    for arrow_file in arrow_files:
        try:
            is_meta = "meta" in str(arrow_file).lower()
            folder_name = "meta" if is_meta else "reviews"
            out_path = os.path.join(output_folder, f"{folder_name}_parquet")
            os.makedirs(out_path, exist_ok=True)

            dataset = load_dataset("arrow", data_files=str(arrow_file), split="train", streaming=True)

            batch = []
            seen_keys = set()

            for row in dataset:
                if not row:
                    continue
                if not is_meta:
                    key = (row.get("user_id"), row.get("asin"), row.get("text"))
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                batch.append(row)

                if len(batch) >= batch_size:
                    df = pd.DataFrame(batch)
                    df.to_parquet(os.path.join(out_path, f"{category}_batch_{batch_num}.parquet"), index=False)
                    print(f"Saved batch {batch_num} ({len(batch)} rows)")
                    batch = []
                    batch_num += 1
                    total_rows += 1

            if batch:
                df = pd.DataFrame(batch)
                df.to_parquet(os.path.join(out_path, f"{category}_batch_{batch_num}.parquet"), index=False)
                print(f"Saved final batch {batch_num} ({len(batch)} rows)")

        except Exception as e:
            print(f"Error processing {arrow_file.name}: {e}")

    shutil.rmtree(temp_path)
    print(f"Temp folder removed: {temp_path}")

def process_and_stream(category, review_tar, meta_tar, output_dir):
    try:
        preprocess_category(review_tar, meta_tar, output_dir, category)

        review_df = convert_to_dd(os.path.join(output_dir, "reviews_parquet"), category)
        meta_df = convert_to_dd(os.path.join(output_dir, "meta_parquet"), category)

        if review_df is not None and meta_df is not None:
            cleaned = clean_data_dask(category, review_df, meta_df)

            # EITHER: write full cleaned category to disk
            cleaned.to_parquet(f"cleaned_data/{category}.parquet", compression="snappy")

            # OR: write small sample or stats only
            sample = cleaned.sample(frac=0.01, random_state=42)
            sample.to_parquet(f"samples/{category}_sample.parquet", compression="snappy")

            # BONUS: write summary stats only
            stats = cleaned.groupby("rating").size().compute()
            stats.to_csv(f"stats/{category}_rating_counts.csv")

        print(f"✅ Finished {category}")
    except Exception as e:
        print(f"❌ Failed {category}: {e}")
    finally:
        gc.collect()

