#  Imports

from pathlib import Path
import os
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

reviews_path = "C:\Big Data\A3\Data\reviews" # path to all the raw review files
meta_path = "C:\Big Data\A3\Data\meta"       # path to all the meta review files
output_path_pkls = "C:\Big Data\A3\Data\output"   # path for where the first batch of pkl files go,
                                                  # Two subdirectories are made, one for the review pkls and
                                                  # one for the meta
review_pkls_path = "C:\Big Data\A3\Data\output\review_pkl" #path of the review pkl files
meta_pkls_path = "C:\Big Data\A3\Data\output\meta_pkl"     #path of the meta pkl files

categories = ['Unknown', 'Magazine_Subscriptions', 'Movies_and_TV'] # These are the ones that we have left to run


# Function for extracting the arrow files from the tar paths temporarily it is run in the preprocess category function
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

'''
this function is used to extract all the arrow files and break them down to even smaller
sizes if they pass a certain size limit. it then puts them together into tables and pkl file batches
and deletes the temp folder of arrow files
'''
def preprocess_category(review_tar_path, meta_tar_path, output_folder, category,batch_size=1000):
    temp_path = "Data/temp_extract" # change as needed
    os.makedirs(output_folder, exist_ok=True)

    print("Extracting tar files...")
    extract_tar_bz2(review_tar_path, temp_path)
    extract_tar_bz2(meta_tar_path, temp_path)

    arrow_files = list(Path(temp_path).rglob("*.arrow"))
    print(f"Found {len(arrow_files)} Arrow files")

    for arrow_file in arrow_files:
        try:
            is_meta = "meta" in str(arrow_file).lower()
            folder_name = "meta" if is_meta else "reviews"

            pkl_output_path = os.path.join(output_folder, f"{folder_name}_pkl")
            os.makedirs(pkl_output_path, exist_ok=True)

            # print(f"Streaming {arrow_file.name} → {parquet_output_path}")
            dataset = load_dataset("arrow", data_files=str(arrow_file), split="train", streaming=True)

            batch = []
            seen_keys = set()
            batch_num = 0

            for i, row in enumerate(dataset):
                if not row:
                    continue

                if not is_meta:
                    key = (row.get("user_id"), row.get("asin"), row.get("text"))
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                batch.append(row)

                if len(batch) >= batch_size:
                    table = pa.Table.from_pylist(batch)
                    # pq.write_to_dataset(table, root_path=parquet_output_path)

                    # convert to pandas and save as .pkl batch
                    df = pd.DataFrame(batch)
                    df.to_pickle(os.path.join(pkl_output_path, f"{category}_batch_{batch_num}.pkl"))
                    print(f"Saved batch {batch_num} ({len(batch)} rows) to .pkl")
                    batch = []
                    batch_num += 1

            # Final batch
            if batch:
                table = pa.Table.from_pylist(batch)
                # pq.write_to_dataset(table, root_path=parquet_output_path)

                df = pd.DataFrame(batch)
                df.to_pickle(os.path.join(pkl_output_path, f"{category}_batch_{batch_num}.pkl"))
                print(f"Saved final batch {batch_num} ({len(batch)} rows)")

        except Exception as e:
            print(f"Error processing {arrow_file.name}: {e}")

    shutil.rmtree(temp_path)
    print("All done, temp folder removed.")

# This function handles the extraction of the brands
def extract_brand(details, store):
    try:
        if isinstance(details, dict) and "brand" in details and details["brand"]:
            return details["brand"]
    except Exception:
        pass
    if isinstance(store, str) and store.strip():
        return store
    return "Unknown"

# This function converts the pkl files to dataframes by taking in the pkl directories
# feed the review pkl path with the category to here followed by the meta with the category
# there is 2 separate calls to this function in the for loop below
# the data frames returned need to be stored to be merged and then cleaned
def convert_to_df(folder, category):
    df_r = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".pkl") and category.lower() in fname.lower():
            try:
                file_path = os.path.join(folder, fname)
                review_df = pd.read_pickle(file_path)
                print(f"{fname} loaded: shape = {review_df.shape}")
                df_r.append(review_df)
            except Exception as e:
                print(f"Error in {fname}:", e)

    if df_r:
        review_df = pd.concat(df_r, ignore_index=True)
        print("All .pkl files loaded. Final shape:", review_df.shape)

    print("Removed reviews pkl folder")
    return review_df

# this function merges the review and meta data dataframes, cleans them 
# and returns the datframe made to ensure that it was put together and contains data
def clean_data(category):
    output_dir = r"D:/UWI/Year 3/Sem 2/COMP3610-Big-Data/Assignments/Assignment#3/A3/datasets/output_folder/cleaned"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Merging review and meta...")
    merged_df = pd.merge(review_df, meta_df, on="parent_asin", how="inner")
    print("Merged")

    print("Filtering invalid ratings...")
    merged_df = merged_df[merged_df["rating"].between(1.0, 5.0, inclusive="both")]

    print("Dropping empty review text...")
    merged = merged_df[merged_df["text"].notna() & (merged_df["text"].str.strip() != "")]

    print("Extracting brand from metadata...")
    merged["brand"] = merged.apply(lambda row: extract_brand(row.get("details"), row.get("store")), axis=1)

    print("Removing duplicate reviews...")
    merged.drop_duplicates(subset=["user_id", "asin", "text"], keep="first", inplace=True)

    print("Computing review length...")
    merged["review_length"] = merged["text"].str.split().apply(len)

    print("Extracting year from timestamp...")
    merged["year"] = pd.to_datetime(merged["timestamp"], unit="ms", errors="coerce").dt.year

    output_file = os.path.join(output_dir, f"{category}_cleaned_merged.pkl.bz2")
    merged.to_pickle(output_file, compression="bz2")

    print(" All cleaning steps completed.")
    
    test = merged
    return test



# For loop to iterate the categories, clean them and convert to compressed pkl zips
# also removes the uncompressed files from the system once they've been done
for category in categories:
    base_path = r"" # replace with path to tar files
    meta_path = r"" # replace with path to meta pkl files
    review_path = r"" # replace with path to review pkl files

    # review pkled folder
    rev_pkl  = r"/root/Data/output_folder musical-video_games/reviews_pkl" # Make sure this is the folder with review .pkl batches
    meta_pkl = r"/root/Data/output_folder musical-video_games/meta_pkl"  # Make sure this is the folder with meta .pkl batches

    # preprocess_category(meta_path, review_path, "output_folder", category)
    review_df = convert_to_df(review_path, category)
    meta_df = convert_to_df(meta_path, category)
    cleaned = clean_data(category, review_df, meta_df)
    print(cleaned)
    del cleaned
    del meta_df
    del review_df

    # remove the review and meta pkl files that aren't compressed
    if os.path.exists(rev_pkl):
        shutil.rmtree(rev_pkl)
    else:
        print(f"{rev_pkl} path does not exist")

    if os.path.exists(meta_pkl):
        shutil.rmtree(meta_pkl)
    else:
        print(f"{meta_pkl} path does not exist")

