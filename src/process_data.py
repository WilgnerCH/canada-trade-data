import os
import zipfile
import pandas as pd
from huggingface_hub import HfApi

RAW_DIR = "data_raw"
OUTPUT_DIR = "data_processed"

FULL_DATASET = os.path.join(OUTPUT_DIR, "canada_trade_full.csv.gz")
BRAZIL_DATASET = os.path.join(OUTPUT_DIR, "canada_trade_brazil.csv.gz")


def find_csv_in_zip(zip_path):

    with zipfile.ZipFile(zip_path, "r") as z:
        for file in z.namelist():
            if file.endswith(".csv"):
                return file
    return None


def process_zip(zip_path, trade_type):

    csv_name = find_csv_in_zip(zip_path)

    if csv_name is None:
        print(f"No CSV found in {zip_path}")
        return None

    with zipfile.ZipFile(zip_path) as z:
        with z.open(csv_name) as f:
            df = pd.read_csv(f, low_memory=False)

    df["trade_type"] = trade_type

    print(f"Rows loaded: {len(df)}")

    return df


def clean_dates(df):

    col = "YearMonth/AnnéeMois"

    ym = df[col].astype(str)

    df["date"] = ym.str[:4] + "-" + ym.str[4:6]

    df = df.drop(columns=[col])

    return df


def create_hs_levels(df):

    if "HS10" not in df.columns:
        return df

    hs = df["HS10"].astype(str).str.replace(",", "")

    df["HS8"] = hs.str[:8]
    df["HS6"] = hs.str[:6]
    df["HS4"] = hs.str[:4]
    df["HS2"] = hs.str[:2]

    return df


def create_brazil_dataset(df):

    if "Country/Pays" not in df.columns:
        return None

    brazil = df[df["Country/Pays"] == "BR"]

    print("Brazil rows:", len(brazil))

    return brazil


def main():

    print("Starting dataset processing...")

    all_data = []

    for file in os.listdir(RAW_DIR):

        if not file.endswith(".zip"):
            continue

        full_path = os.path.join(RAW_DIR, file)

        if "Imp" in file:
            trade = "Import"
        else:
            trade = "Export"

        print(f"Processing {file}")

        df = process_zip(full_path, trade)

        if df is not None:
            all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)

    print("Cleaning dates...")
    final_df = clean_dates(final_df)

    print("Creating HS levels...")
    final_df = create_hs_levels(final_df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Saving FULL dataset...")
    final_df.to_csv(FULL_DATASET, index=False, compression="gzip")

    print("Creating Brazil dataset...")
    brazil_df = create_brazil_dataset(final_df)

    if brazil_df is not None:
        brazil_df.to_csv(BRAZIL_DATASET, index=False, compression="gzip")

    print("Uploading to HuggingFace...")

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("HF_TOKEN not found")

    api = HfApi()

    api.upload_file(
        path_or_fileobj=FULL_DATASET,
        path_in_repo="canada_trade_full.csv.gz",
        repo_id="WilgnerCH/canada-trade-data",
        repo_type="dataset",
        token=hf_token
    )

    api.upload_file(
        path_or_fileobj=BRAZIL_DATASET,
        path_in_repo="canada_trade_brazil.csv.gz",
        repo_id="WilgnerCH/canada-trade-data",
        repo_type="dataset",
        token=hf_token
    )

    print("Upload completed.")


if __name__ == "__main__":
    main()
