import os
import zipfile
import pandas as pd
from huggingface_hub import HfApi

RAW_DIR = "data_raw"
OUTPUT_DIR = "data_processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "canada_trade_full.csv.gz")


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

    print("Rows loaded:", len(df))

    return df


def add_date_column(df):

    col = "YearMonth/AnnéeMois"

    if col not in df.columns:
        return df

    ym = df[col].astype(str)

    df["date"] = ym.str[:4] + "-" + ym.str[4:6] + "-01"

    # mover date para primeira coluna
    cols = list(df.columns)
    cols.insert(0, cols.pop(cols.index("date")))
    df = df[cols]

    return df


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

        print("Processing", file)

        df = process_zip(full_path, trade)

        if df is not None:
            all_data.append(df)

    if not all_data:
        print("No data processed.")
        return

    final_df = pd.concat(all_data, ignore_index=True)

    print("Adding date column...")
    final_df = add_date_column(final_df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Saving dataset...")

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
        compression="gzip"
    )

    print("Dataset created:")
    print("Rows:", len(final_df))

    print("Uploading to HuggingFace...")

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("HF_TOKEN not found.")

    api = HfApi()

    api.upload_file(
        path_or_fileobj=OUTPUT_FILE,
        path_in_repo="canada_trade_full.csv.gz",
        repo_id="WilgnerCH/canada-trade-data",
        repo_type="dataset",
        token=hf_token
    )

    print("Upload completed.")


if __name__ == "__main__":
    main()
