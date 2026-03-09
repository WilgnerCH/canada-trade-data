import os
import zipfile
import pandas as pd
from huggingface_hub import HfApi

RAW_DIR = "data_raw"
OUTPUT_DIR = "data_processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "canada_trade_full.csv.gz")


def find_csv_in_zip(zip_path):
    """
    Find the HS6 dataset inside the zip file
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        for file in z.namelist():
            if "ODPFN015" in file and file.endswith(".csv"):
                return file
    return None


def process_zip(zip_path, trade_type):
    """
    Extract and read the correct CSV from a ZIP file
    """

    csv_name = find_csv_in_zip(zip_path)

    if csv_name is None:
        print(f"⚠️ No HS6 file found in {zip_path}")
        return None

    with zipfile.ZipFile(zip_path) as z:
        with z.open(csv_name) as f:
            df = pd.read_csv(
                f,
                low_memory=False
            )

    df["trade_type"] = trade_type

    print(f"   rows loaded: {len(df)}")

    return df


def main():

    print("Starting dataset processing...")

    if not os.path.exists(RAW_DIR):
        print("Raw data folder not found.")
        return

    all_data = []

    for file in os.listdir(RAW_DIR):

        if not file.endswith(".zip"):
            continue

        full_path = os.path.join(RAW_DIR, file)

        if "Imp" in file:
            trade = "Import"
        else:
            trade = "Export"

        print(f"Processing {file} ({trade})")

        df = process_zip(full_path, trade)

        if df is not None:
            all_data.append(df)

    if not all_data:
        print("No data processed.")
        return

    print("Combining datasets...")

    final_df = pd.concat(all_data, ignore_index=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Saving compressed dataset...")

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
        compression="gzip"
    )

    print("Dataset created successfully:")
    print(OUTPUT_FILE)
    print(f"Total rows: {len(final_df)}")

    # Upload to HuggingFace
    print("Uploading dataset to HuggingFace...")

    hf_token = os.environ["HF_TOKEN"]

    api = HfApi()

    api.upload_file(
        path_or_fileobj=OUTPUT_FILE,
        path_in_repo="canada_trade_full.csv.gz",
        repo_id="WilgnerCH/canada-trade-data",
        repo_type="dataset",
        token=hf_token
    )

    print("Upload completed successfully.")


if __name__ == "__main__":
    main()
