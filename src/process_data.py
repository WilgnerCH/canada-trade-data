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

            # Import dataset (HS10)
            if "Imp" in zip_path and "ODPFN014" in file and file.endswith(".csv"):
                return file

            # Export dataset (HS10)
            if "Exp" in zip_path and "ODPFN017" in file and file.endswith(".csv"):
                return file

    return None


def process_zip(zip_path, trade_type):

    csv_name = find_csv_in_zip(zip_path)

    if csv_name is None:
        print(f"No CSV found in {zip_path}")
        return None

    print("Using dataset file:", csv_name)

    with zipfile.ZipFile(zip_path) as z:
        with z.open(csv_name) as f:

            df = pd.read_csv(
                f,
                low_memory=False
            )

    df["trade_type"] = trade_type

    print("Rows loaded:", len(df))

    return df


def clean_dataset(df):

    print("Formatting dataset...")

    # format date
    ym = df["YearMonth/AnnéeMois"].astype(str)
    df["date"] = ym.str[:4] + "-" + ym.str[4:6]

    # rename columns
    df = df.rename(columns={
        "Country/Pays": "Country",
        "State/État": "State",
        "Value/Valeur": "Value",
        "Quantity/Quantité": "Quantity"
    })

    # fix HS10 formatting
    df["HS10"] = (
        df["HS10"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )

    # remove empty HS codes
    df = df[df["HS10"].notna()]
    df = df[df["HS10"] != ""]
    df = df[df["HS10"] != "nan"]

    # ensure 10 digits
    df["HS10"] = df["HS10"].str.zfill(10)

    # keep only necessary columns
    df = df[
        [
            "date",
            "HS10",
            "Country",
            "Province",
            "State",
            "Value",
            "Quantity",
            "trade_type"
        ]
    ]

    # sort dataset
    df = df.sort_values("date")

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

    print("Combining datasets...")

    final_df = pd.concat(all_data, ignore_index=True)

    print("Cleaning dataset...")

    final_df = clean_dataset(final_df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Saving dataset...")

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
        compression="gzip"
    )

    print("Dataset created successfully")
    print("Rows:", len(final_df))

    print("Uploading dataset to HuggingFace...")

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
