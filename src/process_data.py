import os
import zipfile
import pandas as pd
from huggingface_hub import HfApi

RAW_DIR = "data_raw"
OUTPUT_DIR = "data_processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "canada_trade_full.csv.gz")


def find_csv_in_zip(zip_path):

    filename = os.path.basename(zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:

        for file in z.namelist():

            # IMPORT dataset (HS10)
            if "Imp" in filename and "ODPFN014" in file and file.endswith(".csv"):
                return file

            # EXPORT dataset (HS10)
            if "Exp" in filename and "ODPFN017" in file and file.endswith(".csv"):
                return file

    return None


def process_zip(zip_path, trade_type):

    csv_name = find_csv_in_zip(zip_path)

    if csv_name is None:
        print("No CSV found in:", zip_path)
        return None

    print("Using dataset:", csv_name)

    with zipfile.ZipFile(zip_path) as z:
        with z.open(csv_name) as f:

            df = pd.read_csv(
                f,
                dtype={"HS10": "string"},  # 🔥 evita perda de zeros
                low_memory=False
            )

    df["trade_type"] = trade_type

    print("Rows loaded:", len(df))

    return df


def clean_dataset(df):

    print("Formatting dataset...")

    # 📅 Date (YYYY-MM)
    ym = df["YearMonth/AnnéeMois"].astype(str)
    df["date"] = ym.str[:4] + "-" + ym.str[4:6]

    # 🔤 Rename columns
    df = df.rename(columns={
        "Country/Pays": "Country",
        "State/État": "State",
        "Value/Valeur": "Value",
        "Quantity/Quantité": "Quantity"
    })

    # 🚨 REMOVE NULLS BEFORE ANYTHING
    df = df[df["HS10"].notna()]

    # convert to string safely
    df["HS10"] = df["HS10"].astype(str)

    # remove invalid values
    df = df[~df["HS10"].str.contains("nan", case=False)]
    df = df[~df["HS10"].str.contains("<NA>", case=False)]
    df = df[df["HS10"].str.strip() != ""]

    # remove ".0"
    df["HS10"] = df["HS10"].str.replace(".0", "", regex=False)

    # keep only numeric HS codes
    df = df[df["HS10"].str.match(r"^\d+$")]

    # ensure 10 digits
    df["HS10"] = df["HS10"].str.zfill(10)

    # 📦 Keep only needed columns
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

    # 📊 Sort
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
        elif "Exp" in file:
            trade = "Export"
        else:
            continue

        print("Processing:", file)

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
