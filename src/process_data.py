import os
import zipfile
import pandas as pd
from huggingface_hub import HfApi

RAW_DIR = "data_raw"
OUTPUT_DIR = "data_processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "canada_trade_full.parquet")


# FIND CORRECT CSV
def find_csv_in_zip(zip_path):

    filename = os.path.basename(zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:

        for file in z.namelist():

            # IMPORT → HS10
            if "Imp" in filename and "ODPFN014" in file and file.endswith(".csv"):
                return file

            # EXPORT → HS8
            if "Exp" in filename and "ODPFN017" in file and file.endswith(".csv"):
                return file

    return None


# LOAD DATA
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
                dtype={
                    "HS10": "string",
                    "HS8": "string"
                },
                low_memory=False
            )

    df["trade_type"] = trade_type

    print("Rows loaded:", len(df))

    return df


# FORMAT HS CODE
def format_hs(code):

    if pd.isna(code):
        return None

    code = str(code).strip()

    if not code.isdigit():
        return None

    if len(code) == 8:
        code = code.zfill(8)
        return f"{code[:4]}.{code[4:6]}.{code[6:8]}"

    elif len(code) == 10:
        code = code.zfill(10)
        return f"{code[:4]}.{code[4:6]}.{code[6:8]} {code[8:10]}"

    return None

# CLEAN DATASET
def clean_dataset(df):

    print("Formatting dataset...")

    # Date
    ym = df["YearMonth/AnnéeMois"].astype(str)
    df["date"] = ym.str[:4] + "-" + ym.str[4:6]

    # Rename
    df = df.rename(columns={
        "Country/Pays": "Country",
        "State/État": "State",
        "Value/Valeur": "Value",
        "Quantity/Quantité": "Quantity"
    })

    # CREATE UNIFIED HS COLUMN
    df["HS_raw"] = df["HS10"].fillna(df["HS8"])

    # remove null
    df = df[df["HS_raw"].notna()]

    df["HS_raw"] = df["HS_raw"].astype(str)

    # remove lixo textual comum
    df = df[~df["HS_raw"].str.contains("nan", case=False)]
    df = df[~df["HS_raw"].str.contains("<NA>", case=False)]
    df = df[df["HS_raw"].str.strip() != ""]

    # remove ".0"
    df["HS_raw"] = df["HS_raw"].str.replace(".0", "", regex=False)

    # remove qualquer coisa que não seja número
    df["HS_raw"] = df["HS_raw"].str.replace(r"\D", "", regex=True)

    # mantém apenas HS válidos (8 ou 10 dígitos)
    df = df[df["HS_raw"].str.len().isin([8, 10])]

    # FORMAT FINAL
    df["HS"] = df["HS_raw"].apply(format_hs)

    # remove inválidos após formatação
    df = df[df["HS"].notna()]

    # GARANTE STRING
    df["HS"] = df["HS"].astype("string")

    # Final columns
    df = df[
        [
            "date",
            "HS",
            "Country",
            "Province",
            "State",
            "Value",
            "Quantity",
            "trade_type"
        ]
    ]

    df = df.sort_values("date")

    print("Final rows after cleaning:", len(df))

    return df


# MAIN PIPELINE
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

    print("Saving dataset as PARQUET (preserves schema)...")

    final_df.to_parquet(
        OUTPUT_FILE,
        index=False
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
        path_in_repo="canada_trade_full.parquet",
        repo_id="WilgnerCH/canada-trade-data",
        repo_type="dataset",
        token=hf_token
    )

    print("Upload completed.")


if __name__ == "__main__":
    main()
