import os
import zipfile
import pandas as pd

RAW_DIR = "data_raw"
OUTPUT_FILE = "data_processed/canada_trade_full.csv"

def find_csv_in_zip(zip_path):
    """Find the HS6 dataset inside the zip"""
    with zipfile.ZipFile(zip_path, 'r') as z:
        for file in z.namelist():
            if "ODPFN015" in file and file.endswith(".csv"):
                return file
    return None

def process_zip(zip_path, trade_type):
    """Extract and read the correct CSV from a ZIP file"""
    
    csv_name = find_csv_in_zip(zip_path)

    if csv_name is None:
        print(f"No HS6 file found in {zip_path}")
        return None

    with zipfile.ZipFile(zip_path) as z:
        with z.open(csv_name) as f:
            df = pd.read_csv(f)

    df["trade_type"] = trade_type
    return df


def main():

    all_data = []

    for file in os.listdir(RAW_DIR):

        if not file.endswith(".zip"):
            continue

        full_path = os.path.join(RAW_DIR, file)

        if "Imp" in file:
            trade = "Import"
        else:
            trade = "Export"

        print("Processing:", file)

        df = process_zip(full_path, trade)

        if df is not None:
            all_data.append(df)

    if not all_data:
        print("No data processed.")
        return

    final_df = pd.concat(all_data, ignore_index=True)

    os.makedirs("data_processed", exist_ok=True)

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("Dataset created:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
