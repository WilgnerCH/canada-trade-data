import os
import requests
import datetime

# Folder where files will be saved
DATA_DIR = "data_raw"

# Create folder if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Detect current year automatically
current_year = datetime.datetime.now().year

# Build file list automatically
FILES = {}

# Start from 2024
for year in range(2024, current_year + 1):

    FILES[f"CIMT-CICM_Imp_{year}.zip"] = (
        f"https://www150.statcan.gc.ca/n1/pub/71-607-x/2021004/zip/CIMT-CICM_Imp_{year}.zip"
    )

    FILES[f"CIMT-CICM_Tot_Exp_{year}.zip"] = (
        f"https://www150.statcan.gc.ca/n1/pub/71-607-x/2021004/zip/CIMT-CICM_Tot_Exp_{year}.zip"
    )


def download_file(filename, url):

    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        print(f"{filename} already exists. Skipping.")
        return

    print(f"Downloading {filename}...")

    max_retries = 3

    for attempt in range(max_retries):

        try:
            response = requests.get(url, stream=True, timeout=120)

            if response.status_code != 200:
                print(f"Error downloading {filename}")
                return

            with open(filepath, "wb") as f:

                for chunk in response.iter_content(chunk_size=8192):

                    if chunk:
                        f.write(chunk)

            print(f"{filename} downloaded successfully.")
            return

        except requests.exceptions.RequestException as e:

            print(f"Attempt {attempt+1} failed: {e}")

    print(f"Failed to download {filename} after {max_retries} attempts.")


def main():

    print("Starting download of trade data...")

    for filename, url in FILES.items():

        download_file(filename, url)


if __name__ == "__main__":
    main()
