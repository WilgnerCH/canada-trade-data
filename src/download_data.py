import os
import requests
import datetime

# Pasta onde os arquivos serão salvos
DATA_DIR = "data_raw"

# Criar pasta se não existir
os.makedirs(DATA_DIR, exist_ok=True)

# Detectar ano atual automaticamente
current_year = datetime.datetime.now().year

# Construir lista de arquivos automaticamente
FILES = {}

for year in range(2023, current_year + 1):

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
    for filename, url in FILES.items():
        download_file(filename, url)


if __name__ == "__main__":
    main()
