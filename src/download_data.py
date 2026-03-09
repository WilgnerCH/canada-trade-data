import os
import requests

# Pasta onde os arquivos serão salvos
DATA_DIR = "data_raw"

# Criar pasta se não existir
os.makedirs(DATA_DIR, exist_ok=True)

# URLs oficiais do Statistics Canada
FILES = {
    "CIMT-CICM_Imp_2023.zip": "https://www150.statcan.gc.ca/n1/pub/71-607-x/2021004/zip/CIMT-CICM_Imp_2023.zip",
    "CIMT-CICM_Imp_2024.zip": "https://www150.statcan.gc.ca/n1/pub/71-607-x/2021004/zip/CIMT-CICM_Imp_2024.zip",
    "CIMT-CICM_Imp_2025.zip": "https://www150.statcan.gc.ca/n1/pub/71-607-x/2021004/zip/CIMT-CICM_Imp_2025.zip",

    "CIMT-CICM_Tot_Exp_2023.zip": "https://www150.statcan.gc.ca/n1/pub/71-607-x/2021004/zip/CIMT-CICM_Tot_Exp_2023.zip",
    "CIMT-CICM_Tot_Exp_2024.zip": "https://www150.statcan.gc.ca/n1/pub/71-607-x/2021004/zip/CIMT-CICM_Tot_Exp_2024.zip",
    "CIMT-CICM_Tot_Exp_2025.zip": "https://www150.statcan.gc.ca/n1/pub/71-607-x/2021004/zip/CIMT-CICM_Tot_Exp_2025.zip",
}

def download_file(filename, url):
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        print(f"{filename} already exists. Skipping.")
        return

    print(f"Downloading {filename}...")

    response = requests.get(url, stream=True)

    if response.status_code != 200:
        print(f"Error downloading {filename}")
        return

    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"{filename} downloaded successfully.")

def main():
    for filename, url in FILES.items():
        download_file(filename, url)

if __name__ == "__main__":
    main()
