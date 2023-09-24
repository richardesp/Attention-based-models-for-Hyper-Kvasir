# Execute: bash scripts/download_dataset.sh
# IMPORTANT: This script must be executed in the root directory of the project

PATH_TO_SAVE_DATASET="./data"

wget https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip -P $PATH_TO_SAVE_DATASET

mkdir -p $PATH_TO_SAVE_DATASET/hyper-kvasir-dataset

unzip $PATH_TO_SAVE_DATASET/hyper-kvasir-labeled-images.zip -d $PATH_TO_SAVE_DATASET/hyper-kvasir-dataset