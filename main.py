from src.data.make_dataset import MakeDataset


def main():
    test = MakeDataset(config="config", dataset_path="./data/hyper-kvasir-dataset", download=False)
    test.pre_process_dataset(delete_green=False, verbose=True)
    print(test.get_dataset_df())


if __name__ == "__main__":
    main()
