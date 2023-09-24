"""
Instantiated class for load HyperKvasir dataset from simula.io

:author: Ricardo Espantaleón Pérez
"""
import warnings

from ..base.base_make_dataset import BaseMakeDataset
from ..utils.get_vprint import get_vprint
import os
import pandas as pd
import numpy as np
import glob
import cv2


class MakeDataset(BaseMakeDataset):
    def __init__(self, config, dataset_path="./data", download=True,
                 data_url="https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled"
                          "-images.zip"):
        """
        Parametrized constructor to load the configuration file of the dataset

        :param config: config file where to load all params for load a specific dataset
        :param dataset_path: path where to download the dataset
        :param download: boolean to download the dataset
        :param data_url: url where to download the dataset
        :author: Ricardo Espantaleón Pérez
        """
        super(MakeDataset, self).__init__(config)
        self.data_url = data_url
        self.dataset_path = dataset_path
        self.pre_processed_dataset_path = None
        self.dataset = None
        self.get_training_data = None
        self.get_validation_data = None

        if download:
            os.system("sh ./scripts/download_dataset.sh " + self.data_url + " " + self.dataset_path)

            if os.path.exists(self.dataset_path + "/hyper-kvasir-labeled-images.zip"):
                os.remove(self.dataset_path + "/hyper-kvasir-labeled-images.zip")

            self.dataset_path = self.dataset_path + "/hyper-kvasir-dataset"

    def __load_images_from_folder(self, folder) -> list:
        """
        Method to load images from a folder

        :param folder: string with the path of the folder
        :return: list of images loaded
        :author: Ricardo Espantaleón Pérez
        """
        images = []

        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.resize(img, (512, 512))
                images.append(img)

        return images

    def __detect_green_patches(self, image: np.ndarray, threshold: int = 130) -> bool:
        """
        Method to detect green patches in the image

        :param image: np.ndarray with the image
        :param threshold: integer number to detect green patches
        :return: true if the image has green patches, false otherwise
        :author: Ricardo Espantaleón Pérez
        """
        green_mark_pixel = image[450, 50, :]

        return green_mark_pixel[1] > threshold

    def delete_green_patches(self, image: np.ndarray, threshold: int = 130, epsilon: int = 10) -> None:
        new_image = image.copy()

        if not self.__detect_green_patches(image, threshold=threshold):
            return new_image

        # Creating a mask for each channel
        mask_1 = np.zeros(image.shape, np.uint8)
        mask_2 = np.zeros(image.shape, np.uint8)
        mask_3 = np.zeros(image.shape, np.uint8)

        mask_1 = new_image[:, :, 1] > 160
        mask_2 = new_image[:, :, 2] > 130
        mask_3 = new_image[:, :, 0] < 30

        mask = mask_1 & mask_2 & mask_3
        rect = cv2.rectangle(np.zeros(image.shape, np.uint8), (0, 0), (0, 0), (255, 255, 255), -1)

        for i in range(rect.shape[0]):
            for j in range(rect.shape[1]):
                if mask[i, j]:
                    rect[i, j] = 255

        img = rect.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 50, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # print(f"Number of contours: {len(contours)}")

        for cnt in contours:
            x1, y1 = cnt[0][0]
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            x, y, w, h = cv2.boundingRect(cnt)

            # Min size of the path
            if w >= 50 and h >= 50 and 0 <= x <= 250 and y >= 300:
                mask = cv2.drawContours(img, [cnt], -1, (255, 255, 255), 3)
                # Fill the entire contour
                mask = cv2.fillPoly(img, pts=[cnt], color=(255, 255, 255))

                # Filling the entire green patch to delete
                mask = cv2.rectangle(img, (x - epsilon, y - epsilon), (x + w + epsilon, y + h + epsilon),
                                     (255, 255, 255),
                                     -1)

                # We want to eliminate the green patch
                mask = cv2.bitwise_not(mask)
                new_image = cv2.bitwise_and(new_image, mask)

        return new_image

    def pre_process_dataset(self, delete_green=False,
                            pre_processed_dataset_path="./data/hyper-kvasir-dataset-final", verbose=True) -> None:
        """
        Method to pre-process the dataset. The steps that are performed are:
            1. Reorganize the dataset in a new folder with a specific structure for each class
            2. Delete green patches if delete_green is true
            3. Resize the images to 512x512
            4. Convert the images to RGB
            5. Rename all the images to a sequential number

        :param delete_green: boolean to delete green patches
        :param pre_processed_dataset_path: string with the path of the pre-processed dataset
        :param verbose: boolean to print the progress
        :return: None
        :author: Ricardo Espantaleón Pérez
        """

        vprint = get_vprint(verbose)

        if os.path.exists(pre_processed_dataset_path):
            warnings.warn("The dataset has already been pre-processed")

        self.pre_processed_dataset_path = pre_processed_dataset_path

        if not os.path.exists(self.pre_processed_dataset_path):
            os.mkdir(self.pre_processed_dataset_path)

        # Move all the directories in dataset_path/labeled-images/lower-gj-tract/anatomical-landmarks
        # to pre_processed_dataset_path
        os.system("mv " + self.dataset_path + "/labeled-images/lower-gi-tract/anatomical-landmarks/* "
                  + self.pre_processed_dataset_path)

        # Move all the directories in dataset_path/labeled-images/lower-gj-tract/pathological-findings
        # to final_dataset_path
        os.system("mv " + self.dataset_path + "/labeled-images/lower-gi-tract/pathological-findings/* "
                  + self.pre_processed_dataset_path)

        # Move all the directories in dataset_path/labeled-images/lower-gj-tract/quality-of-mucosal-view
        # to final_dataset_path
        os.system(
            "mv " + self.dataset_path + "/labeled-images/lower-gi-tract/quality-of-mucosal-views/* "
            + self.pre_processed_dataset_path)

        # Move all the directories in dataset_path/labeled-images/lower-gj-tract/therapeutic-interventions
        # to final_dataset_path
        os.system(
            "mv " + self.dataset_path + "/labeled-images/lower-gi-tract/therapeutic-interventions/* "
            + self.pre_processed_dataset_path)

        # Move all the directories in dataset_path/labeled-images/upper-gj-tract/anatomical-landmarks
        # to final_dataset_path
        os.system("mv " + self.dataset_path + "/labeled-images/upper-gi-tract/anatomical-landmarks/* "
                  + self.pre_processed_dataset_path)

        # Move all the directories in dataset_path/labeled-images/upper-gj-tract/pathological-findings
        # to final_dataset_path
        os.system("mv " + self.dataset_path + "/labeled-images/upper-gi-tract/pathological-findings/* "
                  + self.pre_processed_dataset_path)

        # Delete self.dataset_path folder
        os.system("rm -rf " + self.dataset_path)
        self.dataset_path = self.pre_processed_dataset_path

        # Count the number of directories in final_dataset_path
        if len(os.listdir(self.pre_processed_dataset_path)) != 23:
            raise RuntimeError("Error ‼️ Number of directories in final_dataset_path is not 23")

        vprint("Success ✅ Number of classes in final_dataset_path is 23")

        total_images = sum(
            [len(os.listdir(self.pre_processed_dataset_path + "/" + folder))
             for folder in os.listdir(self.pre_processed_dataset_path)])

        vprint(f"The number of images per class is ({total_images} images in total): ")

        for folder in os.listdir(self.pre_processed_dataset_path):
            vprint(
                f"{folder}: {len(os.listdir(self.pre_processed_dataset_path + '/' + folder))} images "
                f"({round(len(os.listdir(self.pre_processed_dataset_path + '/' + folder)) / total_images * 100, 2)} %)")

        if delete_green:
            vprint("Deleting green patches...")

            # Pre-process all the images deleting the green patches

            # List all folder from "../data/hyper-kvasir-dataset-final", not detect the hidden folders
            folders = os.listdir(self.pre_processed_dataset_path)

            for folder in folders:
                current_path = os.path.join(self.pre_processed_dataset_path, folder)

                if os.path.isdir(current_path):
                    vprint(f"Processing folder: {folder}")

                    current_images = self.__load_images_from_folder(current_path)

                    # Do the previous for in one line
                    new_current_images = [self.delete_green_patches(image) for image in current_images]

                    # Erase the previous folder with all the images
                    os.system(f"rm -rf {current_path}")

                    # Create the folder again
                    os.mkdir(current_path)

                    # Save the new images in the same folders name but in the final path
                    for index, new_current_image in enumerate(new_current_images):
                        new_current_image = cv2.cvtColor(new_current_image, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(os.path.join(self.pre_processed_dataset_path, folder, f"{index + 1}.jpg"),
                                    new_current_image)


                else:
                    vprint(f"Folder {folder} is not a directory")

                # current_class_images = load_images_from_folder(folder)

    def get_training_data(self):
        """
        Method to load the training data

        :return: training data
        :author: Ricardo Espantaleón Pérez
        """
        raise NotImplementedError

    def get_validation_data(self):
        """
        Method to load the validation data

        :return: validation data
        :author: Ricardo Espantaleón Pérez
        """
        raise NotImplementedError

    def get_dataset_df(self) -> pd.DataFrame:
        """
        Method to load the dataset in a pandas dataframe

        :return: dataset in a pandas dataframe
        :author: Ricardo Espantaleón Pérez
        """

        categories = []
        for folder_name in os.listdir(self.dataset_path):
            if os.path.isdir(os.path.join(self.dataset_path, folder_name)):
                nbr_files = len(
                    glob.glob(os.path.join(self.dataset_path, folder_name) + "/*.jpg")
                )
                categories.append(np.array([folder_name, nbr_files]))

        categories.sort(key=lambda a: a[0])
        cat = np.array(categories)

        categories, nbr_files = cat[:, 0], cat[:, 1]

        return pd.DataFrame({"category": categories, "nbr_files": nbr_files})
