# Attention-based Models for Hyper-Kvasir

**Automatic and accurate analysis of medical images is a subject of great importance in our current society. In particular, this work focuses on gastrointestinal endoscopy images, as the study of these images helps to detect possible health conditions in those regions.**

## Table of Contents

1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Usage](#usage)
4. [Dataset](#dataset)
5. [License](#license)
6. [Acknowledgments](#acknowledgments)

## Installation

To set up the environment and install the necessary dependencies, follow these steps:

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/Attention-based-models-for-Hyper-Kvasir.git
cd Attention-based-models-for-Hyper-Kvasir
```

2. **Create a Conda environment**:

```bash
conda create --name hyperkvasir_env python=3.8
conda activate hyperkvasir_env
```

3. **Install the required packages:**:

```bash
conda install --file requirements.txt
```

## Project Structure

```bash
.
├── LICENSE.txt
├── README.md
├── main.py
├── notebooks
│   ├── Pre-processing notebooks...
│   ├── Model training notebooks...
│   └── Visualization and analysis notebooks...
├── requirements.txt
├── scripts
│   └── download_dataset.sh
└── src
    ├── base
    │   └── base_make_dataset.py
    ├── data
    │   └── make_dataset.py
    ├── features
    │   └── build_features.py
    ├── models
    │   ├── predict_model.py
    │   └── train_model.py
    ├── utils
    │   ├── check_gpu_arm.py
    │   └── get_vprint.py
    └── visualization
        └── visualize.py
```

## Usage

Any training or prediction on images can be performed by executing the provided notebooks in the `notebooks` directory. Navigate to the specific notebook that matches your desired operation:

- For preprocessing tasks, refer to the notebooks prefixed with `1.x-rep`.
- For model training, especially with Vision Transformers (ViT), refer to the notebooks prefixed with `2.x-rep`.
- For initial runs with pre-trained Vision Transformers, use the notebook `3.0-rep-pre-trained-vit-initial-run.ipynb`.
- For visualization and analysis, including attention extraction, UMAP, and importance correlation plots, refer to the notebooks prefixed with `5.x-rep`.

## Dataset

The dataset used in this project can be found at [Hyper-Kvasir Dataset](https://datasets.simula.no/hyper-kvasir/).

### Dataset Details

The dataset can be split into four distinct parts: 
- Labeled image data
- Unlabeled image data
- Segmented image data
- Annotated video data

Each part is further described below:

#### Labeled images 
In total, the dataset contains 10,662 labeled images stored using the JPEG format. The images can be found in the `images` folder. The classes, which each of the images belong to, correspond to the folder they are stored in (e.g., the ’polyp’ folder contains all polyp images, the ’barretts’ folder contains all images of Barrett’s esophagus, etc.). The number of images per class are not balanced, which is a general challenge in the medical field due to the fact that some findings occur more often than others. This adds an additional challenge for researchers, since methods applied to the data should also be able to learn from a small amount of training data. The labeled images represent 23 different classes of findings.

#### Unlabeled Images 
In total, the dataset contains 99,417 unlabeled images. The unlabeled images can be found in the `unlabeled` folder which is a subfolder in the image folder, together with the other labeled image folders. In addition to the unlabeled image files, we also provide the extracted global features and cluster assignments in the Hyper-Kvasir Github repository as Attribute-Relation File Format (ARFF) files. ARFF files can be opened and processed using, for example, the WEKA machine learning library, or they can easily be converted into comma-separated values (CSV) files.

#### Segmented Images 
We provide the original image, a segmentation mask, and a bounding box for 1,000 images from the polyp class. In the mask, the pixels depicting polyp tissue, the region of interest, are represented by the foreground (white mask), while the background (in black) does not contain polyp pixels. The bounding box is defined as the outermost pixels of the found polyp. For this segmentation set, we have two folders, one for images and one for masks, each containing 1,000 JPEG-compressed images. The bounding boxes for the corresponding images are stored in a JavaScript Object Notation (JSON) file. The image and its corresponding mask have the same filename. The images and files are stored in the `segmented images` folder. It is important to point out that the segmented images have duplicates in the `images` folder of polyps since the images were taken from there.

#### Annotated Videos 
The dataset contains a total of 373 videos containing different findings and landmarks. This corresponds to approximately 11.62 hours of videos and 1,059,519 video frames that can be converted to images if needed. Each video has been manually assessed by a medical professional working in the field of gastroenterology and resulted in a total of 171 annotated findings.

## License

The license for the Hyper-Kvasir dataset is **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

More information can be found [here](https://creativecommons.org/licenses/by/4.0/).

## Acknowledgments

We would like to extend our sincere gratitude to:

- Isabel Jiménez-Velasco
- Manuel J. Marín-Jiménez
- Rafael Muñoz-Salinas

for their invaluable contributions and insights that greatly benefited this project.


