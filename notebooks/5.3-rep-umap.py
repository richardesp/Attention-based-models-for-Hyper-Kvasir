# -*- coding: utf-8 -*-
"""5.3-rep-umap.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1y7oHpg8PdNEb7GTrO4DZfhH3DVu8zuhT
"""

!pip install -U keras-cv-attention-models # Library to use the pre-trained models
!pip install tensorflow-addons

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)

from psutil import virtual_memory

ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('Not using a high-RAM runtime')
else:
  print('You are using a high-RAM runtime!')

"""# Preparing the data"""

import os
import numpy as np
import pandas as pd
import cv2

from google.colab import drive
drive.mount('/content/gdrive')

dataset_dir = "gdrive/MyDrive/hyper-kvasir-dataset-green-patches"

def get_dataCategories(dataset_dir):
    import glob

    categories = []
    for folder_name in os.listdir(dataset_dir):
        if os.path.isdir(os.path.join(dataset_dir, folder_name)):
            nbr_files = len(
                glob.glob(os.path.join(dataset_dir, folder_name) + "/*.jpg")
            )
            categories.append(np.array([folder_name, nbr_files]))

    categories.sort(key=lambda a: a[0])
    cat = np.array(categories)

    return list(cat[:, 0]), list(cat[:, 1])

categories, nbr_files = get_dataCategories(dataset_dir)

# Create DataFrame
df = pd.DataFrame({"categorie": categories, "numbre of files": nbr_files})
print("number of categories: ", len(categories))
df

def get_x_y(datadir, categories, img_wid, img_high):
    X, y = [], []
    for category in categories:
        path = os.path.join(datadir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                ima_resize_rgb = cv2.resize(img_array, (img_wid, img_high))

                X.append(ima_resize_rgb)
                y.append(class_num)

            except Exception as e:
                print(e)

    y = np.array(y)
    X = np.array(X)

    # reshape X into img_wid x img_high x 3
    X = X.reshape(X.shape[0], img_wid, img_high, 3)

    return X, y


img_wid, img_high = 224, 224
X, y = get_x_y(dataset_dir, categories, img_wid, img_high)

print(f"X: {X.shape}")
print(f"y: {y.shape}")

"""## Split the data into train and test"""

from sklearn.model_selection import train_test_split

Y = np.reshape(y, (len(y), 1))


X_fold_0, X_fold_1, y_fold_0, y_fold_1 = train_test_split(
    X, Y, train_size=0.5, random_state=42, stratify=Y
)

k_folds = [[X_fold_0, y_fold_0 ], [X_fold_1, y_fold_1]]

print(f"X_fold_0: {X_fold_0.shape}")
print(f"y_fold_0: {y_fold_0.shape}")
print(f"X_fold_1: {X_fold_1.shape}")
print(f"y_fold_1: {y_fold_1.shape}")

"""# Load the best model

## Split 0
"""

best_model_path = "gdrive/MyDrive/Universidad/TFG/models/Oversampling/Green Patches/MobileViT V2 Large/MobileViTV2Large-split_0-fiery-sweep-6-f1_macro.h5"

from keras_cv_attention_models import mobilevit

model = mobilevit.MobileViT_V2_200(input_shape=(224, 224, 3), num_classes=23)
model.load_weights(best_model_path)

from tensorflow.keras.models import Model

for layer in model.layers:
  print(layer.name)

layer_name = "avg_pool"
prev_layer = model.get_layer(name=layer_name).output
new_model = Model(inputs=model.input, outputs=prev_layer)

# Extrae las características de tus datos de ejemplo (X)
feature_array = new_model.predict(X_fold_0)

!pip install umap-learn

import umap.umap_ as umap

n_neighbors = 15
min_dist = 0.1 
metric = 'minkowski' # correlation, jaccard, minkowski

# Configura UMAP
reductor = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric=metric, transform_seed=3)

# Ajusta UMAP a las características extraídas y transforma los datos
embeddings = reductor.fit_transform(feature_array)

feature_array.shape

embeddings.shape

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

INDEX_TO_LABEL = {
    0: "barretts", 1: "bbps-0-1", 2: "bbps-2-3", 3: "dyed-lifted-polyps",
    4: "dyed-resection-margins", 5: "hemorroids", 6: "ileum", 7: "impacted-stool",
    8: "normal-cecum", 9: "normal-pylorus", 10: "normal-z-line", 11: "oesophagitis-a",
    12: "oesophagitis-b-d", 13: "polyp", 14: "retroflex-rectum", 15: "retroflex-stomach",
    16: "short-segment-barretts", 17: "ulcerative-colities-0-1", 18:"ulcerative-colities-1-2",
    19: "ulcerative-colities-2-3", 20: "ulcerative-colities-grade-1", 21: "ulcerative-colities-grade-2",
    22: "ulcerative-colities-grade-3"
}

# Lista de 23 colores predefinidos
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
          'gold', 'lime', 'teal', 'coral', 'navy', 'maroon', 'yellow', 'dodgerblue', 'hotpink', 
          'chocolate', 'tomato', 'silver', 'indigo']

# Lista de diferentes tipos de marcadores
MARKER_LIST = ['o', 'v', '^', '<', '>', 's']

labels = []

for item in y_fold_0.tolist():
  labels.append(item[0])
  
# Convertimos las etiquetas numéricas a etiquetas de cadena
labels = [INDEX_TO_LABEL[label] for label in labels]

# Creamos un diccionario para mapear cada etiqueta a un color
label_to_color = {label: colors[i] for i, label in enumerate(INDEX_TO_LABEL.values())}

# Creamos el plot
plt.figure(figsize=[10,8])
for i, label in enumerate(INDEX_TO_LABEL.values()):
    idx = np.where(np.array(labels) == label)
    plt.scatter(embeddings[idx, 0], embeddings[idx, 1], color=label_to_color[label], s=10, marker=MARKER_LIST[i%6])

# Eliminamos los ticks del eje x e y
plt.xticks([])
plt.yticks([])

# Añade el valor de n_neighbors y min_dist al gráfico
plt.text(1.05, 0.02, f'n_neighbors: {n_neighbors}\nmin_dist: {min_dist}\nmetric: {metric}', horizontalalignment='left', verticalalignment='bottom', transform=plt.gca().transAxes)

# Creamos la leyenda fuera del gráfico a la derecha
handles = [plt.Line2D([0], [0], marker=MARKER_LIST[i%6], color='w', markerfacecolor=label_to_color[label], markersize=10) for i, label in enumerate(INDEX_TO_LABEL.values())]
plt.legend(handles, INDEX_TO_LABEL.values(), title="Clases", bbox_to_anchor=(1.05, 1), loc='upper left')

# Agregamos un título al gráfico
plt.title('Hyper-Kvasir embebido mediante el algoritmo UMAP')

# Guardamos el gráfico en formato SVG
plt.savefig('output.svg', format='svg', dpi=1200)

plt.show()

labels = []

for item in y_fold_0.tolist():
  labels.append(item[0])

import matplotlib.pyplot as plt
import numpy as np

# Creamos el plot
plt.figure(figsize=[10,8])
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels_encoded)

# Creamos la leyenda
legend1 = plt.legend(*scatter.legend_elements(), title="Clases")
plt.gca().add_artist(legend1)

# Ajustamos las etiquetas de la leyenda a los nombres originales de las clases
for i, label in enumerate(le.classes_):
    legend1.get_texts()[i].set_text(label)

plt.show()

import matplotlib.pyplot as plt

# Visualiza el embedding UMAP con colores para las clases
plt.scatter(embedding[:, 0], embedding[:, 1], c=y_fold_0, cmap='viridis', s=15, alpha=0.5)
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.title('UMAP de las características del modelo preentrenado')
plt.colorbar().set_label('Clase', rotation=270, labelpad=12)
plt.show()

!pip install umap-learn[plot]

import umap.umap_ as umap

mapper = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2).fit(feature_array)

import umap.plot
import numpy as np

INDEX_TO_LABEL = {
    0: "barretts", 1: "bbps-0-1", 2: "bbps-2-3", 3: "dyed-lifted-polyps",
    4: "dyed-resection-margins", 5: "hemorroids", 6: "ileum", 7: "impacted-stool",
    8: "normal-cecum", 9: "normal-pylorus", 10: "normal-z-line", 11: "oesophagitis-a",
    12: "oesophagitis-b-d", 13: "polyp", 14: "retroflex-rectum", 15: "retroflex-stomach",
    16: "short-segment-barretts", 17: "ulcerative-colities-0-1", 18:"ulcerative-colities-1-2",
    19: "ulcerative-colities-2-3", 20: "ulcerative-colities-grade-1", 21: "ulcerative-colities-grade-2",
    22: "ulcerative-colities-grade-3"
}

y_labels = []

for y_item in y:
  y_labels.append(INDEX_TO_LABEL[y_item])

#y_labels = numpy.ndarray(y)

print(y_fold_0.shape)
y = y_fold_0.reshape(-1)
print(y.shape)

ax = umap.plot.points(mapper, labels=y)

umap.plot.connectivity(mapper, show_points=True)

local_dims = umap.plot.diagnostic(mapper, diagnostic_type='local_dim')

umap.plot.diagnostic(mapper, diagnostic_type='neighborhood')

hover_data = pd.DataFrame({'index': np.arange(5341),
                           'label': y[:5341]})

hover_data['item'] = hover_data.label.map(
    {
      0: "barretts",
      1: "bbps-0-1",
      2: "bbps-2-3",
      3: "dyed-lifted-polyps",
      4: "dyed-resection-margins",
      5: "hemorroids",
      6: "ileum",
      7: "impacted-stool",
      8: "normal-cecum",
      9: "normal-pylorus",
      10: "normal-z-line",
      11: "oesophagitis-a",
      12: "oesophagitis-b-d",
      13: "polyp",
      14: "retroflex-rectum",
      15: "retroflex-stomach",
      16: "short-segment-barretts",
      17: "ulcerative-colities-0-1",
      18: "ulcerative-colities-1-2",
      19: "ulcerative-colities-2-3",
      20: "ulcerative-colities-grade-1",
      21: "ulcerative-colities-grade-2",
      22: "ulcerative-colities-grade-3"
    }
)

umap.plot.output_notebook()
p = umap.plot.interactive(mapper, labels=y, hover_data=hover_data, point_size=2)
umap.plot.show(p)

"""## Split 1"""

best_model_path = "gdrive/MyDrive/Universidad/TFG/models/Oversampling/Green Patches/MobileViT V2 Large/MobileViTV2Large-split_1-fiery-sweep-6-f1_macro.h5"

from keras_cv_attention_models import mobilevit

model = mobilevit.MobileViT_V2_200(input_shape=(224, 224, 3), num_classes=23)
model.load_weights(best_model_path)

from tensorflow.keras.models import Model

for layer in model.layers:
  print(layer.name)

layer_name = "avg_pool"
prev_layer = model.get_layer(name=layer_name).output
new_model = Model(inputs=model.input, outputs=prev_layer)

# Extrae las características de tus datos de ejemplo (X)
feature_array = new_model.predict(X_fold_1)

!pip install umap-learn

import umap.umap_ as umap

# Configura UMAP
reductor = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)

# Ajusta UMAP a las características extraídas y transforma los datos
embedding = reductor.fit_transform(feature_array)

import matplotlib.pyplot as plt

# Visualiza el embedding UMAP con colores para las clases
plt.scatter(embedding[:, 0], embedding[:, 1], c=y_fold_1, cmap='viridis', s=15, alpha=0.5)
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.title('UMAP de las características del modelo preentrenado')
plt.colorbar().set_label('Clase', rotation=270, labelpad=12)
plt.show()

!pip install umap-learn[plot]

import umap.umap_ as umap

mapper = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2).fit(feature_array)

import umap.plot

print(y_fold_1.shape)
y = y_fold_1.reshape(-1)
print(y.shape)

umap.plot.points(mapper, labels=y)

umap.plot.connectivity(mapper, show_points=True)

local_dims = umap.plot.diagnostic(mapper, diagnostic_type='local_dim')

umap.plot.diagnostic(mapper, diagnostic_type='neighborhood')

hover_data = pd.DataFrame({'index': np.arange(5341),
                           'label': y[:5341]})

hover_data['item'] = hover_data.label.map(
    {
      0: "barretts",
      1: "bbps-0-1",
      2: "bbps-2-3",
      3: "dyed-lifted-polyps",
      4: "dyed-resection-margins",
      5: "hemorroids",
      6: "ileum",
      7: "impacted-stool",
      8: "normal-cecum",
      9: "normal-pylorus",
      10: "normal-z-line",
      11: "oesophagitis-a",
      12: "oesophagitis-b-d",
      13: "polyp",
      14: "retroflex-rectum",
      15: "retroflex-stomach",
      16: "short-segment-barretts",
      17: "ulcerative-colities-0-1",
      18: "ulcerative-colities-1-2",
      19: "ulcerative-colities-2-3",
      20: "ulcerative-colities-grade-1",
      21: "ulcerative-colities-grade-2",
      22: "ulcerative-colities-grade-3"
    }
)

umap.plot.output_notebook()
p = umap.plot.interactive(mapper, labels=y, hover_data=hover_data, point_size=2)
umap.plot.show(p)