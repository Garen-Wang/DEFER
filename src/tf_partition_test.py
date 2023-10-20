import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
# from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.vgg16 import VGG16
from dag_util import construct_model

model = VGG16(weights='imagenet', include_top=True)
model.summary()
layer_parts = ["block1_pool", "block3_pool", "block4_pool", "flatten"]

graph = tf.get_default_graph()

def partition(model):
    with graph.as_default():
        models = []
        for p in range(len(layer_parts) + 1):
            if p == 0:
                start = model.input._keras_history[0].name
            else:
                start = layer_parts[p-1]
            if p == len(layer_parts):
                end = model.output._keras_history[0].name
            else:
                end = layer_parts[p]
            print(f'before construct model, start={start}, end={end}')
            part = construct_model(model, start, end, part_name=f"part{p+1}")
            models.append(part)
        return models

models = partition(model)