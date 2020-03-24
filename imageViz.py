import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

model = VGG16()

def getAttr(model,layers_name,image):
    attr = {}
    for eachlayerName in layers_name:
        layer = model.get_layer(eachlayerName)
        customModel = Model(inputs=model.input,outputs=layer.output)

        filters,biases = layer.get_weights()
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)

        output = customModel(image)
        attr[eachlayerName] = [filters,output]
    return attr 




