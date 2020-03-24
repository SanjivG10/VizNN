# coding: utf-8

from flask import Flask,render_template,request
import random
from PIL import Image
import numpy as np
import math 
import os 

app = Flask(__name__)

layers_name = ['block1_conv2','block2_conv2','block3_conv3','block4_conv3','block5_conv3']


def visualize(imagename):

    #images with layers in the format {layer_name}_{imageNumber} is generated in images folder
    from imageViz import getAttr,model
    im = Image.open(imageName)
    im = im.resize((224,224))
    image = np.array(im,dtype=np.float32)
    image = np.expand_dims(image,0)
    filterMapDict = getAttr(model,layers_name,image)

    for layer in layers_name:
        filters,outputs = filterMapDict[layer][0], filterMapDict[layer][1]
        outputs = np.array(outputs[0,:,:,:])

        for x in range(outputs.shape[-1]):
            myImage = outputs[:,:,x]
            im = Image.fromarray(myImage)
            im=im.convert('RGB')
            im.save("static/images/{}_{}.jpeg".format(layer,x+1))



images_names = os.listdir('static/images/')

def getImagesFromLayerName(layer_name):
    images_names = [eachImageName for eachImageName in os.listdir('static/images/') if eachImageName.startswith(layer_name)]
    images_names.sort(key=lambda x: int(''.join([ s for s in x if s.isdigit()] )))
    images_names = [images_names[i:i+3] for i in range(0,len(images_names),3) ]
    return images_names


@app.route('/cnn',methods=['GET', 'POST'])
def cnn():
    if request.method == 'POST':
        layer_name = request.form.get('blockName')
        return render_template('image.html',input_image='flower.png',images_name=getImagesFromLayerName(layer_name),layers_name=layers_name)
    return render_template('image.html',input_image='flower.png',images_name=getImagesFromLayerName('block1_conv2'),layers_name=layers_name)

def sigmoid(x):
    return 1/(1+math.exp(-x))

def relu(x):
    return max(0,x)

def feed_forward(x,w=1,b=2):
    return w*x+b

def loss(y_pred,y_actual):
    return 0.5*(y_pred-y_actual)**2


# random.seed(5)

# data=[]
# dataSending = {}
# dataSending["label"]='MSE'
# dataSending["X"] = [x for x in range(-10,11,1)]
# dataSending["index"] = 0 
# dataSending["color"] = '#000' 
# dataSending["Y"] = [loss(x,x*0.98) for x in dataSending.get('X')]

# dataSending2 = {}
# dataSending2["label"]='Feed Forward Sigmoid'
# dataSending2["X"] = [x for x in range(-10,11,1)]
# dataSending2["index"] = 1
# dataSending2["color"] = '#000' 
# dataSending2["Y"] = [loss(sigmoid(feed_forward(x)),0.5) for x in dataSending.get('X')]

# dataSending3 = {}
# dataSending3["label"]='Feed Forward ReLU'
# dataSending3["X"] = [x for x in range(-10,11,1)]
# dataSending3["index"] = 2
# dataSending3["color"] = '#000' 
# dataSending3["Y"] = [loss(relu(feed_forward(x)),0.5) for x in dataSending.get('X')]

# data.append(dataSending)
# data.append(dataSending2)
# data.append(dataSending3)

@app.route('/')
def home():
    return render_template('index.html',data=[])


if __name__=='__main__':
    app.run(debug=True)