import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import os, sys
# import h5py
from keras.models import model_from_json

savemodel_dir = "./_model/"
modelpath = savemodel_dir + "my_model.json"
weightpath = savemodel_dir + "my_model_weights.h5"
#keras
def save_model(model):
    # 目录不存在，创建
    if not os.path.exists(savemodel_dir):
        os.makedirs(savemodel_dir)
    json_string = model.to_json()  # 等价于 json_string = model.get_config()
    open(modelpath, 'w').write(json_string)
    # model.save_weights(weightpath)

def load_model():
    model = model_from_json(open(modelpath).read())
    # model.load_weights(weightpath)
    return model

def save_weight(model):
    model.save_weights(weightpath)

def load_weight(model):
    model.load_weights(weightpath)

def save_weight(model, i):
    model.save_weights(savemodel_dir +"my_model_weights"+str(i)+".h5")

def load_weight(model, i):
    model.load_weights(savemodel_dir +"my_model_weights"+str(i)+".h5")

# score = model.evaluate(x_test, y_test, batch_size=128)
# print (score)