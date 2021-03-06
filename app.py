from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score , log_loss
import opendatasets as od
from flask import Flask
from flask_caching import Cache
from apscheduler.schedulers.background import BackgroundScheduler
from werkzeug.utils import secure_filename
import tensorflow as tf
from zipfile import ZipFile
import time
from keras.applications.inception_resnet_v2 import InceptionResNetV2


app = Flask(__name__)
app.config['CACHE_TYPE'] = 'SimpleCache' 
cache = Cache(app)


app.config['UPLOAD_FOLDER']='./img'

@app.before_first_request
def do_something_only_once():
    if not os.path.exists('cs2-m/weights-02-0.9045.hdf5'):
        od.download('https://www.kaggle.com/datasets/subhendumalakar/cs2-m',force=True)
        time.sleep(1000)
        zf = ZipFile('cs2-m.zip', 'r')
        zf.extractall('/')
        zf.close()

@app.route("/",methods=['GET'])
def hello():
    return render_template('index.html') 

@app.route("/",methods=['POST'])
def get_pre():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    InceptionResNetV2=InceptionResNetV2(include_top=False, weights='imagenet',input_shape=(400, 400, 3), classes=120)
    for i in InceptionResNetV2.layers:
        i.trainable = False
    x = InceptionResNetV2.output

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=120,activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    out = tf.keras.layers.Dense(120, activation='softmax')(x)
    InceptionResNetV2_model = tf.keras.models.Model(inputs=InceptionResNetV2.input, outputs=out)
    InceptionResNetV2_model.load_weights('cs2-m/weights-02-0.9045.hdf5')


    path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pre1=tf.keras.preprocessing.image.load_img(path,target_size=(400, 400)) # For class afghan_hound
    pre2 = tf.keras.preprocessing.image.img_to_array(pre1)
    pre2 = np.expand_dims(pre2, axis = 0)
    pred=InceptionResNetV2_model.predict(pre2)

    ind=np.where(pred[0]==max(pred[0]))[0][0]
    keys=['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 'american_staffordshire_terrier', 'appenzeller', 'australian_terrier', 'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog', 'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick', 'border_collie', 'border_terrier', 'borzoi', 'boston_bull', 'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo', 'doberman', 'english_foxhound', 'english_setter', 'english_springer', 'entlebucher', 'eskimo_dog', 'flat-coated_retriever', 'french_bulldog', 'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer', 'golden_retriever', 'gordon_setter', 'great_dane', 'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael', 'ibizan_hound', 'irish_setter', 'irish_terrier', 'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier', 'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog', 'mexican_hairless', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'newfoundland', 'norfolk_terrier', 'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog', 'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian', 'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki', 'samoyed', 'schipperke', 'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier', 'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier', 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 'standard_poodle', 'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier', 'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel', 'west_highland_white_terrier', 'whippet', 'wire-haired_fox_terrier', 'yorkshire_terrier']
    values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
    print("Predicted Breed: ",keys[values.index(ind)])
    os.remove(path)
    return render_template('index.html',result=keys[values.index(ind)]) 

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8080,debug=True)
    # app.run(debug=True)