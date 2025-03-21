import sys
import traceback

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
from flask import Flask, jsonify, request
import tensorflow as tf
from keras_multi_head import MultiHeadAttention
from flask_cors import CORS



def mymodel_coding(guide_seq):
    code_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    direction_dict = {'A': 2, 'C': 3, 'G': 4, 'T': 5}

    tlen = 23

    gRNA_list = list(guide_seq)
    pair_code = []
    guide_encoded_integers = np.zeros((tlen,), dtype=int)

    for i in range(len(gRNA_list)):
        gRNA_base_code = code_dict[gRNA_list[i].upper()]
        pair_code.append(gRNA_base_code)
        guide_encoded_integers[i] = direction_dict[gRNA_list[i]]

    guide_encoded_integers = np.insert(guide_encoded_integers, 0, 1)
    pair_code_matrix = np.array(pair_code, dtype=np.float32).reshape(1, 1, 23, 4)

    return pair_code_matrix, guide_encoded_integers


app = Flask(__name__)


CORS(app)


@app.route('/aaa', methods=['GET', 'POST'])
def test():
    return "Welcome to machine learning model APIs!"


@app.route('/predictxx', methods=['POST'])
def predictxx():
    data = request.json  
    print("get data")

    database = data["database"]
    sequence = data["sequence"]

    if database == "ESP":
        model = tf.keras.models.load_model(f'./application/CRISPR_HNN_ESP.h5',
                                           custom_objects={'MultiHeadAttention': MultiHeadAttention})
        print('ESP Model loaded')
    elif database == "HF":
        model = tf.keras.models.load_model(f'./application/CRISPR_HNN_HF.h5',
                                           custom_objects={'MultiHeadAttention': MultiHeadAttention})
        print('HF Model loaded')
    else:
        model = tf.keras.models.load_model(f'./application/CRISPR_HNN_WT.h5',
                                           custom_objects={'MultiHeadAttention': MultiHeadAttention})
        print('WT Model loaded')

    codingname = mymodel_coding
    on_hot_encoded, gRNA_encoded = codingname(sequence)  

    on_hot_encoded = np.array(on_hot_encoded, dtype=np.float32).reshape(
        (len(on_hot_encoded), 1, 23, 4)).astype('float32')
    gRNA_encoded = np.array(gRNA_encoded, dtype=np.int32).reshape(1, -1)


    preds = model.predict([on_hot_encoded, gRNA_encoded])
    result = float(preds[0][0])

    return jsonify({'prediction': result})


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])  # This is for a command-line argument
    except:
        port = 12345  # If you don't provide any port then the port will be set to 12345

    # model = tf.keras.models.load_model(f'./application/CRISPR_HNN_WT.h5',
    #                                    custom_objects={'MultiHeadAttention': MultiHeadAttention})
    # print('Model loaded')
    app.run(port=port, debug=True, use_reloader=False)
