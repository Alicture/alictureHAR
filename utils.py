from sklearn.externals import joblib
import json
import numpy as np
from urllib.request import urlopen
from keras.models import load_model
from keras import backend as K
import math


def load_ml_model(model_path):
	clf = joblib.load(model_path)
	return clf

def get_ml_return(data,clf):
	# print(data)
	return clf.predict(data)

def get_sensor_data():
	url = r'http://10.0.0.1:5000/sensordata/api/v1.0/send'
	response = urlopen(url)
	html = json.loads(response.read())
	# data=int(html['data'].split(','))
	# print(html['data'])
	data=html['data']
	data=data.split(',')
	# print(data)
	data = list(map(int, data))
	# print(data)
	return data

def load_dl_model(dl_model_path):
	def f1(y_true, y_pred):
	    def recall(y_true, y_pred):
	        """Recall metric.

	        Only computes a batch-wise average of recall.

	        Computes the recall, a metric for multi-label classification of
	        how many relevant items are selected.
	        """
	        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	        recall = true_positives / (possible_positives + K.epsilon())
	        return recall

	    def precision(y_true, y_pred):
	        """Precision metric.

	        Only computes a batch-wise average of precision.

	        Computes the precision, a metric for multi-label classification of
	        how many selected items are relevant.
	        """
	        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	        precision = true_positives / (predicted_positives + K.epsilon())
	        return precision
	    precision = precision(y_true, y_pred)
	    recall = recall(y_true, y_pred)
	    return 2*((precision*recall)/(precision+recall+K.epsilon()))

	model=load_model(dl_model_path,custom_objects={'f1': f1})
	return model

def dl_data_prepare(data):
	data=data.reshape((1,4,4,1))
	data=data/255
	return data

def get_dl_return(data,model):
	return model.predict(data)

def get_BMI(height,weight):
	return round(weight/math.pow(height,2),1)

