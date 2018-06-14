#
from cloudant import Cloudant
from flask import Flask, render_template, request, jsonify,send_file
import atexit
import os
import json
import random
import utils as util
from keras.models import load_model
import keras.objectives
import keras.backend as K
import numpy as np
import tensorflow as tf
from flask_apidoc import ApiDoc
from flask import make_response

app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False

doc = ApiDoc(app=app)


ml_model=util.load_ml_model('train_model.m')
dlmodel=util.load_dl_model('my_model-v2.h5')
graph = tf.get_default_graph()
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

# keras.objectives.custom_loss = f1        
dl_model=load_model('my_model-v2.h5',custom_objects={'f1': f1})


@app.route('/')

@app.route('/index')
def index():
	return render_template('index.html')


@app.route('/datacenter')
def datacenter():
	return render_template('datacenter.html')

@app.route('/cnn')
def cnn():
	return render_template('cnn.html')

@app.route('/ml')
def ml():
	return render_template('ml.html')

@app.route('/demo')
def demo():
	return render_template('demo.html')


@app.route('/randomtest')
def randomtest():
	return jsonify(data=util.get_sensor_data())





@app.route('/showdemo')
def showdemo():
    age=request.args.get('age', 0, type=int)
    weight = request.args.get('weight', 0, type=int)
    height = request.args.get('height', 0, type=float)
    bmi=util.get_BMI(height,weight)
    data=[age,height,weight,bmi]
    data=np.array(data)

    sensordata=util.get_sensor_data()
    sensordata=np.array(sensordata)
    # data=data.append(sensordata[:])
    # print(data)
    # print(sensordata)
    data=np.concatenate((data,sensordata))
    data=data.reshape((1,16))
    # print(data)
    ml_result=util.get_ml_return(data,ml_model)
    # print(ml_result)
    dldata=util.dl_data_prepare(data)
    # print(dldata)
    # dl_result=util.get_dl_return(dldata,dl_model)
    global graph
    with graph.as_default():
        label=dlmodel.predict(dldata)
        # print(label)
        label=list(map(lambda x: x==max(x), label)) * np.ones(shape=label.shape)
        # print(label)

    mlresult=ml_result.tolist()
    dlresult=label.tolist()
    return jsonify(mlresult=mlresult,dlresult=dlresult)

    
@app.route('/api/v1.0/ML/KNN', methods=['POST'])
def KNN_api():
    """
    @api {post} /api/v1.0/ML/KNN KNN方式检测
    @apiVersion 1.0.0
    @apiName KNN
    @apiGroup HAR
    @apiParam {int}  weight      (必须)    用户体重
    @apiParam {float}  height    (必须)    用户身高
    @apiParam {int}  age    (必须)    用户年龄
    @apiParam {int}  sensordata    (必须)    用户传感器数据
    @apiParamExample {json} Request-Example:
        {
            "weight": 82,
            "height": 1.82,
            "age": 22,
            "sensordata": [-30,16,-15,20,-90,-32,-68,150,12,-24,-12,-42]
        }

    @apiSuccess (回参) {String} result  用户行为检测结果
    @apiSuccessExample {json} Success-Response:
        {
            "errno":0,
            "errmsg":"检测成功！",
            "result": "行走"
        }

    @apiErrorExample {json} Error-Response:
        {
            "errno":400,
            "errmsg":"数据错误！"
        }

    """
    if not request.json or not 'weight' in request.json or not 'height' in request.json or not 'sensordata' in request.json or not 'age' in request.json:
        abort(400)
    age=request.json['age']
    weight=request.json['weight']
    height=request.json['height']
    sensordata=request.json['sensordata']
    bmi=util.get_BMI(height,weight)
    data=[age,height,weight,bmi]
    data=np.array(data)
    sensordata=np.array(sensordata)
    data=np.concatenate((data,sensordata))
    data=data.reshape((1,16))
    ml_result=util.get_ml_return(data,ml_model)
    mlresult=ml_result.tolist()
    if mlresult[0][0]==1:
        result='坐着'
    if mlresult[0][1]==1:
        result='下坐'
    if mlresult[0][2]==1:
        result='站立'
    if mlresult[0][3]==1:
        result='站起'
    if mlresult[0][4]==1:
        result='行走'
    task = {
        'errno': '0',
        'errmsg': '检测成功！',
        'result': result
    }
    return jsonify(task), 201

@app.route('/api/v1.0/DL/ResNet-50mini', methods=['POST'])
def DL_api():
    """
    @api {post} /api/v1.0/DL/ResNet-50mini 卷积神经网络检测
    @apiVersion 1.0.0
    @apiName DL_Res50mini
    @apiGroup HAR
    @apiParam {int}  weight      (必须)    用户体重
    @apiParam {float}  height    (必须)    用户身高
    @apiParam {int}  age    (必须)    用户年龄
    @apiParam {int}  sensordata    (必须)    用户传感器数据
    @apiParamExample {json} Request-Example:
        {
            "weight": 82,
            "height": 1.82,
            "age": 22,
            "sensordata": [-30,16,-15,20,-90,-32,-68,150,12,-24,-12,-42]
        }

    @apiSuccess (回参) {String} result  用户行为检测结果
    @apiSuccessExample {json} Success-Response:
        {
            "errno":0,
            "errmsg":"检测成功！",
            "result": "行走"
        }

    @apiErrorExample {json} Error-Response:
        {
            "errno":400,
            "errmsg":"数据错误！"
        }

    """
    if not request.json or not 'weight' in request.json or not 'height' in request.json or not 'sensordata' in request.json or not 'age' in request.json:
        abort(400)
    age=request.json['age']
    weight=request.json['weight']
    height=request.json['height']
    sensordata=request.json['sensordata']
    bmi=util.get_BMI(height,weight)
    data=[age,height,weight,bmi]
    data=np.array(data)
    sensordata=np.array(sensordata)
    data=np.concatenate((data,sensordata))
    data=data.reshape((1,16))
    dldata=util.dl_data_prepare(data)
    global graph
    with graph.as_default():
        label=dlmodel.predict(dldata)
        label=list(map(lambda x: x==max(x), label)) * np.ones(shape=label.shape)
    dlresult=label.tolist()
    if dlresult[0][1]==1:
        result='坐着'
    if dlresult[0][2]==1:
        result='下坐'
    if dlresult[0][3]==1:
        result='站立'
    if dlresult[0][4]==1:
        result='站起'
    if dlresult[0][5]==1:
        result='行走'
    task = {
        'errno': '0',
        'errmsg': '检测成功！',
        'result': result
    }
    return jsonify(task), 201


@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify({'errno': '400','errmsg':'数据错误！'}), 400)

