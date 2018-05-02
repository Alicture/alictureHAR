from flask import Flask, jsonify, render_template, request, Blueprint
import random
from utils import *
# import recite
app = Flask(__name__)
# ck = Blueprint('ck_page', __name__, static_folder=chartkick.js(), static_url_path='/static')
# app.register_blueprint(ck, url_prefix='/ck')
# app.jinja_env.add_extension("chartkick.ext.charts")

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
	return jsonify(data=[random.randint(-100,100),random.randint(-100,100),random.randint(-100,100),random.randint(-100,100),random.randint(-100,100),random.randint(-100,100),random.randint(-100,100),random.randint(-100,100),random.randint(-100,100),random.randint(-100,100),random.randint(-100,100),random.randint(-100,100)])

if __name__ == "__main__":
	app.run()