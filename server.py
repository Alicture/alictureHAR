from cloudant import Cloudant
from flask import Flask, render_template, request, jsonify
import atexit
import os
import json

app = Flask(__name__, static_url_path='')

db_name = 'mydb'
client = None
db = None

if 'VCAP_SERVICES' in os.environ:
    vcap = json.loads(os.getenv('VCAP_SERVICES'))
    print('Found VCAP_SERVICES')
    if 'cloudantNoSQLDB' in vcap:
        creds = vcap['cloudantNoSQLDB'][0]['credentials']
        user = creds['username']
        password = creds['password']
        url = 'https://' + creds['host']
        client = Cloudant(user, password, url=url, connect=True)
        db = client.create_database(db_name, throw_on_exists=False)
elif os.path.isfile('vcap-local.json'):
    with open('vcap-local.json') as f:
        vcap = json.load(f)
        print('Found local VCAP_SERVICES')
        creds = vcap['services']['cloudantNoSQLDB'][0]['credentials']
        user = creds['username']
        password = creds['password']
        url = 'https://' + creds['host']
        client = Cloudant(user, password, url=url, connect=True)
        db = client.create_database(db_name, throw_on_exists=False)

# On IBM Cloud Cloud Foundry, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 8000
port = int(os.getenv('PORT', 8000))

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
	app.run(host='0.0.0.0', port=port, debug=True)