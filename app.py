# flask_web/app.py

from flask import Flask
import numpy as np

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'TEST TEST TEST'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
