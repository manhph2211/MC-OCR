import os
from flask import Flask, request, jsonify, render_template
from main import process_image
basedir = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/upload',methods=['GET','POST'])
def upload_file():
    pass

@app.route('/predict', methods=['POST'])
def predict():
    pass

if __name__ == '__main__':
    app.run(host="0.0.0.0", port = 8080, debug=True)