import os
from flask import Flask, request, Response, jsonify, render_template, send_from_directory
from main import process_image
from werkzeug.utils import secure_filename


basedir = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = 'static/img'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def getExtention(image_file):
    filename, file_extension = os.path.splitext(image_file)
    return filename, file_extension


@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            print("file was uploaded in {} ".format(path))
            output = process_image(path)
            output = list(output[path].keys())[0]
            return output

    elif request.method == 'GET':
        return Response(status=200) 
    else:
        return Response(status=200)


@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)


@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8085, debug=True)