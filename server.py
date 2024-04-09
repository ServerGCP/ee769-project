from processing import check_valid, draw_bbox, img_display
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import PIL, io, os, cv2, sys
import requests, threading
import streamlit as st
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

PROCESS_FOLDER = os.path.join(UPLOAD_FOLDER, 'processed')
app.config['PROCESS_FOLDER'] = UPLOAD_FOLDER
os.makedirs(PROCESS_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    return jsonify({'success': 'Flask running'}), 400
    
@app.route('/process', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Not a Image'}), 400
    
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty Image'}), 400
    
        if file:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
    
            image = check_valid(save_path)
            if image is not False:
                image = draw_bbox(image)
                out_path = os.path.join(app.config['PROCESS_FOLDER'], filename)
                cv2.imwrite(out_path, image)
                return send_file(out_path, mimetype='image/jpeg')
            else:
                return jsonify({'error': 'Incompatible Image'}), 400
                
    except:
        return jsonify({'error': 'Unknown'}), 400

def run_flask():
    app.run(port=5000)

flask_thread = threading.Thread(target=run_flask)
flask_thread.start()

st.title('Image Upload for Processing')

image = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
if image is not None:
    st.image(image, caption='Uploaded Image', use_column_width=True)

    buf = io.BytesIO()
    buf.write(image.getvalue())
    buf.seek(0)

    url = 'http://localhost:5000/process'
    files = {'image': (image.name, buf, 'multipart/form-data')}
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        st.success('Image successfully processed!')
        processed_image = PIL.Image.open(io.BytesIO(response.content))
        st.image(processed_image, caption='Processed Image', use_column_width=True)
    else:
        error_message = f'Failed Response: {response.text}'
        st.error(error_message)
