from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
CORS(app)
data_storage = []
led_state = 0
model = load_model('soncheckpoint1.h5')

# Sınıf indekslerini yükleme (Bu, eğitilmiş modelinizin sınıf indeksleri ile eşleşmelidir)
class_indices = {
    'A10': 0, 'A400M': 1, 'AG600': 2, 'AV8B': 3, 'B1': 4, 'B2': 5, 'B52': 6, 
    'Be200': 7, 'C130': 8, 'C17': 9, 'C2': 10, 'C5': 11, 'E2': 12, 'E7': 13, 
    'EF2000': 14, 'F117': 15, 'F14': 16, 'F15': 17, 'F16': 18, 'F18': 19, 'F22': 20, 
    'F35': 21, 'F4': 22, 'H6': 23, 'J10': 24, 'J20': 25, 'JAS39': 26, 'JF17': 27, 
    'KC135': 28, 'MQ9': 29, 'Mig31': 30, 'Mirage2000': 31, 'P3': 32, 'RQ4': 33, 
    'Rafale': 34, 'SR71': 35, 'Su24': 36, 'Su25': 37, 'Su34': 38, 'Su57': 39, 
    'Tornado': 40, 'Tu160': 41, 'Tu22M': 42, 'Tu95': 43, 'U2': 44, 'US2': 45, 
    'V22': 46, 'Vulcan': 47, 'XB70': 48, 'YF23': 49
}

def prepare_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'Dosya Yok'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    # Dosyayı kaydetme işlemi
    save_path = os.path.join('uploads', file.filename)
    file.save(save_path)
    
    image = prepare_image(save_path)
    prediction = model.predict(image)
    predicted_class_id = int(np.argmax(prediction, axis=1)[0])
    
    # Sınıf adını al
    predicted_class = list(class_indices.keys())[predicted_class_id]
    print(predicted_class)
    # Tahmin sonucunu döndürme
    return jsonify({'message': 'Dosya Yüklendi', 'predicted_class': predicted_class}), 200
@app.route('/api/data', methods=['POST'])
def post_data():
    global led_state
    data = request.get_json()
    data_storage.append(data)
    print(data)  # Gelen veriyi konsolda görüntüle
    return jsonify({'message': 'Data alindi', 'data': data, 'led_state':led_state}), 201

@app.route('/api/data/led', methods=['POST'])
def post_data_led():
    global led_state
    data = request.get_json()
    data_storage.append(data)
    led_state = data.get('led_state')
    print(led_state)  # Gelen veriyi konsolda görüntüle
    return jsonify({'message': 'Data alindi', 'data': data}), 209


@app.route('/api/data', methods=['GET'])
def get_data():
    if data_storage:
        return jsonify(data_storage[-1])  # Son alınan veriyi döndür
    else:
        return jsonify({'message': 'veri bulunamadı'})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
