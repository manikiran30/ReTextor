from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from google.cloud import vision
from google.oauth2 import service_account

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

credentials = service_account.Credentials.from_service_account_file('my ocr google api.json')
client = vision.ImageAnnotatorClient(credentials=credentials)

# Grammar Correction Model
tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")

def correct_grammar(text):
    input_text = "gec: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def normalize_text(text):
    shorthand_map = {
        "u": "you",
        "r": "are",
        "ur": "your",
        "hv": "have",
        "hw": "how",
        "pls": "please",
        "plz": "please",
        "b4": "before",
        "gr8": "great",
        "m8": "mate",
        "l8r": "later",
        "thx": "thanks",
        "ty": "thank you",
        "idk": "I don't know",
        "imo": "in my opinion",
        "omg": "oh my god",
        "btw": "by the way",
        "w8": "wait"
    }

    words = text.split()
    normalized_words = [shorthand_map.get(word.lower(), word) for word in words]
    return ' '.join(normalized_words)

def extract_text_google_vision(image_path):
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description if texts else ''

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    extracted_text = extract_text_google_vision(filepath)
    normalized_text = normalize_text(extracted_text)
    grammar_corrected = correct_grammar(normalized_text)

    return jsonify({
        'extracted_text': extracted_text,
        'normalized_text': normalized_text,
        'grammar_corrected': grammar_corrected
    })

if __name__ == '__main__':
    app.run(debug=True)
