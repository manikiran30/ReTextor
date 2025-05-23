from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import io
from google.cloud import vision
from google.oauth2 import service_account
import language_tool_python

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


credentials = service_account.Credentials.from_service_account_file('my ocr google api.json')
client = vision.ImageAnnotatorClient(credentials=credentials)

# 🔠 Grammar correction tool
grammar_tool = language_tool_python.LanguageTool('en-US')

# 🔁 Common abbreviation expansion
abbreviation_map = {
    "u": "you", "r": "are", "ur": "your", "pls": "please", "plz": "please",
    "btw": "by the way", "idk": "I don't know", "lol": "laughing out loud",
    "hw": "how", "gr8": "great", "b4": "before", "thx": "thanks", "w8": "wait",
    "msg": "message", "im": "I am", "bcz": "because", "tmrw": "tomorrow"
}

def expand_abbreviations(text):
    words = text.split()
    expanded = [abbreviation_map.get(w.lower(), w) for w in words]
    return ' '.join(expanded)

def correct_grammar(text):
    matches = grammar_tool.check(text)
    corrected = language_tool_python.utils.correct(text, matches)
    return corrected

def extract_text_google_vision(image_path):
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    if response.error.message:
        raise Exception(f"Vision API error: {response.error.message}")
    return response.full_text_annotation.text.strip() if response.full_text_annotation.text else ""

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    extracted_text = extract_text_google_vision(filepath)
    expanded_text = expand_abbreviations(extracted_text)
    grammar_corrected = correct_grammar(expanded_text)

    return jsonify({
        'extracted_text': extracted_text,
        'expanded_text': expanded_text,
        'grammar_corrected': grammar_corrected
    })

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
