# 🧠 OCR Text Reconstruction with NLP

This project extracts text from uploaded or scanned images using **Google Cloud Vision API**, then corrects spelling and grammar using NLP.

## 🚀 Features
- Upload handwritten or printed images
- Extract text using Google Vision API
- Spelling correction using TextBlob
- Grammar correction using T5 Transformer model
- Editable final text

## 🛠 Technologies Used
- Python (Flask)
- Google Cloud Vision API
- TextBlob
- HuggingFace Transformers (T5 model)
- HTML/CSS Frontend

## 📂 Project Structure
- `app.py`: Backend logic
- `templates/index.html`: Frontend UI
- `uploads/`: Stores uploaded images
- `static/style.css`: Styling
- `requirements.txt`: Dependencies

## ✅ Setup
1. Create a Google Cloud Project and enable Vision API
2. Download your `service_account_key.json` and rename it as `your_gcloud_api_key.json`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
