import os
from google.colab import files
from shutil import copyfile

def upload_and_save_image(upload_folder="uploads"):
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    uploaded = files.upload()
    uploaded_filename = list(uploaded.keys())[0]
    
    dest_path = os.path.join(upload_folder, uploaded_filename)
    copyfile(uploaded_filename, dest_path)
    
    print(f"\n✅ File saved to: {dest_path}")
    return dest_path
