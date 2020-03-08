####################################################
# Modified by Muhammad Ni'am                       #
# Original code: http://thecodacus.com/            #
####################################################

# Import OpenCV2 for image processing | import opencv untuk memproses gambar
# Import os for file path | Import os untuk jalur file
import cv2

# Import numpy for matrix calculation | Impor numpy untuk perhitungan matriks
import numpy as np

# Import Python Image Library (PIL) | Import pustaka gambar python
from PIL import Image

import os


# Membuat fungsi / method untuk membuat directory baru
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


# Create Local Binary Patterns Histograms for face recognization
# Buat Histogram pola biner lokal untuk pengenalan wajah
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Using prebuilt frontal face training model, for face detection
# Menggunakan model pelatihan wajah frontal prebuilt, untuk deteksi wajah
detector = cv2.CascadeClassifier("face-detect.xml");


# Create method to get the images and label data | buat metede untuk mendapatkan gambar dan label data
def getImagesAndLabels():
    path = ("/home/pi/pengenal/dataset")
    # Get all file path | dapatkan semua jalur file
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    # Initialize empty face sample | inisialisasi sampel wajah kosong
    faceSamples = []

    # Initialize empty id | inisialisasi id kosong
    ids = []

    # Loop all the file path | perulangan semua jalur file
    for imagePath in imagePaths:

        # Get the image and convert it to grayscale | dapatkan gambar dan ubah menjadi skala abu-abu
        PIL_img = Image.open(imagePath).convert('L')

        # PIL image to numpy array | gambar PIL ke array numpy
        img_numpy = np.array(PIL_img, 'uint8')

        # Get the image id | dapatkan ID gambar
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        # Get the face from the training images | dapatkan wajad dari gambar training
        faces = detector.detectMultiScale(img_numpy)

        # Loop for each face, append to their respective ID | loop setiap wajah, tambahkan id masing-masing
        for (x, y, w, h) in faces:
            # Add the image to face samples | tambahkan gambar ke sampel wajah
            faceSamples.append(img_numpy[y:y + h, x:x + w])

            # Add the ID to IDs | tambahkan ID ke ID
            ids.append(id)

    # Pass the face array and IDs array | lulus array wajah dan array ID
    return faceSamples, ids


#dapatkan wajah dan ID
# faces, ids = getImagesAndLabels('dataset')
faces, ids = getImagesAndLabels()

#latih model menggunakan wajah dan ID
recognizer.train(faces, np.array(ids))

#simpan medel ke dalam trainer.yml
assure_path_exists('trainer/')
recognizer.save('trainer/trainer.yml')
