####################################################
# Modified by Muhammad Ni'am                       #
# Original code: http://thecodacus.com/            #
####################################################

import os

# Import OpenCV2 for image processing
import cv2


# Import numpy for matrices calculations

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Membuat Local Binary Patterns Histograms untuk face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

assure_path_exists("trainer/")

# muat mode yang data yang sudah di simpan pada recognizer
recognizer.read('trainer/trainer.yml')

cascadePath = "face-detect.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# mengatur font style text
font = cv2.FONT_HERSHEY_SIMPLEX

# inisiasi camera
cam = cv2.VideoCapture(0)

# Loop
while True:
    # membaca camera
    ret, im =cam.read()

    # merubah gambar menjadi grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # mendeteksi wajah pada gambar
    wajah = faceCascade.detectMultiScale(gray, 1.2, 5)

    # perulangan untun mencocokan wajah dengan data training
    for(x,y,w,h) in wajah:

        # menandai pada bagian wajah dengan kotak / persegi panjang
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        # mengenali id wajah
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # mengecek apakan id wajah sudah ada
        if Id is 1:
            Id = "Tamyiz {0:.2f}%".format(round(100 - confidence, 2))
        if Id is 2:
            Id = "Ni'am {0:.2f}%".format(round(100 - confidence, 2))
        if Id is 3:
            Id = "Riyan {0:.2f}%".format(round(100 - confidence, 2))
        # if Id is 4:
        #     Id = "Kusuma {0:.2f}%".format(round(100 - confidence, 2))
        # if Id is 5:
        #     Id = "Intan {0:.2f}%".format(round(100 - confidence, 2))
        # if Id is 6:
        #     Id = "Sukma {0:.2f}%".format(round(100 - confidence, 2))
        # if Id is 7:
        #     Id = "Afif {0:.2f}%".format(round(100 - confidence, 2))

        #print(Id)




        # menampilkan teks pada wajah yang terdeteksi
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(Id), (x,y-40), font, 1, (255,255,255), 3)

    # menampilkan video frame untuk user
    cv2.imshow('Pengujian Pengenalan Wajah',im)

    # tekan 'q' untuk menghentikan program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# keluar dari camera
cam.release()

# keluar dari semua jendela
cv2.destroyAllWindows()
