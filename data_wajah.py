####################################################
# Modified by Muhammad Ni'am                       #
# Original code: http://thecodacus.com/            #
####################################################

# Import OpenCV2 for image processing
import cv2
import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Start capturing video
vid_cam = cv2.VideoCapture(0)

# Pendeteksi object video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier('face-detect.xml')

# memberikan id untuk masing2 wajah
face_id = input("masukkan Id baru : ")

# inisialisasi jumlah gambar
jumlah = 0

assure_path_exists("dataset/")

# Membuat Perulangan untuk mengambil 100 gambar wajah
while(True):

    # Mengambil video frame
    _, image_frame = vid_cam.read()

    # Merubah video frame menjadi gambar grayscale
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Mendeteksi wajah pada videocam
    wajah = face_detector.detectMultiScale(gray, 1.3, 5)

    # mengambil gambar yang sudah di crop dan menyimpannya pada forder dataset
    for (x,y,w,h) in wajah:

        # Meng-crop hanya pada bagian wajah dengan sebuah kotak biru dengan ketebalan 2
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)

        # nemabahkan nilai count (jumlah gambar pada id tertentu)
        jumlah += 1

        # Menyimpan gambar yang telah di tangkap pada folder dataset dengan nama sesuai ID dan count
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(jumlah) + ".jpg", gray[y:y + h, x:x + w])

        # Menampilkan video camera untuk untuk user yang akan di ambil gambar wajahnya
        cv2.imshow('Pengambilan Data Wajah', image_frame)

    # tekan q untuk menghentikan video frame
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # jika gambar yang diambi sudah mencapai 100 maka video frame akan berhenti secara otomatis
    elif jumlah>100:
        break

# menghentikan kamera
vid_cam.release()

# keluar dari semua jendela program
cv2.destroyAllWindows()
