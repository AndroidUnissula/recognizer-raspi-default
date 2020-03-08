import cv2

face = cv2.CascadeClassifier('face-detect.xml')
cam = cv2.VideoCapture(0)

jumlah = 0
id = input("Masukkan ID :")



while True:
    _, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    muka = face.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in muka:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if cv2.waitKey(1) & 0xff == ord('c'):
            #User. id. serial (jumlah).jpg
            cv2.imwrite('datawajah/User.'+id+'.'+str(jumlah)+'.jpg', gray[y:y+h, x:x+w])
            jumlah +=1
    cv2.imshow('Capturing', frame)

    if jumlah > 30:
        break

cam.release()
cv2.destroyAllWindows()
