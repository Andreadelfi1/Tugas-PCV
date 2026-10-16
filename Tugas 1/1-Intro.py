import cv2
import numpy as np

def ubah_warna(gambar, hue):
    hsv = cv2.cvtColor(gambar, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    h[:] = hue      
    s[:] = 255      

    hsv_baru = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_baru, cv2.COLOR_HSV2BGR)


def tiga_warna(gambar):
    merah = ubah_warna(gambar, 0)
    kuning = ubah_warna(gambar, 30)
    hijau = ubah_warna(gambar, 60)
    return np.hstack([merah, kuning, hijau])   

img = cv2.imread("sazabi_kit.jpeg")

if img is None:
    print("Foto tidak ditemukan!")
    exit()

img = cv2.resize(img, (400, 533))
cv2.imshow("Sazabi: Merah | Kuning | Hijau", tiga_warna(img))
print("Tekan tombol apa saja untuk lanjut ke video...")
cv2.waitKey(0)
cv2.destroyAllWindows()

# FILTER COLOR VIDEO
cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (400, 300))
    cv2.imshow("Video: Merah | Kuning | Hijau", tiga_warna(frame))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()