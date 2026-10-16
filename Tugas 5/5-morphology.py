import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("sazabi_kit.jpeg")

if img is None:
    print("Foto tidak ditemukan! Cek nama file dan lokasinya.")
    exit()

img = img[::2, ::2]

B = img[:, :, 0].astype(float)
G = img[:, :, 1].astype(float)
R = img[:, :, 2].astype(float)
gray = 0.114 * B + 0.587 * G + 0.299 * R

ambang = 150
biner = (gray > ambang).astype(np.uint8)

def erosi(b, se):
    k = se.shape[0]
    p = k // 2
    h, w = b.shape
    pad = np.zeros((h + 2 * p, w + 2 * p), dtype=np.uint8)
    pad[p:p + h, p:p + w] = b
    hasil = np.ones((h, w), dtype=np.uint8)
    for i in range(k):
        for j in range(k):
            if se[i, j] == 1:
                hasil = hasil & pad[i:i + h, j:j + w]
    return hasil

def dilasi(b, se):
    k = se.shape[0]
    p = k // 2
    h, w = b.shape
    pad = np.zeros((h + 2 * p, w + 2 * p), dtype=np.uint8)
    pad[p:p + h, p:p + w] = b
    hasil = np.zeros((h, w), dtype=np.uint8)
    for i in range(k):
        for j in range(k):
            if se[i, j] == 1:
                hasil = hasil | pad[i:i + h, j:j + w]
    return hasil

def opening(b, se):
    return dilasi(erosi(b, se), se)

def closing(b, se):
    return erosi(dilasi(b, se), se)


se = np.ones((5, 5), dtype=np.uint8)

hasil_erosi = erosi(biner, se)
hasil_dilasi = dilasi(biner, se)
hasil_opening = opening(biner, se)
hasil_closing = closing(biner, se)

np.random.seed(0)
noisy = biner.copy()
acak = np.random.rand(*biner.shape)
noisy[acak < 0.02] = 0
noisy[acak > 0.98] = 1

noisy_opening = opening(noisy, se)
noisy_closing = closing(noisy, se)
noisy_open_close = closing(opening(noisy, se), se)

gambar = [gray, biner, hasil_erosi, hasil_dilasi, hasil_opening, hasil_closing]
judul = ["Grayscale", "Biner", "Erosi 5x5",
         "Dilasi 5x5", "Opening 5x5", "Closing 5x5"]

plt.figure(figsize=(14, 8))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(gambar[i], cmap="gray", vmin=0, vmax=(255 if i == 0 else 1))
    plt.title(judul[i])
    plt.axis("off")
plt.tight_layout()
plt.show()

gambar = [biner, noisy, noisy_opening, noisy_closing, noisy_open_close]
judul = ["Biner Asli", "Biner + Noise", "Opening",
         "Closing", "Opening lalu Closing"]

plt.figure(figsize=(14, 8))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(gambar[i], cmap="gray", vmin=0, vmax=1)
    plt.title(judul[i])
    plt.axis("off")
plt.tight_layout()
plt.show()