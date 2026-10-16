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

gray = (0.114 * B + 0.587 * G + 0.299 * R).astype(np.uint8)

negatif = 255 - gray
c = 255 / np.log(256)
logaritmik = (c * np.log(1 + gray.astype(float))).astype(np.uint8)

def gamma_transform(g, gamma):
    return (255 * (g / 255.0) ** gamma).astype(np.uint8)

gamma_terang = gamma_transform(gray, 0.5)   
gamma_gelap = gamma_transform(gray, 2.0)   

r_min = float(gray.min())
r_max = float(gray.max())
stretching = ((gray - r_min) / (r_max - r_min) * 255).astype(np.uint8)

def hitung_histogram(g):
    hist = [0] * 256
    for baris in g:
        for piksel in baris:
            hist[piksel] += 1
    return hist

def ekualisasi(g):
    hist = hitung_histogram(g)
    total = g.shape[0] * g.shape[1]

    pdf = [h / total for h in hist]
    cdf = []
    kumulatif = 0
    for p in pdf:
        kumulatif += p
        cdf.append(kumulatif)

    tabel = [round(255 * c) for c in cdf]
    hasil = np.zeros_like(g)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            hasil[i, j] = tabel[g[i, j]]

    return hasil

hasil_eq = ekualisasi(gray)

gambar = [gray, negatif, logaritmik, gamma_terang, gamma_gelap, stretching]
judul = ["Grayscale", "Negatif", "Logaritmik",
         "Gamma 0.5", "Gamma 2.0", "Contrast Stretching"]

plt.figure(figsize=(12, 7))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(gambar[i], cmap="gray", vmin=0, vmax=255)
    plt.title(judul[i])
    plt.axis("off")
plt.tight_layout()
plt.show()

hist_asli = hitung_histogram(gray)
hist_eq = hitung_histogram(hasil_eq)

plt.figure(figsize=(10, 7))

plt.subplot(2, 2, 1)
plt.imshow(gray, cmap="gray", vmin=0, vmax=255)
plt.title("Sebelum Ekualisasi")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(hasil_eq, cmap="gray", vmin=0, vmax=255)
plt.title("Sesudah Ekualisasi")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.bar(range(256), hist_asli)
plt.title("Histogram Sebelum")

plt.subplot(2, 2, 4)
plt.bar(range(256), hist_eq)
plt.title("Histogram Sesudah")

plt.tight_layout()
plt.show()