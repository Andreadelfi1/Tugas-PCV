import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("sazabi_kit.jpeg")

if img is None:
    print("Foto tidak ditemukan")
    exit()

img = img[::4, ::4]
rgb = img[:, :, ::-1].copy()

def konvolusi(g, kernel):
    k = kernel.shape[0]
    p = k // 2
    h, w = g.shape
    pad = np.zeros((h + 2 * p, w + 2 * p))
    pad[p:p + h, p:p + w] = g
    kernel = kernel[::-1, ::-1]
    hasil = np.zeros((h, w))
    for i in range(k):
        for j in range(k):
            hasil += kernel[i, j] * pad[i:i + h, j:j + w]
    return hasil
def ke_uint8(x):
    return np.clip(x, 0, 255).astype(np.uint8)

def kernel_gaussian(ukuran, sigma):
    p = ukuran // 2
    kernel = np.zeros((ukuran, ukuran))
    for y in range(-p, p + 1):
        for x in range(-p, p + 1):
            kernel[y + p, x + p] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()

def filter_median(g, k=3):
    p = k // 2
    h, w = g.shape
    pad = np.zeros((h + 2 * p, w + 2 * p))
    pad[p:p + h, p:p + w] = g
    jendela = []
    for i in range(k):
        for j in range(k):
            jendela.append(pad[i:i + h, j:j + w])
    jendela = np.sort(np.stack(jendela), axis=0)
    return jendela[len(jendela) // 2]

def per_channel(fungsi, citra):
    return np.dstack([fungsi(citra[:, :, c].astype(float)) for c in range(3)])

kernel_mean = np.ones((5, 5)) / 25
kernel_gauss = kernel_gaussian(5, 1.0)
kernel_sharpen = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

kernel_laplacian = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])

kernel_sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

kernel_sobel_y = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])

np.random.seed(0)
noisy = rgb.copy()
acak = np.random.rand(rgb.shape[0], rgb.shape[1])
noisy[acak < 0.025] = 0
noisy[acak > 0.975] = 255

hasil_mean = per_channel(lambda c: ke_uint8(konvolusi(c, kernel_mean)), noisy)
hasil_gauss = per_channel(lambda c: ke_uint8(konvolusi(c, kernel_gauss)), noisy)
hasil_median = per_channel(lambda c: ke_uint8(filter_median(c, 3)), noisy)

hasil_sharpen = per_channel(lambda c: ke_uint8(konvolusi(c, kernel_sharpen)), rgb)
hasil_laplacian = per_channel(lambda c: ke_uint8(np.abs(konvolusi(c, kernel_laplacian))), rgb)
hasil_sobel_x = per_channel(lambda c: ke_uint8(np.abs(konvolusi(c, kernel_sobel_x))), rgb)
hasil_sobel_y = per_channel(lambda c: ke_uint8(np.abs(konvolusi(c, kernel_sobel_y))), rgb)
hasil_sobel = per_channel(
    lambda c: ke_uint8(np.sqrt(konvolusi(c, kernel_sobel_x) ** 2 +
                               konvolusi(c, kernel_sobel_y) ** 2)), rgb)

gambar = [rgb, noisy, hasil_mean, hasil_gauss, hasil_median]
judul = ["Asli", "Noise Salt & Pepper", "Mean 5x5", "Gaussian 5x5", "Median 3x3"]

plt.figure(figsize=(14, 7))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(gambar[i])
    plt.title(judul[i])
    plt.axis("off")
plt.tight_layout()
plt.show()

gambar = [rgb, hasil_sharpen, hasil_laplacian,
          hasil_sobel_x, hasil_sobel_y, hasil_sobel]
judul = ["Asli", "Sharpening", "Laplacian",
         "Sobel X", "Sobel Y", "Sobel (Magnitudo)"]

plt.figure(figsize=(14, 7))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(gambar[i])
    plt.title(judul[i])
    plt.axis("off")
plt.tight_layout()
plt.show()