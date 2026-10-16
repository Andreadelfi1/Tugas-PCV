import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("sazabi_kit.jpeg")

if img is None:
    print("Foto tidak ditemukan")
    exit()

img = img[::2, ::2]
rgb = img[:, :, ::-1].astype(float) / 255.0

R = rgb[:, :, 0]
G = rgb[:, :, 1]
B = rgb[:, :, 2]


def rgb_ke_cmyk(R, G, B):
    K = 1 - np.maximum(np.maximum(R, G), B)
    penyebut = 1 - K
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(penyebut == 0, 0, (1 - R - K) / penyebut)
        M = np.where(penyebut == 0, 0, (1 - G - K) / penyebut)
        Y = np.where(penyebut == 0, 0, (1 - B - K) / penyebut)
    return C, M, Y, K


def rgb_ke_hsv(R, G, B):
    maks = np.maximum(np.maximum(R, G), B)
    mins = np.minimum(np.minimum(R, G), B)
    delta = maks - mins

    V = maks

    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(maks == 0, 0, delta / maks)

        H = np.zeros_like(maks)
        H = np.where(maks == R, 60 * (((G - B) / delta) % 6), H)
        H = np.where(maks == G, 60 * ((B - R) / delta + 2), H)
        H = np.where(maks == B, 60 * ((R - G) / delta + 4), H)
        H = np.where(delta == 0, 0, H)

    return H, S, V


def rgb_ke_hsi(R, G, B):
    I = (R + G + B) / 3

    mins = np.minimum(np.minimum(R, G), B)
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(I == 0, 0, 1 - mins / I)
    atas = 0.5 * ((R - G) + (R - B))
    bawah = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_theta = np.where(bawah == 0, 1, atas / bawah)
    theta = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
    H = np.where(B <= G, theta, 360 - theta)
    return H, S, I


C, M, Y, K = rgb_ke_cmyk(R, G, B)
H_hsv, S_hsv, V_hsv = rgb_ke_hsv(R, G, B)
H_hsi, S_hsi, I_hsi = rgb_ke_hsi(R, G, B)

plt.figure(figsize=(14, 7))

tampil = [rgb, R, G, B]
judul = ["RGB (Asli)", "Channel R", "Channel G", "Channel B"]
for i in range(4):
    plt.subplot(2, 4, i + 1)
    if i == 0:
        plt.imshow(tampil[i])
    else:
        plt.imshow(tampil[i], cmap="gray", vmin=0, vmax=1)
    plt.title(judul[i])
    plt.axis("off")

tampil = [rgb, C, M, Y, K]
judul = ["Asli", "Cyan", "Magenta", "Yellow", "Key (Black)"]
for i in range(1, 5):
    plt.subplot(2, 4, 4 + i)
    plt.imshow(tampil[i], cmap="gray", vmin=0, vmax=1)
    plt.title(judul[i])
    plt.axis("off")

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 7))

tampil = [H_hsv / 360, S_hsv, V_hsv, H_hsi / 360, S_hsi, I_hsi]
judul = ["HSV: Hue", "HSV: Saturation", "HSV: Value",
         "HSI: Hue", "HSI: Saturation", "HSI: Intensity"]
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(tampil[i], cmap="gray", vmin=0, vmax=1)
    plt.title(judul[i])
    plt.axis("off")
plt.tight_layout()
plt.show()
