import matplotlib.pyplot as plt
import cv2


def show_results(clean, noisy, denoised):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title('Clean Image')
    plt.imshow(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 2)
    plt.title('Noisy Image')
    plt.imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 3, 3)
    plt.title('Denoised Image')
    plt.imshow(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))

    plt.show()