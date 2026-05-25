import cv2
import os
import numpy as np
from tqdm import tqdm

def add_poisson_noise(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)


# JPEG Compression Noise

def add_jpeg_noise(image, quality=10):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg


# Low-light Noise

def add_lowlight_noise(image):
    gamma = 2.5
    dark = np.array(255 * (image / 255) ** gamma, dtype='uint8')

    noise = np.random.normal(0, 15, image.shape)
    noisy = dark + noise

    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)


# Dataset 생성

def generate_noisy_dataset(input_dir, output_dir, noise_type):
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir)):
        path = os.path.join(input_dir, filename)
        image = cv2.imread(path)

        if image is None:
            continue

        if noise_type == 'poisson':
            noisy = add_poisson_noise(image)

        elif noise_type == 'jpeg':
            noisy = add_jpeg_noise(image)

        elif noise_type == 'lowlight':
            noisy = add_lowlight_noise(image)

        else:
            raise ValueError('Unknown noise type')

        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, noisy)


if __name__ == '__main__':
    generate_noisy_dataset(
        '../datasets/resized_images',
        '../noisy_data/poisson',
        'poisson'
    )
    generate_noisy_dataset(
        '../datasets/resized_images',
        '../noisy_data/jpeg',
        'jpeg'
    )
    generate_noisy_dataset(
        '../datasets/resized_images',
        '../noisy_data/lowlight',
        'lowlight'
    )