import cv2
import os
from tqdm import tqdm


def gaussian_denoise(input_dir, output_dir, kernel_size=(5, 5)):
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir)):
        path = os.path.join(input_dir, filename)

        image = cv2.imread(path)

        if image is None:
            continue

        denoised = cv2.GaussianBlur(image, kernel_size, 0)

        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, denoised)


if __name__ == '__main__':
    gaussian_denoise(
        '../noisy_data/poisson',
        '../results/gaussian'
    )