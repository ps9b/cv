from PIL import Image
import os

# 입력 폴더와 출력 폴더 경로
input_dir = "./datasets/train"
output_dir = "./datasets/resized_images"

# 출력 폴더 생성
os.makedirs(output_dir, exist_ok=True)

# 지원하는 이미지 확장자
extensions = (".jpg", ".jpeg", ".png", ".bmp")

# 이미지 resize 수행
for filename in os.listdir(input_dir):
    if filename.lower().endswith(extensions):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 이미지 열기
        img = Image.open(input_path)

        # 128x128로 resize
        resized_img = img.resize((128, 128))

        # 저장
        resized_img.save(output_path)

        print(f"Resized: {filename}")

print("모든 이미지 resize 완료!")