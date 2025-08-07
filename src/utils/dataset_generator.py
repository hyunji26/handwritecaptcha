import os
import csv
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import random
from PIL import ImageFilter
import numpy as np

class HandwritingDatasetGenerator:
    def __init__(self, font_paths, background_color=(255, 255, 255), text_color=(0, 0, 0)):
        """
        font_paths: .ttf 폰트 파일 경로 리스트
        """
        self.fonts = []
        for font_path in font_paths:
            try:
                # 폰트 이름 추출 (파일명에서 확장자 제외)
                font_name = os.path.splitext(os.path.basename(font_path))[0]
                font = ImageFont.truetype(font_path, size=32)  # 기본 크기
                self.fonts.append((font_name, font))
            except Exception as e:
                print(f"폰트 로드 실패 {font_path}: {e}")
        
        self.background_color = background_color
        self.text_color = text_color

    def create_text_image(self, text, font, size=None, padding=None):
        """텍스트를 이미지로 변환"""
        # 폰트 크기 무작위 지정
        random_size = random.randint(28, 40)
        if size is None:
            size = random_size
        # 패딩 무작위 지정
        random_padding_x = random.randint(10, 30)
        random_padding_y = random.randint(10, 30)
        if padding is None:
            padding = (random_padding_x, random_padding_y)
        
        font = ImageFont.truetype(font.path, size=size)
        
        # 텍스트 크기 측정
        dummy_img = Image.new('RGB', (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        text_bbox = dummy_draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 패딩을 포함한 이미지 크기
        img_width = text_width + (2 * padding[0])
        img_height = text_height + (2 * padding[1])
        
        # 이미지 생성
        image = Image.new('RGB', (img_width, img_height), self.background_color)
        draw = ImageDraw.Draw(image)
        
        # 텍스트 위치 무작위 지정 (이미지 밖으로 나가지 않게)
        max_x = max(padding[0], img_width - text_width - 1)
        max_y = max(padding[1], img_height - text_height - 1)
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        draw.text((x, y), text, font=font, fill=self.text_color)
        
        # --- Augmentation: Gaussian blur, noise, affine transform 중 하나 무작위 적용 ---
        aug_type = random.choice(['blur', 'noise', 'affine', 'none'])
        if aug_type == 'blur':
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        elif aug_type == 'noise':
            arr = np.array(image)
            noise = np.random.randint(0, 50, arr.shape, dtype='uint8')
            mask = np.random.rand(*arr.shape[:2]) < 0.03  # 3% salt & pepper
            arr[mask] = noise[mask]
            image = Image.fromarray(arr)
        elif aug_type == 'affine':
            dx = random.randint(-5, 5)
            dy = random.randint(-3, 3)
            coeffs = (1, random.uniform(-0.2, 0.2), dx, random.uniform(-0.2, 0.2), 1, dy)
            image = image.transform(image.size, Image.AFFINE, coeffs, resample=Image.BICUBIC)
        # else: 아무것도 적용하지 않음
        
        return image

    def generate_dataset(self, word_list_path, output_dir):
        """단어 리스트에서 텍스트를 읽어 데이터셋 생성"""
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        
        # 단어 리스트 읽기
        with open(word_list_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f.readlines() if line.strip()]
        
        # CSV 레이블 파일 생성
        csv_path = os.path.join(output_dir, 'labels.csv')
        with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'label'])  # 헤더 작성
            
            # 진행 상황 표시
            total_combinations = len(words) * len(self.fonts)
            pbar = tqdm(total=total_combinations, desc="데이터셋 생성 중")
            
            # 각 단어와 폰트 조합으로 이미지 생성
            for word_idx, word in enumerate(words):
                for font_idx, (font_name, font) in enumerate(self.fonts):
                    try:
                        # 이미지 파일명 생성 ([단어]_[폰트]_[word_idx]_[font_idx].png)
                        image_filename = f'{word}_{font_name}_{word_idx:03d}_{font_idx:02d}.png'
                        image_path = os.path.join(output_dir, 'images', image_filename)
                        
                        # 이미지 생성 및 저장
                        image = self.create_text_image(word, font)
                        image.save(image_path)
                        
                        # CSV에 레이블 정보 저장
                        writer.writerow([image_filename, word])
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"이미지 생성 실패 (단어: {word}, 폰트: {font_name}): {e}")
            
            pbar.close()
        
        print("데이터셋 생성 완료!")
        print(f"생성된 이미지 개수: {total_combinations}")
        print(f"레이블 파일 저장 위치: {csv_path}")

def main():
    # 설정
    word_list_path = 'data/word_list.txt'
    font_dir = 'data/fonts'  # .ttf 파일들이 있는 디렉토리
    output_dir = 'data/generated_dataset'
    
    # 폰트 파일 목록 가져오기
    font_paths = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if f.endswith('.ttf')]
    
    if not font_paths:
        print("경고: data/fonts 디렉토리에 .ttf 파일이 없습니다!")
        return
        
    print(f"발견된 폰트 파일: {len(font_paths)}개")
    
    # 데이터셋 생성기 초기화
    generator = HandwritingDatasetGenerator(font_paths)
    
    # 데이터셋 생성
    generator.generate_dataset(word_list_path, output_dir)

if __name__ == '__main__':
    main()