import os
import csv
import argparse
from typing import Optional, Iterable
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

class HandwritingDatasetGenerator:
    def __init__(self, font_paths, background_color=(255, 255, 255), text_color=(0, 0, 0), blacklist_names: Optional[Iterable[str]] = None):
        """
        font_paths: .ttf/.otf 폰트 파일 경로 리스트
        """
        self.fonts = []  # (font_name, font_path)
        self.blacklist = set(blacklist_names or [])
        for font_path in font_paths:
            try:
                font_name = os.path.splitext(os.path.basename(font_path))[0]
                if font_name in self.blacklist:
                    print(f"폰트 제외({font_name}): 블랙리스트 매칭")
                    continue
                # 사전 검증: 안전 로더로 간단 렌더링 테스트
                try:
                    font = self._load_font_safe(font_path, size=24)
                    # 실제 드로잉/텍스트 박스 계산까지 테스트
                    tmp = Image.new('RGB', (64, 64), (255, 255, 255))
                    drw = ImageDraw.Draw(tmp)
                    drw.text((2, 2), '테스트', font=font, fill=(0, 0, 0))
                    _ = drw.textbbox((0, 0), '테스트', font=font)
                except Exception as inner:
                    print(f"폰트 제외({font_name}): 사전 렌더링 실패: {inner}")
                    continue
                self.fonts.append((font_name, font_path))
            except Exception as e:
                print(f"폰트 로드 실패 {font_path}: {e}")

        self.background_color = background_color
        self.text_color = text_color

    @staticmethod
    def _load_font_safe(font_path: str, size: int) -> ImageFont.FreeTypeFont:
        """LAYOUT_BASIC 우선, 실패 시 일반 로드로 폴백"""
        try:
            return ImageFont.truetype(font_path, size=size, layout_engine=ImageFont.LAYOUT_BASIC)
        except Exception:
            return ImageFont.truetype(font_path, size=size)

    def create_text_image(self, text, font_path, size: Optional[int] = None, padding: Optional[tuple] = None):
        """텍스트를 이미지로 변환 (증강 없음, 결정적 생성)

        - size 기본값: 32
        - padding 기본값: (16, 16)
        - 텍스트는 중앙 정렬로 배치
        """
        if size is None:
            size = 32
        if padding is None:
            padding = (16, 16)

        # 폰트 로드 (안전 로더 사용)
        font = self._load_font_safe(font_path, size=size)
        
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
        # 텍스트 중앙 정렬
        x = (img_width - text_width) // 2
        y = (img_height - text_height) // 2
        draw.text((x, y), text, font=font, fill=self.text_color)
        
        return image

    def generate_dataset(self, word_list_path: str, output_dir: str, seed: Optional[int] = None):
        """단어 리스트에서 텍스트를 읽어 데이터셋 생성 (증강 없이, 단어×폰트 조합당 1장 고정)"""

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

            total_images = len(words) * len(self.fonts)
            pbar = tqdm(total=total_images, desc="데이터셋 생성 중")
            print(f"유효 폰트 수: {len(self.fonts)}개, 단어 수: {len(words)}개, 증강: 없음 (조합당 1장)")

            # 각 단어와 폰트 조합으로 이미지 생성
            for word_idx, word in enumerate(words):
                for font_idx, (font_name, font_path) in enumerate(self.fonts):
                    try:
                        image_filename = f'{word}_{font_name}_{word_idx:03d}_{font_idx:02d}.png'
                        image_path = os.path.join(output_dir, 'images', image_filename)
                        image = self.create_text_image(word, font_path)
                        image.save(image_path)
                        writer.writerow([image_filename, word])
                        pbar.update(1)
                    except Exception as e:
                        print(f"이미지 생성 실패 (단어: {word}, 폰트: {font_name}): {e}")

            pbar.close()

        print("데이터셋 생성 완료!")
        print(f"생성된 이미지 개수: {total_images}")
        print(f"레이블 파일 저장 위치: {csv_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_list_path', type=str, default='data/word_list.txt')
    parser.add_argument('--font_dir', type=str, default='data/fonts')
    parser.add_argument('--output_dir', type=str, default='data/generated_dataset')
    # 증강 제거에 따라 augment 옵션은 사용하지 않습니다.
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    word_list_path = args.word_list_path
    font_dir = args.font_dir
    output_dir = args.output_dir

    # 폰트 파일 목록 가져오기 (.ttf/.otf 대소문자 무시)
    font_paths = [
        os.path.join(font_dir, f)
        for f in os.listdir(font_dir)
        if os.path.splitext(f)[1].lower() in ('.ttf', '.otf')
    ]

    if not font_paths:
        print('경고: data/fonts 디렉토리에 .ttf/.otf 파일이 없습니다!')
        return

    print(f'발견된 폰트 파일: {len(font_paths)}개')

    # 데이터셋 생성기 초기화 (문제 폰트 블랙리스트 제외)
    blacklist = {'나눔손글씨 하나손글씨'}
    generator = HandwritingDatasetGenerator(font_paths, blacklist_names=blacklist)

    # 데이터셋 생성 (증강 없음)
    generator.generate_dataset(word_list_path, output_dir, seed=args.seed)

if __name__ == '__main__':
    main()