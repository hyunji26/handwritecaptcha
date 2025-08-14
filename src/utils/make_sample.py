import os
import argparse
import random
import time
from typing import Tuple
from PIL import Image, ImageDraw, ImageFont


def load_font_safe(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(font_path, size=size, layout_engine=ImageFont.LAYOUT_BASIC)
    except Exception:
        return ImageFont.truetype(font_path, size=size)


def render_text_image(text: str, font_path: str, size: int = 48, padding: Tuple[int, int] = (24, 24),
                      bg=(255, 255, 255), fg=(0, 0, 0)) -> Image.Image:
    font = load_font_safe(font_path, size=size)
    # measure
    dummy = Image.new('RGB', (1, 1))
    draw_dummy = ImageDraw.Draw(dummy)
    bbox = draw_dummy.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    iw = tw + padding[0] * 2
    ih = th + padding[1] * 2

    img = Image.new('RGB', (iw, ih), bg)
    draw = ImageDraw.Draw(img)
    x = (iw - tw) // 2
    y = (ih - th) // 2
    draw.text((x, y), text, font=font, fill=fg)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, default='피자')
    parser.add_argument('--font_dir', type=str, default='data/fonts')
    parser.add_argument('--output_dir', type=str, default='data/generated_data/sample')
    parser.add_argument('--size', type=int, default=48)
    args = parser.parse_args()

    if not os.path.isdir(args.font_dir):
        raise FileNotFoundError(f'폰트 디렉터리를 찾을 수 없습니다: {args.font_dir}')

    fonts = [
        os.path.join(args.font_dir, f)
        for f in os.listdir(args.font_dir)
        if os.path.splitext(f)[1].lower() in ('.ttf', '.otf')
    ]
    if not fonts:
        raise FileNotFoundError('폰트 파일(.ttf/.otf)을 찾을 수 없습니다.')

    font_path = random.choice(fonts)
    font_name = os.path.splitext(os.path.basename(font_path))[0]

    img = render_text_image(args.text, font_path, size=args.size)

    os.makedirs(args.output_dir, exist_ok=True)
    ts = int(time.time())
    # 파일명에 파일 시스템에 안전하지 않은 문자를 최소화
    safe_text = args.text.replace('/', '_').replace('\\', '_')
    filename = f'{safe_text}_{font_name}_{ts}.png'
    save_path = os.path.join(args.output_dir, filename)
    img.save(save_path)
    print(f'Saved: {save_path}')


if __name__ == '__main__':
    main()



