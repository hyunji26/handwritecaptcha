import os
import json
import torch
import argparse
from PIL import Image
import numpy as np

from .model.crnn import CRNN
from .utils.data_utils import process_image, decode_prediction

class HandwritingPredictor:
    def __init__(self, model_path, char_to_idx, idx_to_char, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        
        # 모델 로드
        self.model = CRNN(
            num_channels=1,
            num_classes=len(char_to_idx) + 1  # CTC blank(0) 포함
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def predict(self, image):
        """이미지에서 텍스트 예측"""
        # 이미지 전처리
        if isinstance(image, str):
            image = Image.open(image).convert('L')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('L')
        
        img_array = process_image(image)
        img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        # 예측
        with torch.no_grad():
            outputs = self.model(img_tensor)
            predictions = decode_prediction(outputs, self.idx_to_char)
        
        return predictions[0]

def main(args):
    # 문자 집합 로드 (charset_path 우선)
    if args.charset_path:
        with open(args.charset_path, 'r', encoding='utf-8') as f:
            charset = json.load(f)
            idx_to_char = charset['idx_to_char']
            char_to_idx = charset['char_to_idx']
    else:
        idx_to_char = args.idx_to_char
        char_to_idx = args.char_to_idx

    # 예측기 초기화
    predictor = HandwritingPredictor(
        args.model_path,
        char_to_idx,
        idx_to_char
    )
    
    # 이미지에서 텍스트 예측
    prediction = predictor.predict(args.image_path)
    print(f'Predicted text: {prediction}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--charset_path', type=str, default=None, help='models/charset.json 경로')
    # 하위 호환: 직접 dict 전달도 허용 (권장 X)
    parser.add_argument('--char_to_idx', type=dict, default=None)
    parser.add_argument('--idx_to_char', type=dict, default=None)

    args = parser.parse_args()
    main(args)

