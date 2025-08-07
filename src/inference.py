import torch
import argparse
from PIL import Image
import numpy as np

from model.crnn import CRNN
from utils.data_utils import process_image, decode_prediction

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
            num_classes=len(char_to_idx)
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
    # 예측기 초기화
    predictor = HandwritingPredictor(
        args.model_path,
        args.char_to_idx,
        args.idx_to_char
    )
    
    # 이미지에서 텍스트 예측
    prediction = predictor.predict(args.image_path)
    print(f'Predicted text: {prediction}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--char_to_idx', type=dict, required=True)
    parser.add_argument('--idx_to_char', type=dict, required=True)
    
    args = parser.parse_args()
    main(args)

