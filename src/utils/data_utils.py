import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

def load_image(image_path):
    """이미지를 로드하고 전처리합니다."""
    image = Image.open(image_path).convert('L')  # 그레이스케일로 변환
    return image

def process_image(image, target_height=32):
    """이미지를 지정된 높이로 리사이즈하고 정규화합니다."""
    w, h = image.size
    new_w = int(w * (target_height / h))
    image = image.resize((new_w, target_height), Image.LANCZOS)
    
    # PIL Image를 numpy array로 변환
    img_array = np.array(image).astype(np.float32)
    
    # 정규화 (0-1 범위로)
    img_array = img_array / 255.0
    
    return img_array

def encode_text(text, char_to_idx):
    """텍스트를 인덱스 시퀀스로 변환합니다."""
    return [char_to_idx[char] for char in text]

def decode_prediction(pred, idx_to_char):
    """CTC 디코딩을 수행합니다."""
    # pred: [seq_length, batch_size, num_classes]
    pred = pred.permute(1, 0, 2).cpu().numpy()  # [batch_size, seq_length, num_classes]
    
    outputs = []
    for p in pred:
        p = p.argmax(axis=1)  # 각 타임스텝에서 가장 높은 확률의 문자 선택
        
        # Merge repeated characters and remove blank label
        previous = -1
        out = []
        for c in p:
            if c != previous and c != 0:  # 0은 blank label
                out.append(idx_to_char[c])
            previous = c
        outputs.append(''.join(out))
    
    return outputs

class HandwritingDataset(Dataset):
    """손글씨 데이터셋 클래스"""
    def __init__(self, image_paths, labels, char_to_idx, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.char_to_idx = char_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 이미지 로드 및 전처리
        image = load_image(image_path)
        image = process_image(image)
        
        if self.transform:
            image = self.transform(image)
        
        # 텍스트를 인덱스로 변환
        label_encoded = encode_text(label, self.char_to_idx)
        
        return {
            'image': torch.FloatTensor(image).unsqueeze(0),  # [1, H, W]
            'label': torch.LongTensor(label_encoded),
            'label_length': len(label_encoded),
            'text': label
        }

