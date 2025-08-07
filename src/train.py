import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from tqdm import tqdm
from tensorboard.writer import SummaryWriter

from utils.data_utils import HandwritingDataset, decode_prediction
from model.crnn import CRNN

def train(args):
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터 로더 설정
    train_dataset = HandwritingDataset(
        args.train_image_paths,
        args.train_labels,
        args.char_to_idx
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # 모델 설정
    model = CRNN(
        num_channels=1,
        num_classes=len(args.char_to_idx)
    ).to(device)
    model._init_weights()
    
    # 손실 함수와 옵티마이저 설정
    criterion = CTCLoss(zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # TensorBoard 설정
    writer = SummaryWriter(args.log_dir)
    
    # 학습 루프
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            label_lengths = batch['label_length']
            
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(images)
            output_lengths = torch.full(
                size=(outputs.size(1),),
                fill_value=outputs.size(0),
                dtype=torch.long
            )
            
            # CTC 손실 계산
            loss = criterion(
                outputs.log_softmax(2),
                labels,
                output_lengths,
                label_lengths
            )
            
            # 역전파
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            
            total_loss += loss.item()
            
            # 진행 상황 업데이트
            progress_bar.set_postfix({'loss': loss.item()})
            
            # TensorBoard에 배치 손실 기록
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/batch', loss.item(), global_step)
            
            # 예측 결과 샘플 출력
            if batch_idx % args.print_freq == 0:
                pred = decode_prediction(outputs, args.idx_to_char)
                print(f'\nSample prediction: {pred[0]}')
                print(f'Ground truth: {batch["text"][0]}')
        
        # 에폭 평균 손실 계산
        avg_loss = total_loss / len(train_loader)
        print(f'\nEpoch {epoch+1} average loss: {avg_loss:.4f}')
        
        # TensorBoard에 에폭 손실 기록
        writer.add_scalar('Loss/epoch', avg_loss, epoch)
        
        # 학습률 조정
        scheduler.step(avg_loss)
        
        # 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(args.save_dir, 'best_model.pth'))
    
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_image_paths', type=list, required=True)
    parser.add_argument('--train_labels', type=list, required=True)
    parser.add_argument('--char_to_idx', type=dict, required=True)
    parser.add_argument('--idx_to_char', type=dict, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--clip_grad', type=float, default=5.0)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--log_dir', type=str, default='runs')
    
    args = parser.parse_args()
    train(args)

