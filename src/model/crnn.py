import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_channels, num_classes, hidden_size=256):
        super(CRNN, self).__init__()
        
        # CNN 부분
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Layer 4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            
            # Layer 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Layer 6
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),
            
            # Layer 7
            nn.Conv2d(512, 512, kernel_size=2, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # RNN 부분
        self.rnn = nn.Sequential(
            nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True),
            nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        )
        
        # 출력 레이어
        self.linear = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # CNN 특징 추출
        conv = self.cnn(x)  # [batch_size, channels, height, width]
        
        # CNN 출력을 RNN 입력 형태로 변환
        batch_size, channels, height, width = conv.size()
        conv = conv.squeeze(2)  # [batch_size, channels, width]
        conv = conv.permute(0, 2, 1)  # [batch_size, width, channels]
        
        # RNN으로 시퀀스 처리
        rnn_input = conv
        self.rnn.flatten_parameters()
        for i in range(2):
            rnn_input, _ = self.rnn[i](rnn_input)
        
        # 최종 출력
        output = self.linear(rnn_input)
        output = output.permute(1, 0, 2)  # [width, batch_size, num_classes]
        
        return output

    def _init_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

