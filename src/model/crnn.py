import torch
import torch.nn as nn
from typing import Literal


class CRNN(nn.Module):
    """Simple CRNN for OCR with CTC.

    - CNN backbone downsamples height to 1 while keeping width as time steps
    - Two-layer BiLSTM head
    - Output shape: [T, B, C]
    """

    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        hidden_size: int = 256,
        cnn_backbone: Literal['basic'] = 'basic',
        use_pretrained_backbone: bool = False,  # 호환성 유지용 (무시)
        **_: dict,
    ) -> None:
        super().__init__()

        assert cnn_backbone == 'basic', 'Only basic backbone is implemented in this rewrite.'
        self.cnn_backbone = cnn_backbone

        # Basic CNN: progressively reduce height
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2, W/2

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/4, W/4

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/8, W/4

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/16, W/4

            nn.Conv2d(512, 512, kernel_size=2, stride=1),  # reduce H by 1 more step
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.rnn = nn.Sequential(
            nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True),
            nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True),
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        feat = self.cnn(x)  # [B, C', H', W'] with H' small
        feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, None))  # [B, C', 1, W]
        feat = feat.squeeze(2).permute(0, 2, 1)  # [B, W, C']

        out, _ = self.rnn[0](feat)
        out, _ = self.rnn[1](out)
        out = self.fc(out)  # [B, W, num_classes]
        out = out.permute(1, 0, 2)  # [T=W, B, num_classes]
        return out


