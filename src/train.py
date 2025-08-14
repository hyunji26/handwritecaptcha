import os
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn import CTCLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from contextlib import nullcontext

from src.utils.data_utils import HandwritingDataset, decode_prediction, ctc_collate_fn
from src.utils.metrics import character_error_rate
from src.model.crnn import CRNN


def compute_input_widths_from_padded_images(padded_images: torch.Tensor) -> torch.Tensor:
    """배치 패딩 이미지에서 각 샘플의 실제 폭(유효 컬럼 수)을 계산합니다.

    padded_images: [B, 1, H, W_max]
    returns: [B] LongTensor (CPU)
    """
    with torch.no_grad():
        col_sum = padded_images.sum(dim=2).sum(dim=1)  # [B, W]
        widths = (col_sum > 0).sum(dim=1).to(torch.long)
    return widths


def compute_output_lengths(input_widths: torch.Tensor) -> torch.Tensor:
    """basic 백본 기준 출력 길이 추정: T = max(1, W//4 - 1)"""
    t = input_widths // 4 - 1
    return torch.clamp(t, min=1)


def greedy_decode_with_lengths(outputs: torch.Tensor, idx_to_char, lengths: torch.Tensor):
    with torch.no_grad():
        probs = outputs.detach().permute(1, 0, 2).cpu()  # [B, T, C]
        if torch.is_tensor(lengths):
            lengths = lengths.detach().cpu().tolist()
        hyps = []
        for i, p in enumerate(probs):
            t_i = int(max(0, lengths[i])) if lengths is not None else p.shape[0]
            t_i = min(t_i, p.shape[0])
            seq = p[:t_i].argmax(dim=1).tolist()
            out = []
            prev = -1
            for c in seq:
                if c != prev and c != 0:
                    out.append(idx_to_char[c])
                prev = c
            hyps.append(''.join(out))
    return hyps


def evaluate_full_cer(model, val_loader, criterion, device, idx_to_char):
    model.eval()
    val_total_loss = 0.0
    all_refs, all_hyps = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_widths = compute_input_widths_from_padded_images(batch['image'])
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            label_lengths = batch['label_length']

            outputs = model(images)  # (T, B, C)
            output_lengths = compute_output_lengths(input_widths)

            loss = criterion(outputs.log_softmax(2), labels, output_lengths, label_lengths)
            val_total_loss += loss.item()

            hyps = greedy_decode_with_lengths(outputs, idx_to_char, output_lengths)
            refs = batch['text']
            n = min(len(hyps), len(refs))
            all_hyps.extend(hyps[:n])
            all_refs.extend(refs[:n])

    avg_val_loss = val_total_loss / max(1, len(val_loader))
    val_cer = character_error_rate(all_refs, all_hyps) if all_refs else 1.0
    return avg_val_loss, val_cer


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_dataset = HandwritingDataset(args.train_image_paths, args.train_labels, args.char_to_idx)
    #배치 샘플링(가중치 기반, 타겟 비율 제어)
    if getattr(args, 'sampler_weights', None):
        sampler = WeightedRandomSampler(weights=torch.tensor(args.sampler_weights, dtype=torch.double), num_samples=len(train_dataset), replacement=True)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=ctc_collate_fn, sampler=sampler
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=ctc_collate_fn
        )
    val_dataset = HandwritingDataset(args.val_image_paths, args.val_labels, args.char_to_idx)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=ctc_collate_fn
    )

    model = CRNN(num_channels=1, num_classes=len(args.char_to_idx) + 1).to(device)

    criterion = CTCLoss(zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    best_cer = math.inf
    patience = getattr(args, 'early_stopping_patience', 12)
    patience_counter = 0

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')

        for batch_idx, batch in enumerate(progress_bar):
            input_widths = compute_input_widths_from_padded_images(batch['image'])
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            label_lengths = batch['label_length']

            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                outputs = model(images)  # (T, B, C)
                output_lengths = compute_output_lengths(input_widths)
                loss = criterion(
                    outputs.log_softmax(2),
                    labels,
                    output_lengths,
                    label_lengths
                )

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/batch', loss.item(), global_step)

            if args.print_freq > 0 and batch_idx % args.print_freq == 0:
                hyps = greedy_decode_with_lengths(outputs, args.idx_to_char, output_lengths)
                refs = batch['text']
                n = min(len(hyps), len(refs))
                sample_cer = character_error_rate(refs[:n], hyps[:n]) if n > 0 else 1.0
                if n > 0:
                    print(f'\nSample prediction: {hyps[0]}')
                    print(f'Ground truth: {refs[0]}')
                writer.add_scalar('CER/sample', sample_cer, global_step)

        avg_train_loss = total_loss / max(1, len(train_loader))
        print(f'\nEpoch {epoch+1} train loss: {avg_train_loss:.4f}')
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

        avg_val_loss, val_cer = evaluate_full_cer(
            model, val_loader, criterion, device,
            idx_to_char=args.idx_to_char,
        )
        print(f'Epoch {epoch+1} val loss: {avg_val_loss:.4f} | val CER: {val_cer:.4f}')
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('CER/val_epoch', val_cer, epoch)

        scheduler.step(avg_val_loss)

        ckpt_last = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'val_cer': val_cer,
        }
        torch.save(ckpt_last, os.path.join(args.save_dir, 'last.pth'))

        if val_cer < best_cer - 1e-6:
            best_cer = val_cer
            torch.save(ckpt_last, os.path.join(args.save_dir, 'best_model_by_cer.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1} (best CER: {best_cer:.4f})')
                break

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_image_paths', type=list, required=True)
    parser.add_argument('--train_labels', type=list, required=True)
    parser.add_argument('--char_to_idx', type=dict, required=True)
    parser.add_argument('--idx_to_char', type=dict, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--clip_grad', type=float, default=5.0)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--log_dir', type=str, default='runs')
    args = parser.parse_args()
    train(args)

import os
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from contextlib import nullcontext

from src.utils.data_utils import HandwritingDataset, decode_prediction, ctc_collate_fn
from src.utils.metrics import character_error_rate
from src.model.crnn import CRNN


def _compute_input_widths_from_padded_images(padded_images: torch.Tensor) -> torch.Tensor:
    """배치 패딩 이미지에서 각 샘플의 실제 폭(유효 컬럼 수)을 계산합니다.

    padded_images: [B, 1, H, W_max] (CPU 텐서 권장)
    returns: [B] LongTensor (CPU)
    """
    with torch.no_grad():
        # 각 샘플별 컬럼 합 -> 0보다 큰 컬럼 수가 원본 폭
        # [B, 1, H, W] → 합계 [B, W]
        col_sum = padded_images.sum(dim=2).sum(dim=1)
        widths = (col_sum > 0).sum(dim=1).to(torch.long)
    return widths


def _compute_output_lengths_from_input_widths(input_widths: torch.Tensor, cnn_backbone: str) -> torch.Tensor:
    """입력 폭으로부터 모델 타임스텝 길이 추정.

    basic: T = max(1, W//4 - 1)
    resnet18: T = max(1, W//16)
    """
    if cnn_backbone == 'basic':
        t = input_widths // 4 - 1
    else:
        t = input_widths // 16
    return torch.clamp(t, min=1)


def _greedy_decode_with_lengths(outputs: torch.Tensor, idx_to_char, lengths: torch.Tensor):
    """lengths로 유효 타임스텝을 제한하여 CTC 그리디 디코딩.

    outputs: [T, B, C], lengths: [B] (CPU Long)
    returns List[str]
    """
    with torch.no_grad():
        probs = outputs.detach().permute(1, 0, 2).cpu()  # [B, T, C]
        if torch.is_tensor(lengths):
            lengths = lengths.detach().cpu().tolist()
        hyps = []
        for i, p in enumerate(probs):
            t_i = int(max(0, lengths[i])) if lengths is not None else p.shape[0]
            t_i = min(t_i, p.shape[0])
            seq = p[:t_i].argmax(dim=1).tolist()
            out = []
            prev = -1
            for c in seq:
                if c != prev and c != 0:  # 0 = blank
                    out.append(idx_to_char[c])
                prev = c
            hyps.append(''.join(out))
    return hyps


def evaluate_full_cer(model, val_loader, criterion, device, idx_to_char, blank_idx):
    model.eval()
    val_total_loss = 0.0
    all_refs, all_hyps = [], []

    with torch.no_grad():
        for batch in val_loader:
            # 길이 계산은 CPU에서 패딩 이미지로 추정
            input_widths = _compute_input_widths_from_padded_images(batch['image'])  # [B]
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            label_lengths = batch['label_length']
            outputs = model(images)                     # (T, B, C)
            output_lengths = _compute_output_lengths_from_input_widths(
                input_widths, getattr(model, 'cnn_backbone', 'basic')
            )
            loss = criterion(outputs.log_softmax(2), labels, output_lengths, label_lengths)
            val_total_loss += loss.item()

            # 디코드 & 참조 수집
            hyps = _greedy_decode_with_lengths(outputs, idx_to_char, output_lengths)  # List[str]
            refs = batch['text']                            # List[str]
            # 길이 맞추기 (안전)
            n = min(len(hyps), len(refs))
            all_hyps.extend(hyps[:n])
            all_refs.extend(refs[:n])

    avg_val_loss = val_total_loss / max(1, len(val_loader))
    val_cer = character_error_rate(all_refs, all_hyps) if all_refs else 1.0
    return avg_val_loss, val_cer


def train(args):
    # ===== Device & AMP =====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ===== Dataloaders =====
    train_dataset = HandwritingDataset(args.train_image_paths, args.train_labels, args.char_to_idx)

    # --- 두-스트림 고정 비율 배치 샘플러 빌더 ---
    def _build_fixed_ratio_loader(epoch: int):
        import random as _rnd

        # 비율 커리큘럼: 선형 증가
        warm = int(getattr(args, 'ratio_warmup_epochs', 0) or 0)
        r0 = float(getattr(args, 'target_ratio_start', 0.3))
        r1 = float(getattr(args, 'target_ratio_end', 0.4))
        if warm > 0:
            alpha = min(1.0, max(0.0, epoch / float(warm)))
            ratio = r0 + (r1 - r0) * alpha
        else:
            ratio = r1

        B = int(args.batch_size)
        B_t = max(1, int(round(B * ratio)))
        B_g = max(1, B - B_t)

        # 인덱스 분리
        target_labels = set(getattr(args, 'target_label_set', set()) or set())
        tgt_idx = [i for i, lb in enumerate(args.train_labels) if lb in target_labels]
        gen_idx = [i for i, lb in enumerate(args.train_labels) if lb not in target_labels]
        Nt, No = len(tgt_idx), len(gen_idx)

        # 스텝 수: min( floor(No/B_g), floor(R_max * Nt / B_t) )
        R_max = int(getattr(args, 'target_repeat_max', 6) or 6)
        steps_by_g = No // max(1, B_g)
        steps_by_t = (R_max * max(1, Nt)) // max(1, B_t)
        steps = max(1, min(steps_by_g, steps_by_t))

        rng = _rnd.Random(1234 + epoch)
        batches = []
        for _ in range(steps):
            bt = [rng.choice(tgt_idx) for __ in range(B_t)] if Nt > 0 else []
            bg = [rng.choice(gen_idx) for __ in range(B_g)] if No > 0 else []
            batch = bt + bg
            rng.shuffle(batch)       # 배치 내부 셔플
            batches.append(batch)
        rng.shuffle(batches)         # 배치 순서 셔플

        class _FixedBatchSampler:
            def __init__(self, batches):
                self._batches = batches
            def __iter__(self):
                for b in self._batches:
                    yield b
            def __len__(self):
                return len(self._batches)

        sampler = _FixedBatchSampler(batches)
        loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=ctc_collate_fn,
        )
        return loader

    # --- 학습 로더 설정: 고정 비율 > 가중 샘플링 > 기본 셔플 ---
    if getattr(args, 'use_fixed_ratio_batch', False) and getattr(args, 'target_label_set', None):
        train_loader = _build_fixed_ratio_loader(epoch=0)
    elif getattr(args, 'sampler_weights', None):
        # 에폭별 재현성 있는 샘플링을 원하면 gen에 시드 부여 가능
        gen = torch.Generator()
        gen.manual_seed(1234)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.tensor(args.sampler_weights, dtype=torch.double),
            num_samples=len(train_dataset),
            replacement=True,
            generator=gen if 'generator' in torch.utils.data.WeightedRandomSampler.__init__.__code__.co_varnames else None,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=ctc_collate_fn,
            sampler=sampler,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=ctc_collate_fn
        )
    val_dataset = HandwritingDataset(args.val_image_paths, args.val_labels, args.char_to_idx)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=ctc_collate_fn
    )

    # ===== Model =====
    model = CRNN(
        num_channels=1,
        num_classes=len(args.char_to_idx) + 1,  # CTC blank(0) 포함
        cnn_backbone=getattr(args, 'cnn_backbone', 'basic'),
        use_pretrained_backbone=getattr(args, 'use_pretrained_backbone', False),
    ).to(device)
    model._init_weights()

    # ===== Optim / Sched / Loss =====
    criterion = CTCLoss(zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # ===== Resume (체크포인트 불러오기) =====
    start_epoch = 0
    best_cer = float('inf')
    best_epoch = -1
    patience_counter = 0

    resume_path = getattr(args, 'resume', None)
    if resume_path and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        best_cer = ck.get('val_cer', float('inf'))
        start_epoch = int(ck.get('epoch', -1)) + 1
        print(f"▶ Resumed from {resume_path} at epoch {start_epoch}, best_cer={best_cer:.4f}")

    # ===== Logging =====
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    # ===== EarlyStopping (by CER) =====
    patience = getattr(args, 'early_stopping_patience', 12)
    warmup_epochs = getattr(args, 'warmup_epochs', 3)
    best_cer = math.inf
    best_epoch = -1
    patience_counter = 0

    # ===== Train Loop =====
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        total_loss = 0.0

        # --- 에폭 시작 시 로더 재생성(비율 커리큘럼 적용) ---
        if getattr(args, 'use_fixed_ratio_batch', False) and getattr(args, 'target_label_set', None):
            train_loader = _build_fixed_ratio_loader(epoch)

        # --- Warmup LR ---
        if epoch < warmup_epochs:
            # 선형 워밍업: base_lr * ((epoch+1)/warmup_epochs)
            warm_ratio = float(epoch + 1) / float(max(1, warmup_epochs))
            for pg in optimizer.param_groups:
                pg["lr"] = args.learning_rate * warm_ratio

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}')

        for batch_idx, batch in enumerate(progress_bar):
            input_widths = _compute_input_widths_from_padded_images(batch['image'])
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            label_lengths = batch['label_length']

            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                outputs = model(images)  # (T, B, C)
                output_lengths = _compute_output_lengths_from_input_widths(
                    input_widths, getattr(model, 'cnn_backbone', 'basic')
                )
                loss = criterion(
                    outputs.log_softmax(2),
                    labels,
                    output_lengths,
                    label_lengths
                )

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/batch', loss.item(), global_step)
            # (선택) 현재 LR 로깅
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step)

            # 샘플 프린트는 빈도 낮춤
            if args.print_freq > 0 and batch_idx % args.print_freq == 0:
                hyps = _greedy_decode_with_lengths(outputs, args.idx_to_char, output_lengths)
                refs = batch['text']
                n = min(len(hyps), len(refs))
                sample_cer = character_error_rate(refs[:n], hyps[:n]) if n > 0 else 1.0
                if n > 0:
                    print(f'\nSample prediction: {hyps[0]}')
                    print(f'Ground truth: {refs[0]}')
                writer.add_scalar('CER/sample', sample_cer, global_step)

        avg_train_loss = total_loss / max(1, len(train_loader))
        print(f'\nEpoch {epoch+1} train loss: {avg_train_loss:.4f}')
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

        # ===== Validation: loss + CER =====
        avg_val_loss, val_cer = evaluate_full_cer(
            model, val_loader, criterion, device,
            idx_to_char=args.idx_to_char,
            blank_idx=len(args.char_to_idx)  # 사용처에 따라 필요시 활용
        )
        print(f'Epoch {epoch+1} val loss: {avg_val_loss:.4f} | val CER: {val_cer:.4f}')
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('CER/val_epoch', val_cer, epoch)

        # 스케줄러는 보수적으로 val loss를 기준
        scheduler.step(avg_val_loss)

        # ===== Checkpointing (best by CER) =====
        ckpt_last = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'val_cer': val_cer,
        }
        torch.save(ckpt_last, os.path.join(args.save_dir, 'last.pth'))

        improved = val_cer < (best_cer - 1e-6)
        if improved:
            best_cer = val_cer
            best_epoch = epoch
            patience_counter = 0
            torch.save(ckpt_last, os.path.join(args.save_dir, 'best_model_by_cer.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1} (best epoch: {best_epoch+1}, val_CER={best_cer:.4f})')
                break

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 주의: list/dict 타입은 CLI에서 직접 파싱 어려움. 일반적으로는 JSON 경로를 받아 읽는 방식을 권장.
    parser.add_argument('--train_image_paths', type=list, required=True)
    parser.add_argument('--train_labels', type=list, required=True)
    parser.add_argument('--char_to_idx', type=dict, required=True)
    parser.add_argument('--idx_to_char', type=dict, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=120)               # ↑ 기본 에폭 확대
    parser.add_argument('--clip_grad', type=float, default=5.0)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--early_stopping_patience', type=int, default=12)   # ↑ CER 기준 patience
    parser.add_argument('--warmup_epochs', type=int, default=3)              # ↑ 워밍업
    args = parser.parse_args()
    train(args)
