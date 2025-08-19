# src/train.py
import os
import inspect
import random
import numpy as np
from contextlib import nullcontext
from typing import Mapping, Sequence, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CTCLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.data_utils import HandwritingDataset, ctc_collate_fn
from src.utils.metrics import character_error_rate
from src.model.crnn import CRNN


# =========================
# 유틸: 재현성/시드 고정
# =========================
def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 유효 길이 계산 & 디코딩
# =========================
@torch.no_grad()
def compute_input_widths_from_padded_images(padded_images: torch.Tensor) -> torch.Tensor:
    """
    padded_images: [B, 1, H, W_max] (CPU 권장)
    returns: [B] (Long, CPU) 각 샘플의 실제 유효 폭(0이 아닌 컬럼 수)
    """
    col_sum = padded_images.sum(dim=2).sum(dim=1)  # [B, W]
    widths = (col_sum > 0).sum(dim=1).to(torch.long)
    return widths


def _downsample_ratio_from_backbone(cnn_backbone: str) -> int:
    """
    백본별 가로축 downsample 비율을 반환.
    - 'basic' : 4 (추정 t = w//4 - 1)
    - 그 외(예: resnet18) : 16 (보편적 구성)
    """
    return 4 if (cnn_backbone or "basic") == "basic" else 16


def compute_output_lengths_from_input_widths(input_widths: torch.Tensor, cnn_backbone: str) -> torch.Tensor:
    """백본 다운샘플 비율에 맞추어 타임스텝 길이를 추정."""
    ratio = _downsample_ratio_from_backbone(cnn_backbone)
    if ratio == 4:
        t = input_widths // 4 - 1
    else:
        t = input_widths // ratio
    return torch.clamp(t, min=1)


@torch.no_grad()
def greedy_decode_with_lengths(
    outputs: torch.Tensor,
    idx_to_char: Union[Sequence[str], Mapping[int, str]],
    lengths: torch.Tensor,
):
    """
    outputs: [T, B, C], lengths: [B] (CPU Long)
    idx_to_char: list 또는 dict 모두 지원(0은 CTC blank)
    """
    # idx -> char 접근 함수
    if isinstance(idx_to_char, Mapping):
        def _itc(i): return idx_to_char.get(i, "")
    else:
        def _itc(i): return idx_to_char[i] if 0 <= i < len(idx_to_char) else ""

    probs = outputs.detach().permute(1, 0, 2).cpu()  # [B, T, C]
    if torch.is_tensor(lengths):
        lengths = lengths.detach().cpu().tolist()
    hyps = []
    for i, p in enumerate(probs):
        t_i = int(max(0, lengths[i])) if lengths is not None else p.shape[0]
        t_i = min(t_i, p.shape[0])
        seq = p[:t_i].argmax(dim=1).tolist()
        out, prev = [], -1
        for c in seq:
            if c != prev and c != 0:  # 0 = blank
                out.append(_itc(c))
            prev = c
        hyps.append(''.join(out))
    return hyps


# =========================
# 검증 루틴
# =========================
def evaluate_full_cer(model, val_loader, criterion, device, idx_to_char):
    model.eval()
    val_total_loss = 0.0
    all_refs, all_hyps = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_widths = compute_input_widths_from_padded_images(batch['image'])  # [B] CPU
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            label_lengths = batch['label_length']

            outputs = model(images)  # (T, B, C)
            output_lengths = compute_output_lengths_from_input_widths(
                input_widths, getattr(model, 'cnn_backbone', 'basic')
            )

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


# =========================
# 두-스트림(고정 비율) 배치 샘플러
# =========================
def build_fixed_ratio_loader(
    epoch: int,
    train_dataset,
    train_labels,
    target_label_set: set,
    batch_size: int,
    target_ratio_start: float,
    target_ratio_end: float,
    ratio_warmup_epochs: int,
    target_repeat_max: int,
    num_workers: int,
):
    import random as _rnd

    # 커리큘럼: epoch 진행에 따라 선형 증가
    if ratio_warmup_epochs > 0:
        alpha = min(1.0, max(0.0, epoch / float(ratio_warmup_epochs)))
        ratio = target_ratio_start + (target_ratio_end - target_ratio_start) * alpha
    else:
        ratio = target_ratio_end

    B = int(batch_size)
    B_t = max(1, int(round(B * ratio)))
    B_g = max(1, B - B_t)

    # 인덱스 분리
    tgt_idx = [i for i, lb in enumerate(train_labels) if lb in target_label_set]
    gen_idx = [i for i, lb in enumerate(train_labels) if lb not in target_label_set]
    Nt, No = len(tgt_idx), len(gen_idx)

    # 스텝 수: min( floor(No/B_g), floor(R_max * Nt / B_t) )
    R_max = int(target_repeat_max)
    steps_by_g = No // max(1, B_g)
    steps_by_t = (R_max * max(1, Nt)) // max(1, B_t)
    steps = max(1, min(steps_by_g, steps_by_t))

    rng = _rnd.Random(1234 + epoch)
    batches = []
    for _ in range(steps):
        bt = [rng.choice(tgt_idx) for __ in range(B_t)] if Nt > 0 else []
        bg = [rng.choice(gen_idx) for __ in range(B_g)] if No > 0 else []
        batch = bt + bg
        rng.shuffle(batch)  # 배치 내부 셔플
        batches.append(batch)
    rng.shuffle(batches)     # 배치 순서 셔플

    print(f"[FixedBatch] epoch={epoch} ratio={ratio:.3f} Bt={B_t} Bg={B_g} steps={steps} Nt={Nt} No={No}")

    class _FixedBatchSampler:
        def __init__(self, batches_):
            self._batches = batches_
        def __iter__(self):
            for b in self._batches:
                yield b
        def __len__(self):
            return len(self._batches)

    sampler = _FixedBatchSampler(batches)
    loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=ctc_collate_fn,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return loader


# =========================
# 가중 샘플링용 weight 계산
# =========================
def compute_sampler_weights(train_labels, target_label_set: set, target_ratio: float):
    t = sum(1 for lb in train_labels if lb in target_label_set)
    g = len(train_labels) - t
    if t == 0 or g == 0:
        return None
    p = max(1e-3, min(1.0 - 1e-3, float(target_ratio)))
    # 기대 타겟 비율 p가 되도록 비타겟 가중치 산출
    w_t = 1.0
    w_g = (t * (1.0 - p)) / (g * p)
    weights = [w_t if lb in target_label_set else max(1e-6, w_g) for lb in train_labels]
    return weights


# =========================
# 학습 본문 (라이브러리 함수)
# =========================
def train(args):
    """
    args는 SimpleNamespace 또는 동등한 객체 형태로 다음 필드를 포함:
    - train_image_paths, train_labels, val_image_paths, val_labels
    - char_to_idx, idx_to_char
    - batch_size, num_workers, learning_rate, num_epochs, clip_grad, print_freq
    - save_dir, log_dir, early_stopping_patience
    - use_fixed_ratio_batch(bool), target_label_set(set), target_ratio_start, target_ratio_end,
      ratio_warmup_epochs, target_repeat_max
    - sampler_weights(Optional[list[float]])
    - cnn_backbone(str, default 'basic'), use_pretrained_backbone(bool)
    - seed(int, default 1234)
    """
    set_seed(getattr(args, 'seed', 1234))

    # 성능 vs 재현성: 필요 시 deterministic 설정
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # 재현성이 최우선이면 False + deterministic=True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext()
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ===== Dataset =====
    train_dataset = HandwritingDataset(args.train_image_paths, args.train_labels, args.char_to_idx)
    val_dataset = HandwritingDataset(args.val_image_paths, args.val_labels, args.char_to_idx)

    # ===== Val loader =====
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=ctc_collate_fn,
        pin_memory=True, persistent_workers=(args.num_workers > 0)
    )

    # ===== Model =====
    model = CRNN(
        num_channels=1,
        num_classes=len(args.char_to_idx) + 1,   # 0=blank + vocab
        cnn_backbone=getattr(args, 'cnn_backbone', 'basic'),
        use_pretrained_backbone=getattr(args, 'use_pretrained_backbone', False),
    ).to(device)
    if hasattr(model, "_init_weights"):
        model._init_weights()

    # ===== Optim / Sched / Loss =====
    criterion = CTCLoss(zero_infinity=True)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=getattr(args, 'weight_decay', 5e-3),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # ===== Resume =====
    start_epoch = 0
    best_cer = float('inf')
    best_epoch = -1
    patience_counter = 0

    resume_path = getattr(args, 'resume', None)
    if resume_path and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        # 선택 필드들: 없으면 무시
        if 'lr_scheduler_state_dict' in ck:
            try:
                scheduler.load_state_dict(ck['lr_scheduler_state_dict'])
            except Exception:
                pass
        if 'scaler_state_dict' in ck:
            try:
                scaler.load_state_dict(ck['scaler_state_dict'])
            except Exception:
                pass
        best_cer = ck.get('val_cer', float('inf'))
        start_epoch = int(ck.get('epoch', -1)) + 1
        print(f"▶ Resumed from {resume_path} at epoch {start_epoch}, best_cer={best_cer:.4f}")

    # ===== Logging =====
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    # ===== EarlyStopping/Warmup =====
    patience = getattr(args, 'early_stopping_patience', 12)
    warmup_epochs = getattr(args, 'warmup_epochs', 3)

    # ===== Train Loop =====
    for epoch in range(start_epoch, args.num_epochs):
        # --- 에폭 시작 시 train_loader 구성 (우선순위: 고정 비율 > 가중 샘플링 > 셔플) ---
        if getattr(args, 'use_fixed_ratio_batch', False) and len(getattr(args, 'target_label_set', set())) > 0:
            train_loader = build_fixed_ratio_loader(
                epoch=epoch,
                train_dataset=train_dataset,
                train_labels=args.train_labels,
                target_label_set=args.target_label_set,
                batch_size=args.batch_size,
                target_ratio_start=getattr(args, 'target_ratio_start', 0.3),
                target_ratio_end=getattr(args, 'target_ratio_end', 0.4),
                ratio_warmup_epochs=getattr(args, 'ratio_warmup_epochs', 10),
                target_repeat_max=getattr(args, 'target_repeat_max', 6),
                num_workers=args.num_workers,
            )
        elif getattr(args, 'sampler_weights', None) is not None:
            sig = inspect.signature(WeightedRandomSampler.__init__)
            kw = {}
            if 'generator' in sig.parameters:
                gen = torch.Generator()
                gen.manual_seed(1234 + epoch)  # 에폭별 재현성
                kw['generator'] = gen
            sampler = WeightedRandomSampler(
                weights=torch.tensor(args.sampler_weights, dtype=torch.double),
                num_samples=len(train_dataset),
                replacement=True,
                **kw,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=ctc_collate_fn,
                sampler=sampler,
                pin_memory=True, persistent_workers=(args.num_workers > 0)
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, collate_fn=ctc_collate_fn,
                pin_memory=True, persistent_workers=(args.num_workers > 0)
            )

        # --- Warmup LR ---
        if epoch < warmup_epochs:
            warm_ratio = float(epoch + 1) / float(max(1, warmup_epochs))
            for pg in optimizer.param_groups:
                pg["lr"] = args.learning_rate * warm_ratio

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
                output_lengths = compute_output_lengths_from_input_widths(
                    input_widths, getattr(model, 'cnn_backbone', 'basic')
                )
                loss = criterion(outputs.log_softmax(2), labels, output_lengths, label_lengths)

            # ✅ AMP + clip (unscale 먼저!)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # <-- 중요
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/batch', loss.item(), global_step)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step)

            # 샘플 CER 프린트(희소)
            if getattr(args, 'print_freq', 0) > 0 and batch_idx % args.print_freq == 0:
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

        # ===== Validation =====
        avg_val_loss, val_cer = evaluate_full_cer(
            model, val_loader, criterion, device, args.idx_to_char
        )
        print(f'Epoch {epoch+1} val loss: {avg_val_loss:.4f} | val CER: {val_cer:.4f}')
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('CER/val_epoch', val_cer, epoch)

        # 스케줄러(보수적으로 val loss 기준)
        scheduler.step(avg_val_loss)

        # ===== Checkpointing =====
        ckpt_last = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_loss': avg_val_loss,
            'val_cer': val_cer,
            # ✅ 백본 저장(평가/재개 시 길이계산 일치)
            'cnn_backbone': getattr(model, 'cnn_backbone', 'basic'),
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
