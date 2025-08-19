# run_train.py
import os
import csv
import json
import argparse
import random
import time
from types import SimpleNamespace
from typing import Mapping, Sequence, Union

import torch
from torch.utils.data import DataLoader

from src.train import train as train_fn, compute_input_widths_from_padded_images
from src.utils.data_utils import HandwritingDataset, ctc_collate_fn
from src.model.crnn import CRNN
from src.utils.metrics import character_error_rate


# -------------------------
# 데이터 로딩 & 전처 유틸
# -------------------------
def read_labels(csv_path: str, images_dir: str):
    image_paths = []
    labels = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_paths.append(os.path.join(images_dir, row['filename']))
            labels.append(row['label'])
    return image_paths, labels


def build_charset(labels):
    # 0은 CTC blank로 예약
    unique_chars = []
    seen = set()
    for text in labels:
        for ch in text:
            if ch not in seen:
                seen.add(ch)
                unique_chars.append(ch)

    idx_to_char = [''] + unique_chars
    char_to_idx = {ch: i for i, ch in enumerate(idx_to_char) if ch != ''}
    return char_to_idx, idx_to_char


def save_charset(save_dir: str, char_to_idx, idx_to_char):
    os.makedirs(save_dir, exist_ok=True)
    charset_path = os.path.join(save_dir, 'charset.json')
    with open(charset_path, 'w', encoding='utf-8') as f:
        json.dump({'idx_to_char': idx_to_char, 'char_to_idx': char_to_idx}, f, ensure_ascii=False, indent=2)
    return charset_path


def split_train_val_test(image_paths, labels, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    # 비율 검증
    assert 0.0 <= train_ratio <= 1.0 and 0.0 <= val_ratio <= 1.0 and 0.0 <= test_ratio <= 1.0
    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-6, 'train:val:test 비율 합이 1.0이어야 합니다.'

    n = len(image_paths)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # 나머지는 test로
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def pick(idxs):
        return [image_paths[i] for i in idxs], [labels[i] for i in idxs]

    tr_images, tr_labels = pick(train_idx)
    val_images, val_labels = pick(val_idx)
    test_images, test_labels = pick(test_idx)
    return tr_images, tr_labels, val_images, val_labels, test_images, test_labels


def _word_accuracy(refs, hyps):
    ok = sum(1 for r, h in zip(refs, hyps) if r == h)
    return ok / max(1, len(refs))


def _downsample_ratio_from_backbone(cnn_backbone: str) -> int:
    return 4 if (cnn_backbone or "basic") == "basic" else 16


@torch.no_grad()
def _compute_output_lengths(input_widths: torch.Tensor, backbone: str) -> torch.Tensor:
    ratio = _downsample_ratio_from_backbone(backbone)
    if ratio == 4:
        t = input_widths // 4 - 1
    else:
        t = input_widths // ratio
    return torch.clamp(t, min=1)


@torch.no_grad()
def _greedy_decode_with_lengths(
    outputs: torch.Tensor,
    idx_to_char: Union[Sequence[str], Mapping[int, str]],
    lengths: torch.Tensor,
):
    # idx -> char 접근 함수
    if isinstance(idx_to_char, Mapping):
        def _itc(i): return idx_to_char.get(i, "")
    else:
        def _itc(i): return idx_to_char[i] if 0 <= i < len(idx_to_char) else ""

    probs = outputs.detach().permute(1, 0, 2).cpu()
    if torch.is_tensor(lengths):
        lengths = lengths.detach().cpu().tolist()
    hyps = []
    for i, p in enumerate(probs):
        t_i = int(max(0, lengths[i])) if lengths is not None else p.shape[0]
        t_i = min(t_i, p.shape[0])
        seq = p[:t_i].argmax(dim=1).tolist()
        out, prev = [], -1
        for c in seq:
            if c != prev and c != 0:
                out.append(_itc(c))
            prev = c
        hyps.append(''.join(out))
    return hyps


@torch.no_grad()
def evaluate_test_set_from_lists(image_paths, labels, model_path: str, char_to_idx, idx_to_char, batch_size: int, num_workers: int):
    ds = HandwritingDataset(image_paths, labels, char_to_idx, transform=None)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=ctc_collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(model_path, map_location=device)

    # ✅ 백본을 ckpt에서 읽어 동일 구성으로 모델 생성
    backbone = ck.get('cnn_backbone', 'basic')
    model = CRNN(num_channels=1, num_classes=len(char_to_idx) + 1, cnn_backbone=backbone).to(device)

    state = ck['model_state_dict'] if isinstance(ck, dict) and 'model_state_dict' in ck else ck
    model.load_state_dict(state)
    model.eval()

    all_refs, all_hyps = [], []
    per_image_times_ms = []
    n_img = 0

    for batch in dl:
        bsz = batch['image'].size(0)
        images = batch['image'].to(device)

        t0 = time.time()
        outs = model(images)  # [T,B,C]
        t1 = time.time()

        input_widths = compute_input_widths_from_padded_images(batch['image'])
        output_lengths = _compute_output_lengths(input_widths, backbone)
        hyps = _greedy_decode_with_lengths(outs, idx_to_char, output_lengths)
        refs = batch['text']

        n = min(len(hyps), len(refs))
        all_hyps.extend(hyps[:n])
        all_refs.extend(refs[:n])
        n_img += n

        per_image_times_ms.extend([((t1 - t0) / max(1, bsz)) * 1000.0] * n)

    cer = character_error_rate(all_refs, all_hyps)
    wacc = _word_accuracy(all_refs, all_hyps)
    avg_ms = sum(per_image_times_ms) / len(per_image_times_ms) if per_image_times_ms else 0.0
    p95 = sorted(per_image_times_ms)[int(0.95 * (len(per_image_times_ms) - 1))] if per_image_times_ms else 0.0

    result = {
        'images': n_img,
        'cer': cer,
        'word_acc': wacc,
        'latency_ms_per_image_avg': avg_ms,
        'latency_ms_per_image_p95': p95,
        'cnn_backbone': backbone,
    }
    print('[Test Eval]', json.dumps(result, ensure_ascii=False, indent=2))
    out_json = os.path.join('models', 'eval_test.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'[Test Eval] Saved: {out_json}')
    return result


# -------------------------
# 메인: 데이터 분할/학습/평가
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='data/generated_dataset')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)  # Windows 안전값
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--clip_grad', type=float, default=5.0)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--early_stopping_patience', type=int, default=12)

    # 분할 비율
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--shuffle_seed', type=int, default=42)
    parser.add_argument('--overfit_n', type=int, default=0)

    # 타겟/제너럴 비율 제어
    parser.add_argument('--word_list_path', type=str, default='data/word_list.txt')
    parser.add_argument('--target_start_line', type=int, default=293, help='word_list.txt에서 타겟 시작 라인(1-base)')
    parser.add_argument('--target_end_line', type=int, default=302, help='word_list.txt에서 타겟 종료 라인(포함)')
    parser.add_argument('--target_ratio', type=float, default=0.4, help='(가중 샘플링용) 배치 기대 타겟 비율')

    # 두-스트림 고정 비율 배치 샘플러
    parser.add_argument('--use_fixed_ratio_batch', action='store_true', help='두-스트림 고정 비율 배치 샘플러 사용')
    parser.add_argument('--target_ratio_start', type=float, default=0.3, help='비율 커리큘럼 시작값')
    parser.add_argument('--target_ratio_end', type=float, default=0.4, help='비율 커리큘럼 종료값')
    parser.add_argument('--ratio_warmup_epochs', type=int, default=10, help='비율을 선형 증가시킬 에폭 수')
    parser.add_argument('--target_repeat_max', type=int, default=6, help='한 에폭 내 타겟 샘플 최대 반복 횟수 R_max')

    # 모델 옵션
    parser.add_argument('--cnn_backbone', type=str, default='basic')  # 'basic' or 'resnet18' 등
    parser.add_argument('--use_pretrained_backbone', action='store_true')

    args = parser.parse_args()

    csv_path = os.path.join(args.dataset_dir, 'labels.csv')
    images_dir = os.path.join(args.dataset_dir, 'images')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'labels.csv를 찾을 수 없습니다: {csv_path}')
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f'images 디렉터리를 찾을 수 없습니다: {images_dir}')

    image_paths, labels = read_labels(csv_path, images_dir)

    # 문자 집합 구성 및 저장
    char_to_idx, idx_to_char = build_charset(labels)
    save_charset(args.save_dir, char_to_idx, idx_to_char)

    # 학습/검증/테스트 분할 (기본 6:2:2)
    tr_images, tr_labels, val_images, val_labels, te_images, te_labels = split_train_val_test(
        image_paths,
        labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.shuffle_seed,
    )

    # 오버핏 테스트: 상위 N개 제한
    if args.overfit_n and args.overfit_n > 0:
        n = int(args.overfit_n)
        tr_images, tr_labels = tr_images[:n], tr_labels[:n]
        val_images, val_labels = val_images[:min(n, len(val_images))], val_labels[:min(n, len(val_labels))]

    # ===== 타겟 라벨 세트 구성 (word_list의 지정 라인 범위)
    target_label_set = set()
    try:
        with open(args.word_list_path, 'r', encoding='utf-8') as f:
            all_words = [line.strip() for line in f.readlines() if line.strip()]
        s = max(1, int(args.target_start_line)) - 1
        e = min(len(all_words), int(args.target_end_line))
        for w in all_words[s:e]:
            target_label_set.add(w)
    except Exception:
        target_label_set = set()

    # ===== 가중 샘플링용 weight 벡터 계산 (train 전용)
    t = sum(1 for lb in tr_labels if lb in target_label_set)
    g = len(tr_labels) - t
    if t > 0 and g > 0:
        p = max(1e-3, min(1.0 - 1e-3, float(args.target_ratio)))
        w_t = 1.0
        w_g = (t * (1.0 - p)) / (g * p)
        sampler_weights = [w_t if lb in target_label_set else max(1e-6, w_g) for lb in tr_labels]
    else:
        sampler_weights = None

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print(f'Train/Val/Test sizes -> train: {len(tr_images)}, val: {len(val_images)}, test: {len(te_images)}')

    # ===== 학습 실행 (train.py 라이브러리 함수 호출)
    train_args = SimpleNamespace(
        train_image_paths=tr_images,
        train_labels=tr_labels,
        val_image_paths=val_images,
        val_labels=val_labels,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        clip_grad=args.clip_grad,
        print_freq=args.print_freq,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        early_stopping_patience=args.early_stopping_patience,
        sampler_weights=sampler_weights,
        # 두-스트림용
        use_fixed_ratio_batch=args.use_fixed_ratio_batch,
        target_label_set=target_label_set,
        target_ratio_start=args.target_ratio_start,
        target_ratio_end=args.target_ratio_end,
        ratio_warmup_epochs=args.ratio_warmup_epochs,
        target_repeat_max=args.target_repeat_max,
        # 모델 옵션
        cnn_backbone=args.cnn_backbone,
        use_pretrained_backbone=args.use_pretrained_backbone,
        # 기타
        seed=1234,
        weight_decay=5e-3,
        warmup_epochs=3,
        resume=args.resume if args.resume else None,
    )
    train_fn(train_args)

    # ===== 학습 종료 후 Test 세트 자동 평가
    best_model_path = os.path.join(args.save_dir, 'best_model_by_cer.pth')
    last_model_path = os.path.join(args.save_dir, 'last.pth')
    model_path = best_model_path if os.path.exists(best_model_path) else last_model_path
    if os.path.exists(model_path):
        print(f'[Test Eval] Evaluating with model: {model_path}')
        evaluate_test_set_from_lists(
            image_paths=te_images,
            labels=te_labels,
            model_path=model_path,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        print('[Test Eval] 모델이 없어 평가를 건너뜁니다.')


if __name__ == '__main__':
    main()
