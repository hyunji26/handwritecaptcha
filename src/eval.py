import os
import csv
import json
import time
import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from src.utils.data_utils import HandwritingDataset, ctc_collate_fn
from src.model.crnn import CRNN
from src.utils.metrics import character_error_rate


def word_accuracy(refs, hyps):
    ok = sum(1 for r, h in zip(refs, hyps) if r == h)
    return ok / max(1, len(refs))


@torch.no_grad()
def greedy_decode_with_lengths(outputs: torch.Tensor, idx_to_char, lengths: torch.Tensor):
    # outputs: [T, B, C]
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


def compute_input_widths_from_padded_images(padded_images: torch.Tensor) -> torch.Tensor:
    # padded_images: [B, 1, H, W_max], 원본 이미지는 배경이 1.0, 패딩은 0.0이므로
    # 열 합이 0인 구간이 패딩 폭, 0보다 크면 유효 폭
    col_sum = padded_images.sum(dim=2).sum(dim=1)  # [B, W]
    widths = (col_sum > 0).sum(dim=1).to(torch.long)
    return widths


def compute_output_lengths_basic(input_widths: torch.Tensor) -> torch.Tensor:
    # basic 백본 기준: 대략 W//4 - 1
    t = input_widths // 4 - 1
    return torch.clamp(t, min=1)


def read_labels(csv_path: str, images_dir: str):
    image_paths, labels = [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_paths.append(os.path.join(images_dir, row['filename']))
            labels.append(row['label'])
    return image_paths, labels


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_dir', required=True)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--charset_path', required=True)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--save_json', default='')
    ap.add_argument('--csv_name', type=str, default='labels.csv', help='labels.csv / train.csv / val.csv / test.csv 중 선택')
    args = ap.parse_args()

    # charset 로드
    with open(args.charset_path, 'r', encoding='utf-8') as f:
        cs = json.load(f)
    idx_to_char = cs['idx_to_char']
    char_to_idx = cs['char_to_idx']

    # 데이터셋 로드
    csv_path = os.path.join(args.dataset_dir, args.csv_name)
    images_dir = os.path.join(args.dataset_dir, 'images')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'labels.csv를 찾을 수 없습니다: {csv_path}')
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f'images 디렉터리를 찾을 수 없습니다: {images_dir}')

    image_paths, labels = read_labels(csv_path, images_dir)
    ds = HandwritingDataset(image_paths, labels, char_to_idx, transform=None)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=ctc_collate_fn)

    # 모델 준비
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN(num_channels=1, num_classes=len(char_to_idx) + 1).to(device)
    ck = torch.load(args.model_path, map_location=device)
    state = ck['model_state_dict'] if isinstance(ck, dict) and 'model_state_dict' in ck else ck
    model.load_state_dict(state)
    model.eval()

    all_refs, all_hyps = [], []
    per_image_times_ms = []
    n_img = 0

    t0 = time.time()
    for batch in dl:
        bsz = batch['image'].size(0)
        images = batch['image'].to(device)

        t_batch0 = time.time()
        outs = model(images)  # [T, B, C]
        t_batch1 = time.time()

        # 길이 추정 및 디코딩
        input_widths = compute_input_widths_from_padded_images(batch['image'])
        output_lengths = compute_output_lengths_basic(input_widths)
        hyps = greedy_decode_with_lengths(outs, idx_to_char, output_lengths)
        refs = batch['text']

        n = min(len(hyps), len(refs))
        all_hyps.extend(hyps[:n])
        all_refs.extend(refs[:n])
        n_img += n

        # 배치 처리 지연을 per-image로 환산
        per_image_time_ms = ((t_batch1 - t_batch0) / max(1, bsz)) * 1000.0
        per_image_times_ms.extend([per_image_time_ms] * n)

    dt = time.time() - t0

    # 지표 집계
    cer = character_error_rate(all_refs, all_hyps)
    wacc = word_accuracy(all_refs, all_hyps)

    buckets = defaultdict(lambda: {'n': 0, 'ok': 0})
    for r, h in zip(all_refs, all_hyps):
        L = len(r)
        buckets[L]['n'] += 1
        buckets[L]['ok'] += int(r == h)
    per_len = {L: (v['ok'] / v['n'] if v['n'] else 0.0) for L, v in sorted(buckets.items())}

    # p95 계산
    if per_image_times_ms:
        times_sorted = sorted(per_image_times_ms)
        idx95 = int(0.95 * (len(times_sorted) - 1))
        p95 = times_sorted[idx95]
        avg_ms = sum(per_image_times_ms) / len(per_image_times_ms)
    else:
        avg_ms, p95 = 0.0, 0.0

    result = {
        'images': n_img,
        'cer': cer,
        'word_acc': wacc,
        'word_acc_by_length': per_len,
        'latency_ms_per_image_avg': avg_ms,
        'latency_ms_per_image_p95': p95,
        'total_time_sec': dt,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.save_json:
        with open(args.save_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()



