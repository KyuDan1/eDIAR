import os
import io
import sys
import time
import json
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from huggingface_hub import hf_hub_download
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
import jiwer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nemotron_ref.configuration_nemotron3_diarization import Nemotron3DiarizationConfig
from nemotron_ref.modeling_nemotron3_diarization import Nemotron3DiarizationForAudioFrameClassification
from nemotron_ref.feature_extraction_nemotron_asr_streaming import NemotronAsrStreamingFeatureExtractor
from wespeaker_resnet import ResNet34


class DualStreamSpeechLLMAcousticEngine:
    """
    Extracts 80ms / 160ms (or arbitrary stride) frame-level acoustic-speaker representations,
    fixed-head Sortformer-4spk / Nemotron-8spk predictions, and Speech-LLM word-level transcripts.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        token = os.environ.get("HF_TOKEN")

        cfg = Nemotron3DiarizationConfig.from_pretrained("nvidia/Nemotron-3-Diarization", token=token)
        self.fe = NemotronAsrStreamingFeatureExtractor.from_pretrained("nvidia/Nemotron-3-Diarization", token=token)
        self.nemotron = (
            Nemotron3DiarizationForAudioFrameClassification.from_pretrained(
                "nvidia/Nemotron-3-Diarization", config=cfg, token=token
            )
            .to(device)
            .eval()
        )

        p_model = hf_hub_download("Wespeaker/wespeaker-voxceleb-resnet34-LM", "avg_model", token=token)
        sd = {
            k: v
            for k, v in torch.load(p_model, map_location="cpu", weights_only=True).items()
            if not k.startswith("projection.")
        }
        self.resnet = ResNet34(feat_dim=80, embed_dim=256, pooling_func="TSTP", two_emb_layer=False)
        self.resnet.load_state_dict(sd, strict=True)
        self.resnet = self.resnet.to(device).eval()

        self.mel_basis_80 = librosa.filters.mel(sr=16000, n_fft=512, n_mels=80, fmin=20.0, fmax=8000.0, htk=True)

    def compute_kaldi_fbank(self, wav_np: np.ndarray) -> np.ndarray:
        wav_pre = np.append(wav_np[0], wav_np[1:] - 0.97 * wav_np[:-1])
        stft = librosa.stft(wav_pre * 32768.0, n_fft=512, hop_length=160, win_length=400, window="hann", center=False)
        power = np.abs(stft) ** 2
        mel = np.log(np.maximum(self.mel_basis_80 @ power, 1e-10)).T
        return mel.astype(np.float32)

    def extract_base_80ms(self, wav_np: np.ndarray, win_80ms: int = 19):
        """
        Extracts base 80ms ResNet34-LM fmap & embeddings + 10ms Nemotron-3 probabilities once per recording.
        From this base representation, any frame stride (80ms, 160ms, 320ms, etc.) can be derived cleanly.
        """
        t0 = time.perf_counter()
        mel = self.compute_kaldi_fbank(wav_np)
        T_mel = mel.shape[0]
        chunk_mel = 4800
        ctx_mel = 160
        fmap_list = []

        with torch.inference_mode():
            for start in range(0, T_mel, chunk_mel):
                end = min(T_mel, start + chunk_mel)
                s_pad = max(0, start - ctx_mel)
                e_pad = min(T_mel, end + ctx_mel)
                m_chunk = torch.from_numpy(mel[s_pad:e_pad]).to(self.device)
                m_chunk = m_chunk - m_chunk.mean(dim=0, keepdim=True)
                x = m_chunk.unsqueeze(0).permute(0, 2, 1).unsqueeze(1)
                out = F.relu(self.resnet.bn1(self.resnet.conv1(x)))
                out = self.resnet.layer4(self.resnet.layer3(self.resnet.layer2(self.resnet.layer1(out))))
                out_flat = out.squeeze(0).reshape(2560, -1)
                s_80 = (start - s_pad) // 8
                e_80 = s_80 + (end - start) // 8
                fmap_list.append(out_flat[:, s_80:e_80].cpu())

            fmap_all = torch.cat(fmap_list, dim=1)  # (2560, T_80ms)

            inputs = self.fe(wav_np, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            nout = self.nemotron(**inputs, output_hidden_states=False)
            probs_10ms = nout.logits[0].sigmoid().cpu().numpy()  # (T_10ms, 8)

        enc_time = time.perf_counter() - t0
        return fmap_all, probs_10ms, enc_time

    def extract_causal_fmap(self, wav_np: np.ndarray, block_mel: int = 32, right_mel: int = 16, left_mel: int = 160, batch: int = 96):
        """
        Streaming-compatible ResNet34 feature map (80 ms frames, 2560 channels).

        Differences from `extract_base_80ms`, which is not causal: (i) mel features are normalised with a
        *cumulative* (causal) mean instead of the mean over a ~50 s chunk; (ii) the ResNet runs on short blocks of
        `block_mel` mel frames (10 ms each) with `left_mel` frames of history and only `right_mel` frames of
        look-ahead, so the feature of a frame depends on at most `right_mel + block_mel` mel frames (0.16-0.48 s
        with the defaults) of the future. Windows are batched; zero padding is used outside the stream.
        """
        mel = self.compute_kaldi_fbank(wav_np)  # (T, 80)
        T = mel.shape[0]
        csum = np.cumsum(mel, axis=0)
        n = np.maximum(np.arange(1, T + 1), 20)[:, None]
        mel = (mel - csum / n).astype(np.float32)
        assert block_mel % 8 == 0 and left_mel % 8 == 0 and right_mel % 8 == 0
        pad = np.zeros((left_mel, 80), np.float32), np.zeros((right_mel + block_mel, 80), np.float32)
        mp = np.concatenate([pad[0], mel, pad[1]], axis=0)
        starts = list(range(0, T, block_mel))
        win = left_mel + block_mel + right_mel
        f0, fb = left_mel // 8, block_mel // 8
        outs = []
        with torch.inference_mode():
            for b0 in range(0, len(starts), batch):
                idx = starts[b0:b0 + batch]
                x = torch.from_numpy(np.stack([mp[s:s + win] for s in idx])).to(self.device)  # (B, win, 80)
                x = x.permute(0, 2, 1).unsqueeze(1)
                out = F.relu(self.resnet.bn1(self.resnet.conv1(x)))
                out = self.resnet.layer4(self.resnet.layer3(self.resnet.layer2(self.resnet.layer1(out))))
                out = out.reshape(out.shape[0], 2560, -1)  # (B, 2560, win/8)
                outs.append(out[:, :, f0:f0 + fb].permute(0, 2, 1).reshape(-1, 2560).cpu())
        n_frames = T // 8
        return torch.cat(outs, dim=0)[:n_frames].T.contiguous()  # (2560, n_frames)

    def pool_causal(self, fmap_all: torch.Tensor, win_ms: int = 1520, look_frames: int = 0, stride_ms: int = 80):
        """Statistics pooling over a window that ends `look_frames` (80 ms) after the current frame
        (look_frames = 0: fully causal; look_frames = win/2: the original centred window)."""
        stride_80 = max(1, int(round(stride_ms / 80.0)))
        win_80 = max(2, int(round(win_ms / 80.0)))
        with torch.inference_mode():
            f = fmap_all.to(self.device)
            pad_r = look_frames
            pad_l = win_80 - 1 - look_frames
            fp = F.pad(f.unsqueeze(0), (pad_l, pad_r), mode="replicate").squeeze(0)
            wins = fp.unfold(1, win_80, stride_80)
            mean = wins.mean(dim=-1)
            std = torch.sqrt(torch.var(wins, dim=-1) + 1e-7)
            stats = torch.cat([mean, std], dim=0).T
            return F.normalize(self.resnet.seg_1(stats), p=2, dim=-1).cpu().numpy()

    def pool_at_stride(self, fmap_all: torch.Tensor, probs_10ms: np.ndarray, stride_ms: int = 80, win_ms: int = 1520):
        """
        Pools frame-level speaker embeddings and activity probabilities at `stride_ms` (e.g. 80ms, 160ms, 320ms).
        """
        # Base fmap_all has 80ms resolution (1 frame = 80ms)
        stride_80 = max(1, int(round(stride_ms / 80.0)))
        win_80 = max(3, int(round(win_ms / 80.0)))
        if win_80 % 2 == 0:
            win_80 += 1

        with torch.inference_mode():
            fmap_gpu = fmap_all.to(self.device)
            pad_l = win_80 // 2
            pad_r = win_80 - 1 - pad_l
            fmap_pad = F.pad(fmap_gpu.unsqueeze(0), (pad_l, pad_r), mode="replicate").squeeze(0)
            wins = fmap_pad.unfold(1, win_80, stride_80)
            mean = wins.mean(dim=-1)
            std = torch.sqrt(torch.var(wins, dim=-1) + 1e-7)
            stats = torch.cat([mean, std], dim=0).T
            spk_embs = F.normalize(self.resnet.seg_1(stats), p=2, dim=-1).cpu().numpy()

        step_10 = int(round(stride_ms / 10.0))
        T_s = min(spk_embs.shape[0], probs_10ms.shape[0] // step_10)
        probs_s = probs_10ms[: T_s * step_10].reshape(T_s, step_10, 8).mean(axis=1)
        spk_embs = spk_embs[:T_s]
        return spk_embs, probs_s


def frames_to_annotation(frame_spks: list[list[int]], stride_sec: float = 0.08, collar_support: float = 0.16) -> Annotation:
    hyp = Annotation()
    for t, labs in enumerate(frame_spks):
        t0 = t * stride_sec
        t1 = (t + 1) * stride_sec
        for lab in labs:
            hyp[Segment(t0, t1)] = f"spk_{lab}"
    if collar_support > 0:
        hyp = hyp.support(collar=collar_support)
    return hyp


def fixed_head_to_frames(probs_s: np.ndarray, max_spks: int = 4, threshold: float = 0.50) -> tuple[list[list[int]], int]:
    p_cap = probs_s[:, :max_spks]
    frame_spks = []
    used = set()
    for t in range(len(p_cap)):
        active = np.where(p_cap[t] > threshold)[0].tolist()
        frame_spks.append(active)
        used.update(active)
    return frame_spks, len(used)


def score_diarization(ref: Annotation, hyp: Annotation, collar: float = 0.25) -> dict:
    metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
    comp = metric(ref, hyp, detailed=True)
    total = max(1e-8, comp["total"])
    gt_k = len(ref.labels())
    pred_k = len(hyp.labels())
    return {
        "DER": float(comp["diarization error rate"] * 100.0),
        "Conf": float(comp["confusion"] / total * 100.0),
        "Miss": float(comp["missed detection"] / total * 100.0),
        "FA": float(comp["false alarm"] / total * 100.0),
        "gt_k": int(gt_k),
        "pred_k": int(pred_k),
        "SCA": 100.0 if gt_k == pred_k else 0.0,
        "MAE": float(abs(gt_k - pred_k)),
    }


def evaluate_word_speaker_attribution(
    words: list[str],
    word_starts: list[float],
    word_ends: list[float],
    ref_word_speakers: list[str],
    frame_spks: list[list[int]],
    stride_sec: float = 0.08,
) -> dict:
    """
    Aligns every decoded word interval [w_start, w_end] with the 80ms / 160ms frame-level speaker stream
    and computes Word Diarization Error Rate (WDER) and Concatenated Minimum-Permutation WER (cpWER).
    """
    T = len(frame_spks)
    if len(words) == 0 or T == 0:
        return {"WDER": 0.0, "cpWER": 0.0}

    # Build nearest active speaker fallback for frames
    nearest_spk = np.zeros(T, dtype=int)
    last_s = 0
    for t in range(T):
        if frame_spks[t]:
            last_s = frame_spks[t][0]
        nearest_spk[t] = last_s
    last_s = 0
    for t in range(T - 1, -1, -1):
        if frame_spks[t]:
            last_s = frame_spks[t][0]
        elif nearest_spk[t] == 0 and last_s != 0:
            nearest_spk[t] = last_s

    pred_word_spks = []
    for ws, we in zip(word_starts, word_ends):
        i0 = max(0, min(T - 1, int(np.floor(ws / stride_sec))))
        i1 = max(i0 + 1, min(T, int(np.ceil(we / stride_sec))))
        votes = {}
        for t in range(i0, i1):
            for spk in frame_spks[t]:
                votes[spk] = votes.get(spk, 0) + 1
        if votes:
            best_spk = max(votes.items(), key=lambda kv: kv[1])[0]
        else:
            best_spk = int(nearest_spk[(i0 + i1) // 2])
        pred_word_spks.append(f"pred_{best_spk}")

    ref_uniq = sorted(list(set(ref_word_speakers)))
    pred_uniq = sorted(list(set(pred_word_spks)))
    r_map = {s: i for i, s in enumerate(ref_uniq)}
    p_map = {s: i for i, s in enumerate(pred_uniq)}

    # Confusion matrix for optimal Hungarian speaker permutation
    cost = np.zeros((len(ref_uniq), len(pred_uniq)), dtype=np.int64)
    for rw, pw in zip(ref_word_speakers, pred_word_spks):
        cost[r_map[rw], p_map[pw]] -= 1

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_pairs = set(zip(row_ind, col_ind))

    correct_words = 0
    for rw, pw in zip(ref_word_speakers, pred_word_spks):
        if (r_map[rw], p_map[pw]) in matched_pairs:
            correct_words += 1
    wder = (1.0 - correct_words / max(1, len(words))) * 100.0

    # Compute cpWER across matched and unmatched speaker streams
    p_to_r = {col: row for row, col in zip(row_ind, col_ind)}
    ref_streams = {i: [] for i in range(max(len(ref_uniq), len(pred_uniq)))}
    hyp_streams = {i: [] for i in range(max(len(ref_uniq), len(pred_uniq)))}

    for w, rw in zip(words, ref_word_speakers):
        w_clean = w.lower().strip(".,!?;:\"'")
        if w_clean:
            ref_streams[r_map[rw]].append(w_clean)

    unmatched_slot = len(ref_uniq)
    unmatched_map = {}
    for w, pw in zip(words, pred_word_spks):
        w_clean = w.lower().strip(".,!?;:\"'")
        if not w_clean:
            continue
        p_idx = p_map[pw]
        if p_idx in p_to_r:
            target_slot = p_to_r[p_idx]
        else:
            if p_idx not in unmatched_map:
                unmatched_map[p_idx] = min(len(hyp_streams) - 1, unmatched_slot)
                unmatched_slot += 1
            target_slot = unmatched_map[p_idx]
        hyp_streams[target_slot].append(w_clean)

    total_ref_words = 0
    total_edit_errors = 0
    for slot_idx in ref_streams:
        r_str = " ".join(ref_streams[slot_idx])
        h_str = " ".join(hyp_streams[slot_idx])
        r_len = len(ref_streams[slot_idx])
        h_len = len(hyp_streams[slot_idx])
        total_ref_words += r_len
        if r_len == 0:
            total_edit_errors += h_len
        elif h_len == 0:
            total_edit_errors += r_len
        else:
            out_w = jiwer.process_words(r_str, h_str)
            total_edit_errors += out_w.substitutions + out_w.deletions + out_w.insertions

    cpwer = (total_edit_errors / max(1, total_ref_words)) * 100.0
    return {"WDER": float(wder), "cpWER": float(cpwer)}
