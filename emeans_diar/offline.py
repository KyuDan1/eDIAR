"""Non-streaming (whole-recording) path: the same slot head and the same slot-conditioned e-means, but the clustering sees every
embedding of the session at once and the encoders run over the full recording (no causal restrictions, no revision horizon)."""
import numpy as np
import torch
import torch.nn.functional as F
import librosa


def _fbank(engine, wav):
    wp = np.append(wav[0], wav[1:] - 0.97 * wav[:-1])
    p = np.abs(librosa.stft(wp * 32768.0, n_fft=512, hop_length=160, win_length=400, window="hann", center=False)) ** 2
    return np.log(np.maximum(engine.mel_basis_80 @ p, 1e-10)).T.astype(np.float32)


def extract(engine, wav, win_ms=1520):
    """Returns (spk_embs [T x 256, one per 80 ms], probs [T x 8] slot posteriors per 80 ms)."""
    dev, r = engine.device, engine.resnet
    mel = _fbank(engine, wav); T = mel.shape[0]; chunk, ctx = 4800, 160; fm = []
    with torch.inference_mode():
        for s in range(0, T, chunk):
            e = min(T, s + chunk); sp, ep = max(0, s - ctx), min(T, e + ctx)
            m = torch.from_numpy(mel[sp:ep]).to(dev); m = m - m.mean(0, keepdim=True)
            o = r.layer4(r.layer3(r.layer2(r.layer1(F.relu(r.bn1(r.conv1(m.unsqueeze(0).permute(0, 2, 1).unsqueeze(1))))))))
            o = o.squeeze(0).reshape(2560, -1); a = (s - sp) // 8; fm.append(o[:, a:a + (e - s) // 8].cpu())
        f = torch.cat(fm, 1).to(dev)
        w = int(round(win_ms / 80.0)); w += (w % 2 == 0); pl = w // 2
        win = F.pad(f.unsqueeze(0), (pl, w - 1 - pl), mode="replicate").squeeze(0).unfold(1, w, 1)
        stats = torch.cat([win.mean(-1), torch.sqrt(win.var(-1) + 1e-7)], 0).T
        embs = F.normalize(r.seg_1(stats), p=2, dim=-1).cpu().numpy()
        inp = engine.fe(wav, sampling_rate=16000, return_tensors="pt")
        inp = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in inp.items()}
        p10 = engine.nemotron(**inp, output_hidden_states=False).logits[0].sigmoid().cpu().numpy()
    n = min(len(embs), len(p10) // 8)
    return embs[:n], p10[:n * 8].reshape(n, 8, 8).mean(1)
