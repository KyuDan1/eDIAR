"""
Real-time (streaming) speaker diarization pipeline for the live meeting demo.

Audio comes in as 16 kHz samples (any chunking). Three causal stages run incrementally:
  LiveNemotron  - official Nemotron-3-Diarization streaming API (speaker cache + FIFO), one chunk at a time
  LiveResNet    - WeSpeaker ResNet34 features on short blocks (cumulative-mean normalisation, 0.16 s look-ahead),
                  causal pooled embeddings e0 (window ends at the frame) and centred ones e9 (available 9 frames = 0.72 s later)
  LiveDiarizer  - slot-conditioned official-e-means diarizer re-run on a bounded window every Nemotron chunk,
                  provisional labels + bounded retroactive revision, permanent speaker ids (overlap matching + centroid bank)
Nothing is written to disk. All state lives in a LiveSession object.
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from scipy.optimize import linear_sum_assignment

STEP = 0.08          # one frame = 80 ms
HOP = 160            # 10 ms mel hop


class LiveNemotron:
    """Feeds the official streaming API. push() returns a list of (n_frames x 8) probability arrays (80 ms frames)."""

    def __init__(self, model, processor, device, mode="low_latency"):
        self.m, self.p, self.dev = model, processor, device
        self.p.set_streaming_mode(mode)
        self.chunk_len, self.right = self.p._streaming_chunk_sizes
        self.reset()

    def reset(self):
        self.buf = np.zeros(0, np.float32); self.off = 0       # absolute sample index of buf[0]
        self.cursor = 0; self.first = True; self.cache = None; self.frames_out = 0

    @property
    def latency_s(self):                                          # input-buffer latency of the mode
        return (self.chunk_len + self.right) * STEP

    def _features(self, seg, first, last):
        """what Nemotron3DiarizationProcessor.__call__ does in streaming mode (re-implemented: the vendored processor file is
        written for another transformers version): centred windows only for the first chunk, trailing frames that reach past a
        non-final chunk are dropped, and the look-ahead frame count switches the model into streaming mode."""
        fe = self.p.feature_extractor
        inp = fe([seg], sampling_rate=16000, center=first, is_last_audio_chunk=last, return_tensors="pt")
        n = int(inp["attention_mask"].sum(-1).max())
        feats, mask = inp["input_features"][:, :n], inp["attention_mask"][:, :n]
        if not last and n != self.p.num_mel_frames_per_audio_chunk:
            raise ValueError(f"chunk has {n} mel frames, expected {self.p.num_mel_frames_per_audio_chunk}")
        kw = {"input_features": feats.to(self.dev), "attention_mask": mask.to(self.dev)}
        if not last: kw["num_lookahead_frames"] = self.right
        return kw

    def _run(self, seg, first, last):
        kw = self._features(seg, first, last)
        with torch.inference_mode():
            out = self.m(**kw, speaker_cache=self.cache, output_hidden_states=False)
        self.cache = out.speaker_cache
        p10 = out.logits[0].sigmoid().float().cpu().numpy()
        n = p10.shape[0] // 8
        return p10[: n * 8].reshape(n, 8, 8).mean(1)

    def push(self, x):
        self.buf = np.concatenate([self.buf, x]); res = []
        while True:
            start = 0 if self.first else self.p.audio_chunk_start(self.cursor)
            need = self.p.num_samples_first_audio_chunk if self.first else self.p.num_samples_per_audio_chunk
            if self.off + len(self.buf) < start + need: break
            seg = self.buf[start - self.off: start - self.off + need]
            res.append(self._run(seg, self.first, False)); self.frames_out += res[-1].shape[0]
            self.cursor += self.p.num_mel_frames_per_step; self.first = False
            keep = self.p.audio_chunk_start(self.cursor) - self.off
            if keep > 0: self.buf = self.buf[keep:]; self.off += keep
        return res

    def flush(self):
        if self.first and len(self.buf) < 1600: return []
        start = 0 if self.first else self.p.audio_chunk_start(self.cursor)
        seg = self.buf[max(0, start - self.off):]
        if len(seg) < 800: return []
        try:
            r = self._run(seg, self.first, True)
        except Exception:
            return []
        self.frames_out += r.shape[0]; return [r]


class LiveResNet:
    """Causal ResNet embeddings. push() returns (new e0 rows, new e9 rows) as numpy arrays; the e9 of frame f appears once
    frame f+9 exists (0.72 s later). Output frame i is aligned with Nemotron frame i."""

    BLOCK, LEFT, RIGHT, CMN_MIN, WIN, LOOK = 32, 160, 16, 20, 19, 9

    def __init__(self, engine):
        self.e = engine; self.dev = engine.device; self.reset()

    def reset(self):
        self.wav = np.zeros(0, np.float32); self.wav_off = 0; self.n_mel = 0; self.csum = np.zeros(80, np.float64)
        self.mel_norm = np.zeros((self.LEFT, 80), np.float32); self.mn_off = 0     # zero history before the stream (as in the offline code); mn_off = padded index of mel_norm[0]
        self.n_blocks = 0; self.frames = []                          # GPU vectors (2560,) per 80 ms frame (last ones only)
        self.f_total = 0; self.first_frame = None; self.n_e0 = 0; self.n_e9 = 0

    def _mel_new(self):
        total = self.wav_off + len(self.wav)
        have = 0 if total < 512 else (total - 512) // HOP + 1     # librosa frames need the full n_fft=512 samples (window 400 centred inside)
        if have <= self.n_mel: return
        a, b = self.n_mel, have
        s0 = a * HOP - 1 if a > 0 else 0
        seg = self.wav[s0 - self.wav_off: (b - 1) * HOP + 512 - self.wav_off]
        pre = (seg[1:] - 0.97 * seg[:-1]) if a > 0 else np.append(seg[0], seg[1:] - 0.97 * seg[:-1])
        stft = librosa.stft(pre * 32768.0, n_fft=512, hop_length=HOP, win_length=400, window="hann", center=False)
        mel = np.log(np.maximum(self.e.mel_basis_80 @ (np.abs(stft) ** 2), 1e-10)).T.astype(np.float32)
        cs = self.csum + np.cumsum(mel.astype(np.float64), axis=0)
        n = np.maximum(np.arange(a + 1, b + 1), self.CMN_MIN)[:, None]
        self.csum = cs[-1]; self.mel_norm = np.concatenate([self.mel_norm, (mel - cs / n).astype(np.float32)]); self.n_mel = b
        keep = self.n_mel * HOP - 1 - self.wav_off
        if keep > 0: self.wav = self.wav[keep:]; self.wav_off += keep

    def _blocks(self, flush):
        out = []
        while True:
            j = self.n_blocks; end = j * self.BLOCK + self.BLOCK + self.RIGHT       # mel frames needed (in unpadded index)
            if self.n_mel < end and not flush: break
            if flush and j * self.BLOCK >= self.n_mel: break
            i0 = j * self.BLOCK - self.mn_off; win = self.mel_norm[i0: i0 + self.LEFT + self.BLOCK + self.RIGHT]
            if len(win) < self.LEFT + self.BLOCK + self.RIGHT:
                win = np.concatenate([win, np.zeros((self.LEFT + self.BLOCK + self.RIGHT - len(win), 80), np.float32)])
            out.append(win); self.n_blocks += 1
        if not out: return None
        drop = self.n_blocks * self.BLOCK - self.mn_off
        if drop > 0: self.mel_norm = self.mel_norm[drop:]; self.mn_off += drop
        x = torch.from_numpy(np.stack(out)).to(self.dev).permute(0, 2, 1).unsqueeze(1)
        with torch.inference_mode():
            r = self.e.resnet; o = F.relu(r.bn1(r.conv1(x))); o = r.layer4(r.layer3(r.layer2(r.layer1(o))))
            o = o.reshape(o.shape[0], 2560, -1)[:, :, self.LEFT // 8: self.LEFT // 8 + self.BLOCK // 8]
            return o.permute(0, 2, 1).reshape(-1, 2560)               # (frames, 2560)

    def _pool(self, rows_f, look):
        """stats pooling for absolute frame indices rows_f over [f-(WIN-1-look), f+look], replicate padding at the edges"""
        base = self.f_total - len(self.frames)                         # absolute index of self.frames[0]
        last = self.f_total - 1; idx = []
        for f in rows_f:
            w = np.clip(np.arange(f - (self.WIN - 1 - look), f + look + 1), 0, last)
            idx.append(w)
        idx = np.array(idx)
        pool = torch.stack([self.first_frame] + self.frames)           # index 0 -> absolute frame 0 only if base == 0; handled below
        pos = np.clip(np.where(idx == 0, 0, idx - base + 1), 0, len(pool) - 1)   # frame 0 stored separately (replicate padding); clip = never index out of range
        W = pool[torch.from_numpy(pos).to(self.dev)]                   # (n, WIN, 2560)
        stats = torch.cat([W.mean(1), torch.sqrt(torch.var(W, dim=1) + 1e-7)], dim=-1)
        with torch.inference_mode():
            return F.normalize(self.e.resnet.seg_1(stats), p=2, dim=-1).cpu().numpy()

    def push(self, x, flush=False):
        self.wav = np.concatenate([self.wav, x]); self._mel_new()
        fm = self._blocks(flush)
        if fm is None: return np.zeros((0, 256), np.float32), np.zeros((0, 256), np.float32)
        for v in fm:
            if self.first_frame is None: self.first_frame = v
            self.frames.append(v); self.f_total += 1
        new0 = list(range(self.n_e0, self.f_total)); e0 = self._pool(new0, 0); self.n_e0 = self.f_total
        top = self.f_total - 1 if flush else self.f_total - 1 - self.LOOK        # e9[f] needs frame f+9 (or the stream end)
        new9 = list(range(self.n_e9, top + 1)); e9 = self._pool(new9, self.LOOK) if new9 else np.zeros((0, 256), np.float32)
        self.n_e9 = max(self.n_e9, top + 1)
        if len(self.frames) > 80: self.frames = self.frames[-80:]       # trim only AFTER pooling: a coalesced backlog adds many frames at once
        return e0, e9

    def flush(self):
        return self.push(np.zeros(0, np.float32), flush=True)


class Projector:
    """Stable 2-D view of the embeddings: PCA on recent active frames, re-fitted every few seconds, rotation-aligned to the previous basis."""

    def __init__(self): self.mean = None; self.comp = None; self.mu = None; self.sd = None

    def fit(self, X):
        if len(X) < 12: return False
        m = X.mean(0); U, S, Vt = np.linalg.svd(X - m, full_matrices=False); comp = Vt[:2]
        if self.comp is not None:                                      # keep orientation: rotate the new axes onto the old ones (Procrustes)
            u, _, vt = np.linalg.svd(self.comp @ comp.T); comp = (u @ vt) @ comp
        Z = (X - m) @ comp.T; mu = Z.mean(0); sd = max(float(Z.std(0).max()) * 3.2, 1e-6)
        self.mean, self.comp, self.mu, self.sd = m, comp, mu, sd; return True

    def __call__(self, X):
        if self.comp is None: return np.zeros((len(X), 2), np.float32)
        return np.round(((X - self.mean) @ self.comp.T - self.mu) / self.sd, 3)


class LiveDiarizer:
    def __init__(self, diarizer_cls, window_s=90.0, horizon_s=10.0, refit_every=1, slot_cos=0.4, use_centred=True):
        self.cls, self.W, self.H = diarizer_cls, int(window_s / STEP), int(horizon_s / STEP); self.use_centred = use_centred
        self.refit_every, self.slot_cos = refit_every, slot_cos
        self.reset()

    def reset(self):
        self.probs, self.e0, self.e9 = [], [], []
        self.cur, self.sent = [], 0          # cur[f] = list of permanent ids (revisable until frozen)
        self.next_gid = 0; self.bank = {}; self.slot_ids = {}   # gid -> unit centroid; slot -> registered gids
        self.n_refit = 0; self.proj = Projector(); self.last_proj_fit = -10 ** 9; self.gtime = {}
        self.dia = self.cls(stride_ms=80, cos_merge_cross=2.0); self.dia.force_saturated = False   # no cross-slot merging (no effect in the offline ablation, harmful for id stability)
        self.refit_ms = 0.0; self.skipped = 0; self.events = []

    def tau(self): return min(len(self.probs), len(self.e0))

    def embedding(self, lo, hi):
        """X[f] = centred embedding if it exists (the retroactive correction), else the causal one"""
        n9 = len(self.e9) if self.use_centred else 0          # use_centred=False: causal embeddings only (ablation)
        return np.stack([self.e9[f] if f < n9 else self.e0[f] for f in range(lo, hi)]) if hi > lo else np.zeros((0, 256), np.float32)

    def refit(self):
        """re-cluster the bounded window; returns (changed: {f: labels}, tau)"""
        tau = self.tau()
        if tau < 20: return {}, tau
        lo = max(0, tau - self.W); t0 = time.perf_counter()
        X = self.embedding(lo, tau); P = np.asarray(self.probs[lo:tau], np.float32)
        lab, _, _, _ = self.dia.fit_predict(X, P)
        # local label -> permanent id, ANCHORED TO THE SLOT: Nemotron's speaker cache keeps a slot bound to one speaker over time, so a slot
        # that is not split keeps its id; the sub-clusters of a split slot are matched to that slot's registered ids by centroid cosine.
        cents, tags = self.dia.last_cents, self.dia.last_tags
        locs = sorted({l for ls in lab for l in ls}); mapping = {}
        for s_ in sorted({int(tags[l]) for l in locs}):
            ls_ = [l for l in locs if int(tags[l]) == s_]; reg = self.slot_ids.setdefault(s_, [])
            C = np.stack([cents[l] / (np.linalg.norm(cents[l]) + 1e-9) for l in ls_])
            if reg:
                Mx = C @ np.stack([self.bank[g] for g in reg]).T; used_loc = set()
                for r, c in zip(*linear_sum_assignment(-Mx)):
                    if Mx[r, c] >= self.slot_cos: mapping[ls_[r]] = reg[c]; used_loc.add(r)
                if len(ls_) == 1 and not used_loc: mapping[ls_[0]] = reg[int(np.argmax(Mx[0]))]      # an unsplit slot is the same speaker as before
                for r, l in enumerate(ls_):
                    if l not in mapping: g = self.next_gid; self.next_gid += 1; reg.append(g); mapping[l] = g; self.bank[g] = C[r]; self.events.append((tau, 'new', s_))
            else:
                for r, l in enumerate(ls_): g = self.next_gid; self.next_gid += 1; reg.append(g); mapping[l] = g; self.bank[g] = C[r]; self.events.append((tau, 'new', s_))
            for r, l in enumerate(ls_):                                     # update the registered centroid
                g = mapping[l]; self.bank[g] = 0.9 * self.bank[g] + 0.1 * C[r]; self.bank[g] /= np.linalg.norm(self.bank[g]) + 1e-9
        self.dia.force_saturated = self.dia.force_saturated or self.dia.last_saturated        # once slots have been exhausted, keep splitting enabled
        frozen = max(lo, tau - self.H); changed = {}
        while len(self.cur) < tau: self.cur.append([])
        for f in range(frozen, tau):
            new = sorted({mapping[l] for l in lab[f - lo]})
            if f >= self.sent or new != self.cur[f]: changed[f] = new
            self.cur[f] = new
        self.refit_ms = 1000 * (time.perf_counter() - t0); self.n_refit += 1
        return changed, tau


class LiveSession:
    """audio in -> JSON-able update messages out. Single-threaded; call feed() from one worker thread."""

    def __init__(self, nem, res, diarizer_cls, mode="low_latency", **ld_kw):
        self.nem, self.res = nem, res
        self.nem.reset(); self.res.reset(); self.nem.p.set_streaming_mode(mode); self.nem.chunk_len, self.nem.right = self.nem.p._streaming_chunk_sizes
        self.ld = LiveDiarizer(diarizer_cls, **ld_kw); self.samples = 0; self.t0 = time.perf_counter(); self.last_reproj = -1e9
        self.pend_probs = []; self.msgs_sent = 0

    def feed(self, pcm_i16: bytes | None, flush=False):
        t_in = time.perf_counter()
        if pcm_i16:
            x = np.frombuffer(pcm_i16, dtype="<i2").astype(np.float32) / 32768.0; self.samples += len(x)
            for p in self.nem.push(x): self.ld.probs.extend(p)
            e0, e9 = self.res.push(x)
        else:
            for p in self.nem.flush(): self.ld.probs.extend(p)
            e0, e9 = self.res.flush()
        self.ld.e0.extend(e0); self.ld.e9.extend(e9)
        tau = self.ld.tau()
        if tau - self.ld.sent < 1 and not flush: return []
        if self.ld.n_refit and (tau - self.ld.sent) < self.nem.chunk_len and not flush: return []      # refit once per Nemotron chunk
        changed, tau = self.ld.refit()
        if tau < 20 or len(self.ld.cur) < tau: return []
        ld = self.ld; new_frames = list(range(ld.sent, tau))
        # projection (refit every ~3 s of stream, aligned to the previous basis)
        reproj = None
        act = [f for f in range(max(0, tau - 800), tau) if ld.cur[f]]
        if tau - ld.last_proj_fit >= 38 and len(act) >= 12:
            if ld.proj.fit(ld.embedding(act[0], tau)[[f - act[0] for f in act]]):
                ld.last_proj_fit = tau; lo = max(0, tau - 1500); reproj = [[f, *map(float, xy)] for f, xy in zip(range(lo, tau), ld.proj(ld.embedding(lo, tau)))]
        xy_new = ld.proj(ld.embedding(ld.sent, tau)) if new_frames else []
        new = [[f, float(xy_new[i][0]), float(xy_new[i][1]), ld.cur[f], round(float(ld.probs[f].max()), 2)] for i, f in enumerate(new_frames)]
        rev = [[f, l] for f, l in changed.items() if f < ld.sent]
        cent = {int(g): [float(v) for v in ld.proj(c[None])[0]] for g, c in ld.bank.items()} if ld.proj.comp is not None else {}
        ld.sent = tau
        audio_end = self.samples / 16000.0
        msg = {"type": "update", "tau": tau, "frame_end_t": round(tau * STEP, 3), "audio_t": round(audio_end, 3), "new": new, "rev": rev, "cent": cent,
               "nspk": len(ld.bank), "refit_ms": round(ld.refit_ms, 1), "proc_ms": round(1000 * (time.perf_counter() - t_in), 1)}
        if reproj is not None: msg["reproj"] = reproj
        self.msgs_sent += 1
        return [msg]
