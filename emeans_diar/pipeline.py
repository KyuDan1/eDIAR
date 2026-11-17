"""Training-free open-speaker diarization on top of Nemotron-3 Diarization (slot head) and WeSpeaker ResNet34-LM (embeddings).

No weights are stored in this repository. The two upstream models are downloaded from their own Hugging Face repositories:
  nvidia/Nemotron-3-Diarization                      (license: openmdw-1.1)
  Wespeaker/wespeaker-voxceleb-resnet34-LM           (see its model card)
"""
import os
from functools import partial
import numpy as np
import torch
import librosa
from huggingface_hub import hf_hub_download

from .nemotron_ref.configuration_nemotron3_diarization import Nemotron3DiarizationConfig
from .nemotron_ref.modeling_nemotron3_diarization import Nemotron3DiarizationForAudioFrameClassification
from .nemotron_ref.feature_extraction_nemotron_asr_streaming import NemotronAsrStreamingFeatureExtractor
from .nemotron_ref.processing_nemotron3_diarization import Nemotron3DiarizationProcessor
from .wespeaker_resnet import ResNet34
from .clustering import OfficialEMeansSlotDiarizer
from .streaming import LiveNemotron, LiveResNet, LiveSession, STEP

NEMOTRON = "nvidia/Nemotron-3-Diarization"
WESPEAKER = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
MODES = ("low_latency", "very_low_latency")


class _Engine:
    """The pieces LiveNemotron / LiveResNet need (same attribute names as the research code)."""

    def __init__(self, device, token=None):
        self.device = device
        cfg = Nemotron3DiarizationConfig.from_pretrained(NEMOTRON, token=token)
        self.fe = NemotronAsrStreamingFeatureExtractor.from_pretrained(NEMOTRON, token=token)
        self.nemotron = Nemotron3DiarizationForAudioFrameClassification.from_pretrained(NEMOTRON, config=cfg, token=token).to(device).eval()
        sd = {k: v for k, v in torch.load(hf_hub_download(WESPEAKER, "avg_model", token=token), map_location="cpu", weights_only=True).items() if not k.startswith("projection.")}
        self.resnet = ResNet34(feat_dim=80, embed_dim=256, pooling_func="TSTP", two_emb_layer=False)
        self.resnet.load_state_dict(sd, strict=True); self.resnet = self.resnet.to(device).eval()
        self.mel_basis_80 = librosa.filters.mel(sr=16000, n_fft=512, n_mels=80, fmin=20.0, fmax=8000.0, htk=True)


class SlotEMeansDiarizer:
    """Open-speaker diarizer on top of the Nemotron-3 slot head. Choose how it runs:

      SlotEMeansDiarizer(streaming=True,  latency="low_latency")   # default: causal, audio arrives in chunks, labels may be revised for 10 s
      SlotEMeansDiarizer(streaming=False)                          # whole recording at once: best accuracy, needs the full audio first

    ``diarize(wav)`` works in both modes and returns [(start_s, end_s, speaker_id)]. In streaming mode it feeds the waveform in 100 ms
    chunks as a live system would; ``new_session()`` gives the incremental interface (streaming only)."""

    def __init__(self, device="cuda", mode="low_latency", token=None, streaming=True, latency=None, **live_kwargs):
        mode = latency or mode; assert mode in MODES
        self.streaming, self.mode, self.device, self.live_kwargs = streaming, mode, device, live_kwargs
        self.engine = _Engine(device, token or os.environ.get("HF_TOKEN"))
        self.proc = Nemotron3DiarizationProcessor(feature_extractor=self.engine.fe, subsampling_factor=8, streaming_mode=mode)

    @classmethod
    def from_pretrained(cls, *args, **kw):          # the repository stores code only; models are fetched from their own repos
        return cls(*args, **kw)

    def new_session(self):
        if not self.streaming: raise RuntimeError("new_session() is the incremental interface; create the diarizer with streaming=True")
        e = self.engine
        return LiveSession(LiveNemotron(e.nemotron, self.proc, self.device, self.mode), LiveResNet(e), partial(OfficialEMeansSlotDiarizer, n_init=1), self.mode, **self.live_kwargs)

    def diarize(self, wav, sr=16000, chunk_s=0.1):
        """wav: float array in [-1, 1], mono. Returns [(start_s, end_s, speaker_id)], speaker ids are permanent (slot-anchored)."""
        if sr != 16000: wav = librosa.resample(np.asarray(wav, np.float32), orig_sr=sr, target_sr=16000)
        if not self.streaming: return self._diarize_offline(np.asarray(wav, np.float32))
        pcm = (np.clip(np.asarray(wav, np.float32), -1, 1) * 32767).astype("<i2").tobytes(); sess = self.new_session(); step = int(chunk_s * 16000) * 2
        for k in range(0, len(pcm), step): sess.feed(pcm[k:k + step])
        sess.feed(None, flush=True)
        return frames_to_segments(sess.ld.cur)


    def _diarize_offline(self, wav):
        from .offline import extract
        embs, probs = extract(self.engine, wav)
        frames, *_ = OfficialEMeansSlotDiarizer(stride_ms=80).fit_predict(embs, probs)
        return frames_to_segments(frames)


def frames_to_segments(frame_labels, gap_s=0.16):
    """per-80 ms-frame lists of speaker ids -> merged segments (gaps shorter than gap_s are bridged)"""
    open_, out = {}, []
    for t, labs in enumerate(list(frame_labels) + [[]]):
        for g in list(open_):
            if g not in labs and t * STEP - open_[g][1] >= gap_s: out.append((open_[g][0], open_[g][1], g)); del open_[g]
        for g in labs:
            open_[g] = (open_[g][0] if g in open_ else t * STEP, (t + 1) * STEP)
    out += [(a, b, g) for g, (a, b) in open_.items()]
    return sorted(out)
