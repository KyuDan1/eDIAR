"""python examples/diarize.py meeting.wav [--offline] [--latency low_latency|very_low_latency] [--device cuda]  ->  RTTM on stdout
   default: streaming (causal, 100 ms chunks).  --offline: non-streaming, the whole recording at once."""
import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import soundfile as sf
from emeans_diar import SlotEMeansDiarizer

ap = argparse.ArgumentParser(); ap.add_argument("wav"); ap.add_argument("--latency", "--mode", dest="latency", default="low_latency", choices=["low_latency", "very_low_latency"], help="streaming input buffer: 1.04 s or 0.64 s"); ap.add_argument("--offline", action="store_true", help="non-streaming: use the whole recording at once"); ap.add_argument("--device", default="cuda")
a = ap.parse_args()
wav, sr = sf.read(a.wav, dtype="float32")
if wav.ndim > 1: wav = wav.mean(1)
name = os.path.splitext(os.path.basename(a.wav))[0]
for s, e, spk in SlotEMeansDiarizer(device=a.device, latency=a.latency, streaming=not a.offline).diarize(wav, sr):
    print(f"SPEAKER {name} 1 {s:.3f} {e - s:.3f} <NA> <NA> spk{spk} <NA> <NA>")
