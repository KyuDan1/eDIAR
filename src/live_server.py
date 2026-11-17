"""
Live meeting demo server: HTTPS + WebSocket (aiohttp). The browser streams 16 kHz mono int16 PCM; the server answers with
JSON updates (embedding dots, speaker labels, revisions). One active session at a time; audio is processed in memory only
and is never written to disk.

  python src/live_server.py --host 127.0.0.1 --port 8123 --device cuda
  (add --cert/--key for HTTPS when serving over the network)
"""
import os, sys, json, time, asyncio, argparse, ssl, logging
from functools import partial
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aiohttp import web, WSMsgType
from pipeline_eval import DualStreamSpeechLLMAcousticEngine
from nemotron_ref.processing_nemotron3_diarization import Nemotron3DiarizationProcessor
from clustering_methods import OfficialEMeansSlotDiarizer
from live_pipeline import LiveNemotron, LiveResNet, LiveSession

log = logging.getLogger("live")
HERE = os.path.dirname(os.path.abspath(__file__))
MODES = ("low_latency", "very_low_latency")
MAX_SECONDS, IDLE_SECONDS = 3 * 3600, 90
STATE = {"engine": None, "proc": None, "busy": False, "pool": ThreadPoolExecutor(1), "device": "cuda"}


def new_session(mode):
    e = STATE["engine"]
    return LiveSession(LiveNemotron(e.nemotron, STATE["proc"], STATE["device"], mode), LiveResNet(e),
                       partial(OfficialEMeansSlotDiarizer, n_init=1), mode)


async def index(request):
    return web.Response(text=open(os.path.join(HERE, "live.html"), encoding="utf-8").read(), content_type="text/html",
                        headers={"Cache-Control": "no-store"})


async def health(request):
    return web.json_response({"ok": True, "busy": STATE["busy"], "modes": MODES})


async def ws_handler(request):
    ws = web.WebSocketResponse(max_msg_size=2 * 1024 * 1024, heartbeat=20)
    await ws.prepare(request)
    if STATE["busy"]:
        await ws.send_json({"type": "busy"}); await ws.close(); return ws
    STATE["busy"] = True
    loop = asyncio.get_running_loop(); queue = asyncio.Queue(); session = None; t_start = time.time(); last_audio = time.time()
    log.info("session start")

    async def worker():
        nonlocal session
        while True:
            item = await queue.get()
            if item is None: break
            chunks, stop = [item], False
            while not queue.empty():                      # coalesce a backlog: one refit for everything that piled up
                nxt = queue.get_nowait()
                if nxt is None: stop = True; break
                chunks.append(nxt)
            try:
                msgs = await loop.run_in_executor(STATE["pool"], session.feed, b"".join(chunks))
                for m in msgs: await ws.send_json(m)
            except Exception as ex:                       # never kill the server for one bad session
                log.exception("feed failed"); await ws.send_json({"type": "error", "msg": str(ex)[:200]})
                if "cuda" in str(ex).lower():                       # a CUDA assert poisons the process: exit, systemd restarts the service
                    log.error("CUDA error -> exiting for restart"); os._exit(3)
                break
            if stop: break

    wtask = None
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                cmd = json.loads(msg.data)
                if cmd.get("type") == "start":
                    mode = cmd.get("mode", "low_latency"); mode = mode if mode in MODES else "low_latency"
                    session = await loop.run_in_executor(STATE["pool"], new_session, mode)
                    wtask = asyncio.create_task(worker())
                    await ws.send_json({"type": "ready", "mode": mode, "latency_s": session.nem.latency_s, "chunk_s": session.nem.chunk_len * 0.08})
                elif cmd.get("type") == "stop":
                    break
            elif msg.type == WSMsgType.BINARY and session is not None:
                last_audio = time.time(); await queue.put(msg.data)
                if time.time() - t_start > MAX_SECONDS: await ws.send_json({"type": "error", "msg": "max duration reached"}); break
            elif msg.type == WSMsgType.ERROR:
                break
        if wtask is not None:
            await queue.put(None); await wtask
            msgs = await loop.run_in_executor(STATE["pool"], partial(session.feed, None, True))
            for m in msgs:
                if not ws.closed: await ws.send_json(m)
            if not ws.closed: await ws.send_json({"type": "final", "frames": len(session.ld.cur)})
    finally:
        STATE["busy"] = False; log.info("session end (%.0f s)", time.time() - t_start)
        if not ws.closed: await ws.close()
    return ws


def build_app():
    app = web.Application()
    app.add_routes([web.get("/", index), web.get("/health", health), web.get("/ws", ws_handler)])
    return app


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8123); ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--cert"); ap.add_argument("--key"); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    STATE["device"] = a.device; log.info("loading models on %s ...", a.device)
    eng = DualStreamSpeechLLMAcousticEngine(device=a.device); STATE["engine"] = eng
    STATE["proc"] = Nemotron3DiarizationProcessor(feature_extractor=eng.fe, subsampling_factor=8, streaming_mode="low_latency")
    import numpy as np                                    # warm-up (CUDA kernels, cuDNN autotune) with 3 s of noise
    s = new_session("low_latency"); s.feed((np.random.randn(48000) * 800).astype("<i2").tobytes()); log.info("models ready")
    ctx = None
    if a.cert:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.load_cert_chain(a.cert, a.key)
    web.run_app(build_app(), host=a.host, port=a.port, ssl_context=ctx, print=lambda *x: None)


if __name__ == "__main__":
    main()
