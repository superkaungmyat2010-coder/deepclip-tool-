import os, re, uuid, shutil, subprocess, threading, json
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")
DATA = BASE / "data"
UPLOADS = DATA / "uploads"
OUTPUTS = DATA / "outputs"
for folder in (DATA, UPLOADS, OUTPUTS):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="DeepClip AI Recap")
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")

JOBS = {}
LOCK = threading.Lock()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
ALLOWED_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}

def run_cmd(args, timeout=None):
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "Command failed")[-2500:])
    return result.stdout

def update(job_id, **kwargs):
    with LOCK:
        JOBS.setdefault(job_id, {}).update(kwargs)

def valid_video_url(url):
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    allowed = ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be",
               "tiktok.com", "www.tiktok.com", "vm.tiktok.com", "vt.tiktok.com")
    if parsed.scheme not in ("http", "https") or not any(host == d or host.endswith("." + d) for d in allowed):
        raise ValueError("YouTube သို့မဟုတ် TikTok Link ကိုသာ ထည့်ပါ။")
    return url.strip()

def get_openai_client():
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    from openai import OpenAI
    return OpenAI(api_key=key)

def transcribe(video_path):
    from faster_whisper import WhisperModel
    model_name = os.getenv("WHISPER_MODEL", "small")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    model = WhisperModel(model_name, device=device, compute_type=compute)
    segments, info = model.transcribe(str(video_path), vad_filter=True, beam_size=5)
    out = []
    for s in segments:
        text = (s.text or "").strip()
        if text:
            out.append({"start": float(s.start), "end": float(s.end), "text": text})
    if not out:
        raise RuntimeError("အသံထဲမှ စာသား မတွေ့ပါ။ Video အသံပါ/မပါ စစ်ဆေးပါ။")
    return out, getattr(info, "language", "unknown")

def choose_segments(segments, client, target_seconds=60):
    # AI selection is optional; if no API key, choose evenly distributed transcript spans.
    if client:
        transcript = "\n".join(f'{i}: [{s["start"]:.1f}-{s["end"]:.1f}] {s["text"]}' for i, s in enumerate(segments))
        prompt = (
            "You select concise, meaningful highlights for a video recap. "
            f"Choose transcript segment indices whose combined duration is about {target_seconds} seconds. "
            "Prefer key facts, turning points, explanations, and conclusions; avoid repetitions and greetings. "
            "Return ONLY JSON: {\"indices\":[integer,...]}. Keep chronological order. "
            "Indices must be from the transcript below.\n\n" + transcript[:45000]
        )
        try:
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role":"user","content":prompt}],
                temperature=0.2,
                response_format={"type":"json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            indices = sorted(set(int(i) for i in data.get("indices", []) if 0 <= int(i) < len(segments)))
            if indices:
                return indices
        except Exception:
            pass
    # Fallback: choose evenly distributed spans, capped to roughly target duration.
    chosen, elapsed = [], 0.0
    step = max(1, len(segments) // max(1, min(12, target_seconds // 5)))
    for i in range(0, len(segments), step):
        chosen.append(i)
        elapsed += max(0.5, segments[i]["end"] - segments[i]["start"])
        if elapsed >= target_seconds:
            break
    return sorted(set(chosen or [0]))

def translate_lines(lines, client):
    if not client:
        return [x for x in lines]  # Original-language subtitles if no API key configured.
    translated = []
    batch_size = 35
    for offset in range(0, len(lines), batch_size):
        batch = lines[offset:offset+batch_size]
        prompt = (
            "Translate each subtitle line into natural, concise Burmese (Myanmar language). "
            "Keep names and numbers accurate. Return ONLY JSON as {\"translations\":[...]} "
            "with exactly one translation per input line, in the same order.\n"
            + json.dumps(batch, ensure_ascii=False)
        )
        try:
            r = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role":"user","content":prompt}],
                temperature=0.2,
                response_format={"type":"json_object"},
            )
            vals = json.loads(r.choices[0].message.content).get("translations", [])
            if len(vals) != len(batch):
                raise ValueError("Translation count mismatch")
            translated.extend(str(v) for v in vals)
        except Exception:
            translated.extend(batch)
    return translated

def ts_srt(seconds):
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def make_srt(chosen, segments, translated, work_dir):
    rows, cursor = [], 0.0
    for n, (idx, subtitle) in enumerate(zip(chosen, translated), 1):
        seg = segments[idx]
        dur = max(0.6, seg["end"] - seg["start"])
        start = cursor
        end = cursor + dur
        rows.append(f"{n}\n{ts_srt(start)} --> {ts_srt(end)}\n{subtitle}\n")
        cursor = end + 0.08
    path = work_dir / "subtitles.srt"
    path.write_text("\n".join(rows), encoding="utf-8")
    return path

def process_job(job_id, source_path, aspect, target_seconds):
    work_dir = OUTPUTS / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        update(job_id, status="processing", progress=5, message="Video ကို စစ်ဆေးနေသည်...")
        client = get_openai_client()
        segments, language = transcribe(source_path)
        update(job_id, progress=35, message="အသံကို စာသားပြောင်းပြီးပါပြီ။ Highlight ရွေးနေသည်...")
        chosen = choose_segments(segments, client, target_seconds)
        raw_lines = [segments[i]["text"] for i in chosen]
        translated = translate_lines(raw_lines, client)
        update(job_id, progress=55, message="Recap အပိုင်းများကို တည်ဆောက်နေသည်...")
        clips = []
        cursor = 0.0
        for n, idx in enumerate(chosen):
            seg = segments[idx]
            start = max(0.0, seg["start"] - 0.15)
            end = max(start + 0.7, seg["end"] + 0.2)
            clip = work_dir / f"clip_{n:03}.mp4"
            run_cmd(["ffmpeg", "-y", "-ss", str(start), "-i", str(source_path), "-t", str(end-start),
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", str(clip)], timeout=600)
            clips.append(clip)
        if not clips:
            raise RuntimeError("Recap clip များ မဖန်တီးနိုင်ပါ။")
        concat_file = work_dir / "concat.txt"
        concat_file.write_text("".join(f"file '{p.as_posix()}'\n" for p in clips), encoding="utf-8")
        joined = work_dir / "joined.mp4"
        run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
                 "-c", "copy", str(joined)], timeout=600)
        srt = make_srt(chosen, segments, translated, work_dir)
        aspect_filters = {
            "9:16": "scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2",
            "16:9": "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
            "1:1": "scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:(ow-iw)/2:(oh-ih)/2",
            "3:4": "scale=900:1200:force_original_aspect_ratio=decrease,pad=900:1200:(ow-iw)/2:(oh-ih)/2",
        }
        vf = aspect_filters.get(aspect, aspect_filters["9:16"])
        # subtitles filter uses a relative path escaped for ffmpeg's filter syntax.
        srt_filter_path = str(srt.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
        vf += f",subtitles='{srt_filter_path}':charenc=UTF-8"
        output = work_dir / "recap.mp4"
        update(job_id, progress=75, message="မြန်မာစာတန်းထိုးပြီး MP4 ထုတ်နေသည်...")
        run_cmd(["ffmpeg", "-y", "-i", str(joined), "-vf", vf, "-c:v", "libx264",
                 "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-b:a", "128k",
                 "-shortest", "-movflags", "+faststart", str(output)], timeout=1200)
        summary = "\n".join(f"• {segments[i]['text']}" for i in chosen)
        if client:
            try:
                r = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[{"role":"user","content":"အောက်ပါ transcript မှ မြန်မာဘာသာဖြင့် အတိုချုံး Video Recap ရေးပေးပါ။ အဓိကအချက်များကို bullet များဖြင့်ရေးပါ။\n\n"+summary[:12000]}],
                    temperature=0.3,
                )
                summary = r.choices[0].message.content or summary
            except Exception:
                pass
        summary_path = work_dir / "summary.txt"
        summary_path.write_text(summary, encoding="utf-8")
        update(job_id, status="done", progress=100, message="ပြီးပါပြီ။ MP4 ကို ဒေါင်းလုဒ်လုပ်နိုင်ပါပြီ။",
               download_url=f"/download/{job_id}", summary=summary, language=language)
    except Exception as e:
        update(job_id, status="error", progress=100, message="Processing မအောင်မြင်ပါ။",
               error=str(e)[:1800])
    finally:
        # Keep output for download; clean source upload to save disk space.
        try:
            if source_path.exists():
                source_path.unlink()
        except Exception:
            pass

@app.get("/", response_class=HTMLResponse)
def home():
    return (BASE / "static" / "index.html").read_text(encoding="utf-8")

@app.post("/api/jobs")
async def create_job(background_tasks: BackgroundTasks,
                     video: UploadFile = File(None),
                     url: str = Form(""),
                     aspect: str = Form("9:16"),
                     target_seconds: int = Form(60)):
    job_id = uuid.uuid4().hex[:12]
    source_path = UPLOADS / f"{job_id}_source"
    try:
        if video and video.filename:
            ext = Path(video.filename).suffix.lower()
            if ext not in ALLOWED_EXTS:
                raise HTTPException(400, "MP4, MOV, MKV, WEBM ဖိုင်များကိုသာ တင်ပါ။")
            data = await video.read()
            if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
                raise HTTPException(413, f"ဖိုင်အရွယ်အစား {MAX_UPLOAD_MB} MB ထက် မကျော်ရပါ။")
            source_path = source_path.with_suffix(ext)
            source_path.write_bytes(data)
        elif url.strip():
            valid_video_url(url)
            source_path = source_path.with_suffix(".mp4")
            update(job_id, status="queued", progress=2, message="Link မှ Video ကို ရယူနေသည်...")
            try:
                run_cmd(["yt-dlp", "--no-playlist", "-f", "bv*+ba/b", "--merge-output-format", "mp4",
                         "--max-filesize", f"{MAX_UPLOAD_MB}M", "-o", str(source_path), url.strip()], timeout=900)
            except Exception as e:
                raise HTTPException(400, "Link မှ Video ရယူမရပါ။ Public link ဖြစ်/မဖြစ် စစ်ပါ။ " + str(e)[-500:])
        else:
            raise HTTPException(400, "Video ဖိုင်တင်ပါ သို့မဟုတ် Link ထည့်ပါ။")
        with LOCK:
            JOBS[job_id] = {"status":"queued","progress":3,"message":"တန်းစီနေသည်။","summary":"","error":""}
        background_tasks.add_task(process_job, job_id, source_path, aspect, max(15, min(180, target_seconds)))
        return {"job_id": job_id}
    except HTTPException:
        if source_path.exists():
            source_path.unlink()
        raise
    except Exception as e:
        if source_path.exists():
            source_path.unlink()
        raise HTTPException(500, str(e)[:1000])

@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job မတွေ့ပါ။")
    return job

@app.get("/download/{job_id}")
def download(job_id: str):
    path = OUTPUTS / job_id / "recap.mp4"
    if not path.exists():
        raise HTTPException(404, "ဖိုင် မပြီးသေးပါ သို့မဟုတ် မတွေ့ပါ။")
    return FileResponse(path, media_type="video/mp4", filename=f"deepclip_{job_id}.mp4")

@app.get("/health")
def health():
    return {"ok": True}
