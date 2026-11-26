"""Loopback-only desktop workbench: python -m demo.workbench.app [--port 8765]."""

import argparse
import hmac
import json
import logging
import os
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

from flask import Flask, jsonify, request, session

from .inputs import LIMIT, ROOT, InputError, address, censys, credentials, uploaded
from .results import STAGES, archive_groups, archives, fail, observe, record, replay
from .worker import CPU_ENV, worker_env

app = Flask(__name__, static_folder="static")
app.config.update(SECRET_KEY=secrets.token_hex(32), LIVE_KEY=secrets.token_urlsafe(32), MAX_CONTENT_LENGTH=LIMIT + 65536,
                  SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Strict", MAX_FORM_PARTS=16)
slot = threading.BoundedSemaphore(1)
lock = threading.Lock()
jobs = OrderedDict()


def provision_key():
    directory = Path.home() / ".cache" / "iotprober-workbench"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / "live.key"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), encoding="ascii") as f:
            info = os.fstat(f.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise RuntimeError("Workbench key must be an owner-only regular file")
            key = f.read(128).strip()
            if len(key) < 32:
                raise RuntimeError("Invalid workbench key")
            app.config["LIVE_KEY"] = key
    else:
        with os.fdopen(fd, "w", encoding="ascii") as f:
            f.write(app.config["LIVE_KEY"] + "\n")


def owner():
    if "client" not in session:
        session["client"] = secrets.token_urlsafe(24)
    return session["client"]


@app.before_request
def protect():
    # No permissive CORS, form POSTs from other sites, or DNS-rebinding access.
    if request.host.split(":")[0].lower() not in {"localhost", "127.0.0.1"}:
        return jsonify(error="请使用 localhost 或 127.0.0.1 的 SSH 转发地址。"), 403
    if request.method == "POST" and not hmac.compare_digest(
            request.headers.get("X-Workbench-Token", "").encode(), owner().encode()):
        return jsonify(error="页面会话已失效，请刷新。"), 403


@app.after_request
def secure(response):
    response.headers.update({"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                             "Referrer-Policy": "no-referrer", "X-Frame-Options": "SAMEORIGIN",
                             "Content-Security-Policy": "default-src 'self'; object-src 'none'; base-uri 'none'; "
                                                        "form-action 'self'; frame-ancestors 'self'"})
    return response


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/setup")
def setup():
    groups = archive_groups()
    return jsonify(token=owner(), samples=groups["verified"], archive_groups=groups,
                   stages=STAGES, censys_ready=bool(credentials()[0]),
                   unlocked=bool(session.get("live")), device="CPU", graph_write=False)


@app.post("/api/unlock")
def unlock():
    key = request.form.get("key", "")
    if not hmac.compare_digest(key.encode(), app.config["LIVE_KEY"].encode()):
        return jsonify(error="工作台口令不正确。它不是 Censys 或 LLM 的 API Key。"), 403
    session["live"] = True
    return jsonify(unlocked=True)


@app.get("/api/archive/<key>")
def archive(key):
    if key not in archives():
        return jsonify(error="没有此归档记录。"), 404
    return jsonify(replay(key))


def supervise(job_id, run, fp):
    try:
        with tempfile.TemporaryDirectory(prefix="iotprober-ui-") as raw:
            folder = Path(raw)
            (folder / "input.json").write_text(json.dumps({"run": run, "fingerprint": fp}))
            progress = folder / "progress.json"
            with lock:
                jobs[job_id]["progress"] = progress
            try:
                command = [sys.executable, "-B", "-m", "demo.workbench.worker", str(folder)]
                idle_io = shutil.which("ionice")
                subprocess.run(([idle_io, "-c", "3"] if idle_io else []) + command,
                               cwd=ROOT, env=worker_env(folder), stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=1800, check=True)
                message = None
            except subprocess.TimeoutExpired:
                message = "超过 30 分钟，已停止本次工作台任务。已完成阶段保留。"
            except (subprocess.CalledProcessError, OSError):
                message = "工作台子进程未正常结束，请检查依赖和可用内存。"
            if progress.exists():
                run = json.loads(progress.read_text())
            if message or run["state"] == "running":
                fail(run, message or "未收到完整结果。")
            with lock:
                jobs[job_id].update(run=run, progress=None)
    except Exception:
        fail(run, "无法保存本次任务状态。")
        with lock:
            jobs[job_id].update(run=run, progress=None)
    finally:
        slot.release()


@app.post("/api/runs")
def start():
    if not session.get("live"):
        return jsonify(error="请先解锁实时识别；归档回放不需要工作台口令。"), 403
    form = request.form
    ip = address(form.get("ip", ""))
    source, mode = form.get("source"), form.get("mode")
    if source not in {"upload", "censys"} or mode not in {"local", "full"}:
        raise InputError("请选择有效的数据来源及识别模式。")
    if mode == "full" and form.get("llm_consent") != "yes":
        raise InputError("完整模式需要逐次确认向配置的 LLM 服务发送指纹和证据。")
    if source == "censys" and form.get("censys_consent") != "yes":
        raise InputError("请确认查询的是获授权的设备，且同意使用一次 Censys 查询额度。")
    if not slot.acquire(blocking=False):
        return jsonify(error="已有识别任务在运行。为保护共享服务器，请稍后再试。"), 429
    try:
        if source == "upload":
            f = request.files.get("file")
            if not f or not f.filename:
                raise InputError("请选择要上传的 JSON 或 CSV 文件。")
            fp = uploaded(ip, f.filename, f.stream.read(LIMIT + 1))
        else:
            fp = censys(ip, form.get("pat", "").strip(), form.get("org", "").strip())
        run = record(ip, source, mode)
        observe(run, fp)
        job_id = secrets.token_urlsafe(16)
        with lock:
            while len(jobs) >= 12:
                jobs.popitem(last=False)
            jobs[job_id] = {"owner": owner(), "run": run, "progress": None}
        threading.Thread(target=supervise, args=(job_id, run, fp), daemon=True).start()
    except BaseException:
        slot.release()
        raise
    return jsonify(id=job_id), 202


@app.get("/api/runs/<job_id>")
def status(job_id):
    with lock:
        job = jobs.get(job_id)
        if not job or job["owner"] != owner():
            return jsonify(error="记录已过期、服务已重启或不属于此浏览器会话。"), 404
        state = job["run"]
        if job["progress"] and job["progress"].is_file():
            state = json.loads(job["progress"].read_text())
    return jsonify(state)


@app.errorhandler(InputError)
def bad_input(exc):
    return jsonify(error=str(exc)), 400


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="上传超过限制（单文件最多 2 MiB）。"), 413


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    os.environ.update(CPU_ENV)
    provision_key()
    logging.basicConfig(level=logging.WARNING)
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False, threaded=True)
