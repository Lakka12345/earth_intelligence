import hashlib
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agents.agent4_download_engine import DownloadEngine, DownloadTask
from agents.agent4_download_manager import plan_download_batches
from models.website_analysis_schemas import SourceSnapshot


DATA = b"earth-intelligence-download-engine" * 256


class RangeHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/login":
            body = b"<!doctype html><html><title>login</title></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        range_header = self.headers.get("Range")
        if range_header:
            start = int(range_header.split("=", 1)[1].split("-", 1)[0])
            body = DATA[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(DATA) - 1}/{len(DATA)}")
        else:
            body = DATA
            self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


def _server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def test_download_engine_resumes_and_verifies_sha256():
    httpd = _server()
    try:
        url = f"http://127.0.0.1:{httpd.server_port}/dataset.bin"
        expected = hashlib.sha256(DATA).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "dataset.bin")
            with open(dest, "wb") as f:
                f.write(DATA[:100])
            result = DownloadEngine(timeout=5).download_one(
                DownloadTask(
                    url=url,
                    dest_path=dest,
                    expected_size=len(DATA),
                    checksum=expected,
                )
            )
            assert result.success
            assert result.resumed
            assert result.size_bytes == len(DATA)
            assert result.checksum_status == "verified"
            with open(dest, "rb") as f:
                assert f.read() == DATA
    finally:
        httpd.shutdown()


def test_download_engine_rejects_html_login_page():
    httpd = _server()
    try:
        url = f"http://127.0.0.1:{httpd.server_port}/login"
        with tempfile.TemporaryDirectory() as tmp:
            result = DownloadEngine(timeout=5).download_one(
                DownloadTask(url=url, dest_path=os.path.join(tmp, "login.html"))
            )
            assert not result.success
            assert result.validation_status == "failed"
            assert "HTML" in result.error
    finally:
        httpd.shutdown()


def test_download_planner_batches_parallel_capable_sources_without_duplicates():
    snapshots = {
        "a": SourceSnapshot(source_id="a", name="a", url="https://example.com/a.csv"),
        "b": SourceSnapshot(source_id="b", name="b", url="https://example.com/b.csv"),
        "a-duplicate": SourceSnapshot(source_id="a-duplicate", name="dup", url="https://example.com/dup.csv"),
    }
    batches = plan_download_batches(["a", "b", "a"], snapshots)
    flattened = [item.source_id for batch in batches for item in batch]
    assert flattened == ["a", "b"]
    assert len(batches[0]) == 2
    assert all(item.can_parallelize for item in batches[0])
