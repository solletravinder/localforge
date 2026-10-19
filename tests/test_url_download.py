import http.server
import threading
from functools import partial

from local_forge.url_download import download_url, normalize_model_url


def test_normalize_huggingface_blob_url():
    url = "https://huggingface.co/org/repo/blob/main/model.gguf?download=true"
    assert normalize_model_url(url) == (
        "https://huggingface.co/org/repo/resolve/main/model.gguf?download=true"
    )


def test_direct_url_download(tmp_path):
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    payload = b"test-model"
    (serve_dir / "model.gguf").write_bytes(payload)

    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(serve_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        target = download_url(
            f"http://127.0.0.1:{port}/model.gguf",
            tmp_path / "models" / "test",
        )
        assert target.read_bytes() == payload
    finally:
        server.shutdown()
        server.server_close()
