from pathlib import Path

from studio.ltx_local import LtxLocalClient


def test_cache_key_changes_with_seed(tmp_path: Path) -> None:
    image = tmp_path / "input.png"
    image.write_bytes(b"image")

    first = LtxLocalClient._cache_key(image, "move", 1)
    second = LtxLocalClient._cache_key(image, "move", 2)

    assert first != second


def test_cached_video_is_reused_without_loading_pipeline(caplog, tmp_path: Path) -> None:
    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    client = LtxLocalClient(tmp_path)
    cache_key = client._cache_key(image, "move", 42)
    (client.cache_dir / f"{cache_key}.mp4").write_bytes(b"video")
    client._load_pipeline = lambda: (_ for _ in ()).throw(AssertionError("loaded"))
    output = tmp_path / "result" / "clip.mp4"

    with caplog.at_level("INFO", logger="uvicorn.error.studio.ltx_local"):
        result = client.generate(image, "move", output, seed=42)

    assert output.read_bytes() == b"video"
    assert result["cached"] is True
    assert result["provider"] == client.provider_name
    assert "LTX 캐시 사용" in caplog.text
    assert "move" not in caplog.text


def test_close_releases_loaded_pipeline(monkeypatch, tmp_path: Path) -> None:
    class FakeCuda:
        def __init__(self):
            self.emptied = False

        @staticmethod
        def is_available() -> bool:
            return True

        def empty_cache(self) -> None:
            self.emptied = True

    fake_cuda = FakeCuda()
    monkeypatch.setattr("torch.cuda", fake_cuda)
    client = LtxLocalClient(tmp_path)
    client._pipeline = object()

    client.close()

    assert client._pipeline is None
    assert fake_cuda.emptied is True


def test_generation_passes_image_to_video_options(monkeypatch, tmp_path: Path) -> None:
    import diffusers.utils
    import torch

    image = tmp_path / "input.png"
    image.write_bytes(b"image")
    output = tmp_path / "result" / "clip.mp4"
    calls = []
    generator_calls = []

    class FakeGenerator:
        def __init__(self, *, device):
            generator_calls.append({"device": device})

        def manual_seed(self, seed):
            generator_calls[-1]["seed"] = seed
            return self

    class FakePipeline:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return type("Result", (), {"frames": [["frame"]]})()

    def fake_export(_frames, path, *, fps):
        Path(path).write_bytes(f"fps={fps}".encode("ascii"))

    monkeypatch.setattr(diffusers.utils, "load_image", lambda path: f"image:{path}")
    monkeypatch.setattr(diffusers.utils, "export_to_video", fake_export)
    monkeypatch.setattr(torch, "Generator", FakeGenerator)
    client = LtxLocalClient(tmp_path)
    client._load_pipeline = lambda: FakePipeline()

    result = client.generate(image, "subtle movement", output, seed=7)

    assert result["cached"] is False
    assert output.is_file()
    assert calls[0]["image"] == f"image:{image}"
    assert calls[0]["prompt"] == "subtle movement"
    assert calls[0]["width"] == 288
    assert calls[0]["height"] == 512
    assert calls[0]["num_frames"] == 49
    assert calls[0]["num_inference_steps"] == 25
    assert calls[0]["generator"].__class__ is FakeGenerator
    assert generator_calls == [{"device": "cuda", "seed": 7}]
