from pathlib import Path

import pytest

import studio.video_provider as video_provider
from studio.ltx_local import LtxLocalClient
from studio.video_provider import VideoProviderConfigurationError


def test_default_provider_is_ltx() -> None:
    assert video_provider.VIDEO_PROVIDER == "ltx"


def test_ltx_provider_does_not_create_wan_client(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        video_provider,
        "Wan22CloudClient",
        lambda: (_ for _ in ()).throw(AssertionError("Wan2.2 initialized")),
    )

    client = video_provider.create_video_generator(tmp_path, "ltx")

    assert isinstance(client, LtxLocalClient)


def test_wan22_provider_is_selected(monkeypatch, tmp_path: Path) -> None:
    expected = object()
    monkeypatch.setattr(video_provider, "Wan22CloudClient", lambda: expected)

    assert video_provider.create_video_generator(tmp_path, "wan22") is expected


def test_invalid_provider_is_rejected() -> None:
    with pytest.raises(VideoProviderConfigurationError, match="VIDEO_PROVIDER"):
        video_provider.validate_video_provider("unknown")
