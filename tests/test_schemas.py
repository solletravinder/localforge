from __future__ import annotations

import io

from PIL import Image

from local_forge.catalog import CATALOG
from local_forge.schemas import ChatMessage, prepare_messages


def _png_bytes(color=(255, 0, 0)) -> bytes:
    image = Image.new("RGB", (4, 4), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_catalog_ids_are_unique():
    ids = [spec.id for spec in CATALOG]
    assert len(ids) == len(set(ids))
    assert "gemma4-e2b-uncensored-ov" in ids


def test_prepare_messages_with_data_url():
    import base64

    payload = base64.b64encode(_png_bytes()).decode("ascii")
    messages = [
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}},
            ],
        )
    ]
    prepared = prepare_messages(messages)
    assert prepared.messages[0]["content"] == "What is this?"
    assert len(prepared.images) == 1
    assert prepared.images[0].startswith(b"\x89PNG")
    assert prepared.had_image_parts is True
