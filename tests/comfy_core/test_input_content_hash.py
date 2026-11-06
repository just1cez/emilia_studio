import hashlib

import pytest
import torch
from comfy.cli_args import args
if not torch.cuda.is_available():
    args.cpu = True

import nodes


@pytest.mark.parametrize('loader', [nodes.LoadImage, nodes.LoadLatent])
@pytest.mark.parametrize('size', [0, 1024 * 1024, 1024 * 1024 + 17])
def test_content_hash_includes_tail_and_same_size_changes(tmp_path, monkeypatch, loader, size):
    path = tmp_path / 'input.bin'
    data = bytes((i % 251 for i in range(size)))
    path.write_bytes(data)
    monkeypatch.setattr(nodes.folder_paths, 'get_annotated_filepath', lambda name: str(path))
    assert loader.IS_CHANGED('input.bin') == hashlib.sha256(data).hexdigest()
    if size:
        changed = data[:-1] + bytes([data[-1] ^ 255])
        path.write_bytes(changed)
        assert loader.IS_CHANGED('input.bin') == hashlib.sha256(changed).hexdigest()
        assert loader.IS_CHANGED('input.bin') != hashlib.sha256(data).hexdigest()
