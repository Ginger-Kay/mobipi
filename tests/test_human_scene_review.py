import json
import pytest
from prepare_human_scene_review import check_preview

def test_reject_wrong_source(tmp_path):
    p=tmp_path/'main.png';p.write_bytes(bytes.fromhex('89504e470d0a1a0a'))
    (tmp_path/'receipt.json').write_text(json.dumps(dict(source=str(tmp_path/'other'),physics_steps=0)))
    with pytest.raises(ValueError,match='Source mismatch'):check_preview(p,tmp_path/'wanted')

def test_reject_action_preview(tmp_path):
    p=tmp_path/'main.png';p.write_bytes(bytes.fromhex('89504e470d0a1a0a'))
    (tmp_path/'receipt.json').write_text(json.dumps(dict(source=str(tmp_path),physics_steps=1)))
    with pytest.raises(ValueError,match='zero-action'):check_preview(p,tmp_path)

def test_reject_corrupt_image(tmp_path):
    p=tmp_path/'main.png';p.write_bytes(b'bad')
    (tmp_path/'receipt.json').write_text(json.dumps(dict(source=str(tmp_path),physics_steps=0)))
    with pytest.raises(ValueError,match='PNG'):check_preview(p,tmp_path)
