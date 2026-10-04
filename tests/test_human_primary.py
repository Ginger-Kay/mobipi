import hashlib
import json
import pytest
from mobiwam.human_primary import SOURCE_FILES, verify_freeze, reserve_primary


def frozen(tmp_path):
    source=tmp_path/'source';source.mkdir()
    for name in SOURCE_FILES: (source/name).write_text(name)
    config=dict(source=str(source),frozen_source=str(source),primary_enabled=True,
        frozen_at='2026-10-04T00:00:00+00:00',batch=str(tmp_path),
        config_version='primary-v1',route_order=['D','A','E'],max_sim_seconds=120,
        freeze_receipt=str(tmp_path/'freeze.json'))
    receipt=dict(version='human-primary-freeze-v1',config=config,
        source_sha256={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in SOURCE_FILES})
    (tmp_path/'freeze.json').write_text(json.dumps(receipt))
    return config


def test_exact_freeze_accepts_and_duplicate_slot_rejected(tmp_path):
    cfg=frozen(tmp_path);verify_freeze(cfg)
    first=reserve_primary(cfg,'D','operator')
    assert json.loads(first.read_text())['status']=='reserved'
    with pytest.raises(ValueError,match='already reserved'):
        reserve_primary(cfg,'D','operator')


def test_config_change_rejected(tmp_path):
    cfg=frozen(tmp_path);cfg['route_order']=['A','D','E']
    with pytest.raises(ValueError,match='differs'):verify_freeze(cfg)


def test_source_change_rejected(tmp_path):
    cfg=frozen(tmp_path);(tmp_path/'source/integration.npy').write_text('changed')
    with pytest.raises(ValueError,match='Source changed'):verify_freeze(cfg)


def test_missing_source_inventory_rejected(tmp_path):
    cfg=frozen(tmp_path);p=tmp_path/'freeze.json';r=json.loads(p.read_text())
    r['source_sha256'].pop('model.xml');p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='Incomplete'):verify_freeze(cfg)
