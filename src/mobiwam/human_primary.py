"""Frozen human collection admission; no physics, policy or outcome changes."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

SOURCE_FILES = ('model.xml', 'integration.npy', 'rng.json', 'source.json',
                'target-binding.json', 'ep_meta.json')


def verify_freeze(config):
    if not config.get('primary_enabled') or not config.get('frozen_at'):
        raise ValueError('Primary configuration is not frozen')
    source = Path(config['source']).resolve()
    if source != Path(config['frozen_source']).resolve():
        raise ValueError('Frozen Source differs')
    receipt = json.loads(Path(config['freeze_receipt']).read_text())
    if receipt.get('version') != 'human-primary-freeze-v1' or receipt.get('config') != config:
        raise ValueError('Configuration differs from freeze receipt')
    if set(receipt.get('source_sha256', {})) != set(SOURCE_FILES):
        raise ValueError('Incomplete Source inventory')
    for name, expected in receipt['source_sha256'].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen Source changed: ' + name)
    if sorted(config['route_order']) != ['A','D','E'] or config.get('max_sim_seconds') != 120:
        raise ValueError('Invalid frozen route order or budget')
    return receipt


def reserve_primary(config, route, operator):
    """Consume one slot before recorder creation; interrupted slots need diagnosis.

    Never auto-delete/retry a slot after a crash. Existing valid failures consume
    their primary route exactly as successes do.
    """
    verify_freeze(config)
    if route not in ('E','D','A'):
        raise ValueError('Invalid primary route')
    key = hashlib.sha256(str(Path(config['source']).resolve()).encode()).hexdigest()
    root = Path(config['batch']) / 'manifests' / 'primary-reservations' / key
    root.mkdir(parents=True, exist_ok=True)
    path = root / (route + '.json')
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError('Primary slot already reserved; inspect saved/interrupted attempt, do not repeat') from exc
    with os.fdopen(fd, 'w') as f:
        json.dump(dict(created_at=datetime.now(timezone.utc).isoformat(),status='reserved',
            pid=os.getpid(),source=config['source'],route=route,operator_id=operator,
            config_version=config['config_version'],freeze_receipt=config['freeze_receipt']),f,indent=2)
        f.flush(); os.fsync(f.fileno())
    return path
