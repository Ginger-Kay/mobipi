"""Human scene collection metadata and record budgets; no control policy."""
import csv
import fcntl
import json
from pathlib import Path

VERSION = 'human-scene-pilot-v1'
FIELDS = 'attempt_id,source_id,config_version,route,record_type,operator_id,order,started_at,ended_at,machine_success,human_observation,stop_reason,safety_status,integrity_status,route_semantics_status,original_video_path,panoramic_video_path,trajectory_path,result_path,reference_selected,notes'.split(',')


def validate_start(config, existing, route, record_type, operator, source):
    if route not in ('E', 'D', 'A'):
        raise ValueError('Select E, D or A')
    if not operator.strip() or len(operator.strip()) > 80:
        raise ValueError('Enter operator ID before recording')
    if record_type == 'engineering_record_smoke':
        if not config.get('allow_engineering_record_smoke'):
            raise ValueError('Engineering mode is not enabled in the human UI')
        return
    if record_type not in ('practice', 'primary', 'reference_supplement'):
        raise ValueError('Unknown record type')
    if record_type != 'practice':
        if not config.get('primary_enabled') or not config.get('frozen_at'):
            raise ValueError('Pilot only: primary scene is not frozen')
        if str(source) != config.get('frozen_source'):
            raise ValueError('Source differs from frozen primary scene')
        completed = [r for r in existing if r['source_id'] == Path(source).name]
        if record_type == 'primary':
            done = [r['route'] for r in completed if r['record_type'] == 'primary']
            expected = config['route_order']
            if done != expected[:len(done)] or len(done) >= len(expected) or route != expected[len(done)]:
                raise ValueError('Follow frozen order; primary outcomes cannot be replaced')
        else:
            if len([r for r in completed if r['record_type'] == 'primary']) != 3:
                raise ValueError('Finish the primary trio before reference supplements')
            if sum(r['record_type'] == record_type for r in completed) >= 2:
                raise ValueError('Two total reference supplements already used')


def append_index(path, row):
    """Serialize writes from the two separate pilot desktops."""
    path = Path(path)
    with path.open('a+', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        rows = list(csv.DictReader(f))
        if any(r['attempt_id'] == row['attempt_id'] for r in rows):
            raise ValueError('Attempt already indexed')
        f.seek(0, 2)
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow({key: row.get(key, '') for key in FIELDS})
        f.flush()


def load_index(path):
    p = Path(path)
    return list(csv.DictReader(p.open())) if p.exists() else []


def phase_for(route, docked):
    return 'navigate' if route == 'D' and not docked else 'manipulate'
