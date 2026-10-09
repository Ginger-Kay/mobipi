"""Exact-provenance deadline guard; never signals another run or keepalive."""
import argparse, json, os, signal, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from pi05_drawer_data_v2 import owned, read, write, now

def main(r):
    deadline=datetime.fromisoformat(read(r/'deadline-config.json')['execution_deadline'])
    while datetime.now(timezone.utc)<deadline-timedelta(seconds=45):
        phase=r/'phase-state.json'
        if phase.exists() and read(phase).get('phase')=='execution_closed':return
        time.sleep(2)
    records=[]
    for path in (r/'launch').glob('*-process.json'):
        rec=read(path)
        if not owned(rec) or rec.get('run_id')!=r.name:continue
        if not any(str(r) in arg for arg in rec['command']):continue
        if path.name.startswith('episode-') or path.name.startswith('prepare-'):records.append(rec)
    for rec in records:
        if owned(rec):os.kill(rec['pid'],signal.SIGINT)
    write(r/'delivery/deadline-guard-receipt.json',dict(at=now(),deadline=deadline.isoformat(),interrupted_owned_PIDs=[x['pid'] for x in records],unknown_terminal_preserved=True))
    while datetime.now(timezone.utc)<deadline-timedelta(seconds=15) and any(owned(x) for x in records):time.sleep(1)
    for rec in records:
        if owned(rec):os.kill(rec['pid'],signal.SIGTERM)
    for path in (r/'launch').glob('service*-process.json'):
        rec=read(path)
        if owned(rec) and rec.get('run_id')==r.name and any(str(r) in a for a in rec['command']):os.kill(rec['pid'],signal.SIGTERM)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
