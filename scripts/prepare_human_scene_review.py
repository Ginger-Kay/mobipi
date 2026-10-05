"""Build a self-contained review packet from existing zero-action evidence."""
import argparse,base64,csv,hashlib,html,json
from pathlib import Path
from datetime import datetime,timezone

def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_preview(path,source):
    path=Path(path);receipt=read(path.parent/'receipt.json')
    bound=receipt.get('source',receipt.get('control'))
    if not bound or Path(bound).resolve()!=Path(source).resolve():raise ValueError('Preview Source mismatch')
    if receipt.get('physics_steps')!=0:raise ValueError('Not a zero-action preview')
    if path.read_bytes()[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('Missing/corrupt PNG signature')
    return dict(path=str(path),sha256=digest(path),receipt=str(path.parent/'receipt.json'))
def build(batch,restore_path,output):
    output.mkdir(parents=True,exist_ok=False)
    restored={r['scene_id']:r for r in read(restore_path)['rows']}
    rows=list(csv.DictReader((batch/'scene-index.csv').open()))
    cards=[];errors=[];fixups=[]
    for row in rows:
        sid=row['scene_id']
        try:
            source=Path(row['source']);cfg_path=Path(row['primary_config'] or row['prepared_config']);cfg=read(cfg_path)
            if cfg['scene_id']!=sid or Path(cfg['source']).resolve()!=source.resolve():raise ValueError('Scene/config Source mismatch')
            if cfg['scene_family_id']!=row['scene_family_id'] or cfg['task']!=row['task']:raise ValueError('Task/family mismatch')
            if ''.join(cfg['route_order'])!=row['planned_order'] or sorted(cfg['route_order'])!=['A','D','E']:raise ValueError('Route order mismatch')
            binding=read(source/'target-binding.json');rr=restored[sid]
            if binding['task']!=cfg['task'] or row['target_joint'] not in [j['name'] for j in binding['joints']]:raise ValueError('Target mismatch')
            if rr['model_sha256']!=digest(source/'model.xml') or rr['restore_max_abs_error']!=0 or rr['restored_rng']!=rr['saved_rng']:raise ValueError('Prior restore receipt stale/failed')
            if rr['checker_initial_success'] or binding['checker_success']:raise ValueError('Target already successful')
            for f in ['integration.npy','rng.json','model.xml','target-binding.json']:
                if not (source/f).is_file():raise ValueError('Missing Source '+f)
            images=[check_preview(row['preview_path'],source),check_preview(row['panoramic_preview'],source)]
            control=Path(row['control_snapshot_path'])
            if control!=source:
                images += [check_preview(control/'preview/main.png',control),check_preview(control/'preview/panoramic.png',control)]
            old=cfg.get('main_preview_path')
            if old and Path(old)!=Path(row['preview_path']):fixups.append(dict(scene_id=sid,inherited_preview=old,review_preview=row['preview_path'],original_config_unchanged=True))
            card=dict(scene_id=sid,task=cfg['task'],family=cfg['scene_family_id'],source=str(source),config=str(cfg_path),
                config_sha256=digest(cfg_path),source_model_sha256=rr['model_sha256'],target_joint=row['target_joint'],
                priority=row['priority'],route_order=cfg['route_order'],frozen=bool(row['frozen_at']),
                factor=cfg.get('factor',dict(name=cfg.get('main_factor'),obstacle=cfg.get('obstacle'))),
                initial_opening=binding['opening'],images=images,restore_receipt=str(restore_path),
                initial_static_clearance_m=cfg.get('initial_static_clearance_m'),review_status='pending_packet_review',
                independent_source_increment=0,full_route_safety='not_established_by_static_review')
            cards.append(card)
        except Exception as e:errors.append(dict(scene_id=sid,error=str(e)))
    packet=dict(created_at=datetime.now(timezone.utc).isoformat(),scenes=cards,errors=errors,
        preview_field_corrections_in_packet_only=fixups,independent_families=len({c['family'] for c in cards}),
        new_sources=0,new_physics_steps=0,new_training=0)
    (output/'packet.json').write_text(json.dumps(packet,indent=2,ensure_ascii=False))
    forms=[dict(scene_id=c['scene_id'],config_sha256=c['config_sha256'],
        image_sha256=[i['sha256'] for i in c['images']],review_status='pending',reviewer=None,reviewed_at=None,
        target_and_handle_visible=None,base_and_obstacles_visible=None,ready_to_try=None,observations='',
        automatic_primary_freeze=False) for c in cards if not c['frozen']]
    (output/'review-forms.json').write_text(json.dumps(forms,indent=2,ensure_ascii=False))
    page=['<!doctype html><meta charset="utf-8"><title>人工场景审核</title><style>body{font:17px sans-serif;max-width:1400px;margin:auto;padding:24px;background:#f5f6f8}section{background:white;padding:20px;margin:20px 0;border-radius:12px}.views{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{width:100%}pre{white-space:pre-wrap}a{color:#1769aa}</style><h1>16 场景预览审核包</h1><p>审核预览与可操作性；静态恢复不证明完整路径安全。先做第一档。16配置仍只有2环境家族。此文件不提交意见、不冻结或启动任何任务。</p>']
    md=['# 场景审核入口','',packet['created_at'],'','先看第一档；当前 MW-O 按 E→D→A 录制。其余未冻结场景先审核目标/把手、底盘/障碍是否看清。回复场景编号、可否尝试和具体观察；机器静态检查不代替人工。','']
    for c in cards:
        sid=c['scene_id'];page.append('<section id="'+sid+'"><h2>'+sid+' / '+c['priority']+'</h2><p>顺序：'+'→'.join(c['route_order'])+'；已冻结：'+str(c['frozen'])+'</p><pre>'+html.escape(json.dumps(c['factor'],ensure_ascii=False,indent=2))+'</pre><div class="views">')
        md += ['## '+sid,'','顺序：'+'→'.join(c['route_order'])+'；已冻结：'+str(c['frozen']),'']
        for label,img in zip(['目标视角','底盘全景','无新增障碍对照：目标','无新增障碍对照：全景'],c['images']):
            page.append('<div>'+label+'<img src="data:image/png;base64,'+base64.b64encode(Path(img['path']).read_bytes()).decode()+'"></div>')
            md.append('['+label+']('+img['path']+')')
        page.append('</div><p>审核：目标/把手能看清？底盘/障碍能看清？愿意尝试？异常或建议：</p></section>')
        md.append('')
    (output/'review.html').write_text(''.join(page))
    (output/'REVIEW.md').write_text(chr(10).join(md))
    print(json.dumps(dict(scenes=len(cards),errors=errors,preview_fields_redirected=len(fixups)),ensure_ascii=False))
    if errors:raise SystemExit(1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--restore',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.batch,a.restore,a.output)
