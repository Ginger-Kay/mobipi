"""A read-only viewer linking every original and its labeled derivative."""
import argparse,html,json
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import quote

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run;out=r/'paper-evidence';rows=[]
 for receipt in sorted((r/'episodes').glob('*/slot-*/engineering-attempt-*/completed.json')):
  q=json.loads(receipt.read_text());attempt=Path(q['attempt']);audit=attempt/'sprint-safety-audit.json';safe=json.loads(audit.read_text()) if audit.exists() else {}
  tag=receipt.parents[2].name;key=tag+'-slot'+str(q['slot']);videos=[]
  for camera,filename,label in [('main','original.mp4','原始侧视'),('panorama','panoramic.mp4','原始全景')]:
   original=attempt/filename;copy=out/'review-copies'/(key+'-'+camera+'.mp4')
   for path,title in [(original,label),(copy,'标注'+label[2:])]:
    if path.exists():videos.append(dict(url=quote('../'+str(path.relative_to(r))),label=title))
  rows.append(dict(tag=tag,purpose=q.get('purpose','policy-dev'),slot=q['slot'],task=q['task'],route=q['route'],status=q['status'],
   native=q['native_success'],semantic=q.get('route_semantics_pass',True),safe=safe.get('safety_qualified_success'),steps=q['steps'],videos=videos))
 data=json.dumps(rows,ensure_ascii=False).replace('<','\\u003c')
 page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>PI05 真实执行视频审阅</title>
<style>body{font-family:system-ui;margin:28px;max-width:1250px;background:#fafafa;color:#18202b}h1{font-size:24px}p{line-height:1.6}select{padding:8px;margin:6px}article{background:white;padding:18px;margin:14px 0;border:1px solid #ddd;border-radius:8px}header{font-weight:600}small{display:block;margin:6px 0;color:#465366}button,a{margin:8px 8px 8px 0}video{display:block;max-width:100%;width:760px;background:#111}button{padding:6px 10px}b{font-weight:500}#count{padding:8px}</style>
<h1>π0.5 真实执行视频审阅</h1><p>全部原片与标注观看副本。每条记录保留失败、路线语义与安全审计状态；人审均待审阅。点击按钮才加载视频。原片以记录时的物理执行为准，标注副本为 1× 播放，初态来自同一 Source 恢复。</p>
<label>任务 <select id="task"><option value="">全部</option><option>CloseDrawer</option><option>CloseSingleDoor</option></select></label>
<label>用途 <select id="purpose"><option value="">全部</option><option value="policy-dev">开发</option><option value="paired">配对</option><option value="online">实际在线</option></select></label>
<label>路线 <select id="route"><option value="">全部</option><option>E</option><option>D</option><option>A</option></select></label><div id="count"></div><main id="list"></main>
<script>const rows=DATA;const value=id=>document.getElementById(id).value;const verdict=v=>v===null?'待安全审计':v?'安全合格成功':'未取得安全合格成功';
function render(){const selected=rows.filter(r=>(!value('task')||r.task===value('task'))&&(!value('purpose')||r.purpose===value('purpose'))&&(!value('route')||r.route===value('route')));document.getElementById('count').textContent=selected.length+' 条记录（保留全部分母）';const root=document.getElementById('list');root.replaceChildren();for(const r of selected){const a=document.createElement('article');const h=document.createElement('header');h.textContent=r.tag+' · slot '+r.slot+' · '+r.task+' / '+r.route;a.append(h);const info=document.createElement('small');info.textContent=r.status+' | 原生成功 '+r.native+' | 路线语义 '+r.semantic+' | '+verdict(r.safe)+' | '+r.steps+' 控制步';a.append(info);const video=document.createElement('video');video.controls=true;video.preload='none';for(const v of r.videos){const button=document.createElement('button');button.textContent=v.label;button.onclick=()=>{video.src=v.url;video.load()};a.append(button);const link=document.createElement('a');link.href=v.url;link.textContent='打开文件';link.target='_blank';a.append(link)}a.append(video);root.append(a)}}for(const id of ['task','purpose','route'])document.getElementById(id).onchange=render;render();</script></html>'''.replace('DATA',data)
 (out/'review.html').write_text(page)
 (out/'review-page-status.json').write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),episodes=len(rows),human_review='pending',read_only=True,links='relative to mirrored full run; exact absolute media paths in all-native-videos.csv'),indent=2)+'\n')
 print('Read-only viewer entries',len(rows),flush=True)
if __name__=='__main__':main()
