"""Standalone figures and measured method facts from immutable run artifacts."""
import argparse,csv,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def read(p):return json.loads(Path(p).read_text())
def save(fig,out,name):
 fig.savefig(out/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();r=a.run;out=r/'paper-evidence';out.mkdir(exist_ok=True)
 fig,ax=plt.subplots(figsize=(7,3.2))
 for name,color,label in [('integrated-fit1','#cc7d22','Original26 configuration groups (nonconformant)'),('integrated-corrective-fit2','#1565a8','Corrective16 original parents (approved)')]:
  rows=[json.loads(x) for x in (r/'policy'/name/'metrics.jsonl').read_text().splitlines()];loss=np.array([x['loss'] for x in rows]);steps=np.array([x['step'] for x in rows]);med=[np.median(loss[max(0,i-24):i+1]) for i in range(len(loss))];ax.plot(steps,med,color=color,label=label)
 ax.set(xlabel='Optimizer updates',ylabel='Training loss (trailing25 median)',yscale='log',title='Training diagnostics;2000 updates do not establish convergence');ax.legend(fontsize=7);fig.tight_layout();save(fig,out,'policy-fit-diagnostics')
 # Coordinates are the predeclared design; no outcome colors or selections.
 rows=list(csv.DictReader((r/'design/start-coverage.csv').open()));fig,axes=plt.subplots(1,2,figsize=(8,3.4),sharex=True,sharey=True)
 for ax,task in zip(axes,['CloseDrawer','CloseSingleDoor']):
  seen=set()
  for row in rows:
   if row['task']!=task:continue
   key=(float(row['distance_offset_m']),float(row['lateral_offset_m']),int(row['tier']))
   if key in seen:continue
   if task=='CloseDrawer' and key[:2]==(0.,0.) and key[2]==2:continue
   seen.add(key);ax.scatter(key[0],key[1],s=70,marker='o' if key[2]==1 else 's',color='#1565a8' if key[2]==1 else '#d17a25');ax.annotate(('slots1,6' if task=='CloseDrawer' and key[:2]==(0.,0.) else 'slot'+row['slot']),(key[0],key[1]),xytext=(4,5),textcoords='offset points',fontsize=8)
  ax.set(title=task,xlabel='Outward offset from anchor (m)');ax.grid(alpha=.2)
 axes[0].set_ylabel('Furniture-local lateral offset (m)');fig.suptitle('Static legal starts:60 first tier,30 additional;30 second-tier slots rejected\nYaw0 and one neutral arm; opening variation overlaps in this2D projection',fontsize=9);fig.tight_layout();save(fig,out,'start-coverage')
 fig,ax=plt.subplots(figsize=(10,4.4));ax.set(xlim=(0,10),ylim=(0,4.4));ax.axis('off')
 def box(x,y,w,h,label,color):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04',facecolor=color,edgecolor='#666'));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=6.8)
 def arrow(x,y,u,v):ax.annotate('',(u,v),(x,y),arrowprops=dict(arrowstyle='->',color='#555'))
 box(.1,2.65,2.15,1.35,'61 qualified demonstrations\n41 reference +20 human\n16 original parents\n2 human templates','#e9f1f8')
 box(2.8,2.65,2.2,1.35,'Approved corrective LoRA\nOriginalfit2 initialization\n2000 updates; batch8\n~0.32 mean window exposures','#e9f1f8');arrow(2.3,3.3,2.75,3.3)
 box(5.6,2.65,2.05,1.35,'Fixed development roster\nOld / corrective1000 /2000\n6 slots each; failures retained\nFreeze one policy + guards','#f1f0f7');arrow(5.05,3.3,5.55,3.3)
 box(8.15,2.65,1.65,1.35,'20 OBC parents\n12train /4dev /4eval\n3 starts, resource gate\n2 fixture families','#f1f0f7');arrow(7.7,3.3,8.1,3.3)
 box(5.6,.2,2.05,1.45,'Real E / D / natural A\nTrain/dev first\nNative fields + masks\nOriginal numerical protection','#edf4e9');arrow(8.95,2.6,7.7,1.65)
 box(2.8,.2,2.2,1.45,'Fixed OBC recipe\nCLIP1024 + geometry21\n+ route onehot3\nMLP / Linear final2000\nRidge21 alpha1; train only','#edf4e9');arrow(5.55,.9,5.05,.9)
 box(.1,.2,2.15,1.45,'Freeze evaluation before outcomes\nPaired lookup + fresh online\nParent cluster; all failures/media\nResearch claim review pending','#fff2de');arrow(2.75,.9,2.3,.9)
 ax.text(.1,2.1,'Reference results remain demonstration qualifications; only current frozen-policy outcomes label OBC.',fontsize=8);save(fig,out,'data-flow')
 if (out/'all-primary-outcomes.jsonl').exists():
  records=[json.loads(x) for x in (out/'all-primary-outcomes.jsonl').read_text().splitlines()];fig,axes=plt.subplots(1,2,figsize=(9,3.3))
  for route in 'EDA':
   rows=[x for x in records if x['route']==route];times=[x['terminal_duration_s'] for x in rows if x['terminal_duration_s'] is not None];paths=[x['base_path_m'] for x in rows if x['base_path_m'] is not None]
   if times:axes[0].plot(np.sort(times),np.arange(1,len(times)+1)/len(times),label=route)
   if paths:axes[1].plot(np.sort(paths),np.arange(1,len(paths)+1)/len(paths),label=route)
  axes[0].set(xlabel='Actual terminal duration (s; includes failures)',ylabel='Empirical cumulative fraction');axes[1].set(xlabel='Actual total base path (m)',ylabel='Empirical cumulative fraction')
  for ax in axes:ax.legend();ax.grid(alpha=.2)
  fig.tight_layout();save(fig,out,'termination-cost-distributions')
  summary=read(out/'comparison-summary.json');fig,ax=plt.subplots(figsize=(9,3.4));series=summary['summary']['real_online'];methods=list(series);lo=np.array([series[m]['parent_equal_safe_opportunity_lower'] for m in methods]);hi=np.array([series[m]['parent_equal_safe_opportunity_upper'] for m in methods]);ax.bar(methods,lo,color='#1565a8');ax.bar(methods,hi-lo,bottom=lo,color='#dcdcdc',hatch='//',label='Unknown/pending opportunity bound');ax.set(ylim=(0,1),ylabel='Parent equal success credit per planned configuration',title='True online outcomes; bounds describe missingness, not population confidence');ax.legend(fontsize=7);fig.tight_layout();save(fig,out,'online-parent-comparison')
 facts='''# PI05-DATA-v1 method facts\n\nAll new artifacts bind this run and preserve historical originals.61 admitted demonstrations comprise41 reference and20 human episodes,16 original parents,26 configuration groups and2 human template ancestors;49,946 valid10-step windows. Source→episode→window equal sampling, train-only hierarchical norm. The original26-group-weight fit is retained as nonconformant; the explicitly authorized correction starts from originalfit2 with a fresh optimizer. Both executed2000 updates; only oldfit2 and corrective1000/2000 are final-policy candidates. Loss and~0.32 mean window exposures are not convergence or capability evidence.\n\nPolicy is existing pi05_base with dual LoRA rank16/32 and action/time projections,52,153,376 trainable parameters; vision and original Transformer leaves frozen. Original43 frozen parameter leaves unchanged. Three real cameras,22 state dimensions padded32,8 nominal action dimensions padded32;10-step query-relative EEF commands, execute5,20Hz. Command conversion composes synchronized moving-base controller origin with actual OSC commanded goals, rather than substituting future achieved displacement. Native positive desired-mode accumulation and local delta mode are retained. No future state is an input.\n\n20 core OBC parent environments split12/4/4, each task6/2/2, seed17. All starts follow their original parent split. Known-development and shared human templates are disclosed; only2 fixture/layout families.90 legal predeclared starts,60 first-tier plus30 additional, with30 rejected extra slots retained. Posed-start distribution, yaw0, one fixed neutral arm; not natural random deployment or a full factorial design.90 actual states and input bindings verified; first-tier3 raw RGB and CLIP/proprio contexts distinct per parent. E zero-duration geometric prefixes with the same arm posture can have unchanged21 geometry fields.\n\nExternal harness controls mobility: E locks base; D stows/navigates/settles before fresh policy observations then locks base; natural A uses fixed-policy EEF intent with whole-body constrained allocation. No artificial minimum displacement/contact success gate. Original15mrad strict joint buffer,0.5mm swept clearance,1e-8 QP tolerance, forbidden contacts, palm and speed limits remain. Base inverse maps installed native raw-XY swap/rotation with actual-goal error5.4e-8. Human/reference outcomes never become pi05 result labels.\n\nAll routes use300sim seconds/2700wall seconds at20Hz. Native task success and safety stay separate. Progress=clip(1-native_opening,0,1); initial opening and improvement separately recorded. Actual failures and protective stops are eligible by reliable field. Joint/QP/sweep stops do not mean actual collision. Unknowns and X retain masks; duration is actual terminal time, including failures and right-censored compute/horizon stops.\n\nFrozen CLIP context1024 +21 pre-outcome geometry +3 route onehot. Geometric fields describe initial and proposed stow/navigation/reach prefixes, not future closed-loop manipulation; view fraction ignores occlusion. Train-only parent/config/route hierarchy with per-head mask renormalization, path/2m and time/300s. MLP1048→32ReLU→5 and Linear1048→5 each one fixed final2000 fit per selected tier; ridge21 alpha1. FP32,seed17,AdamW3e-4,cosine→1e-5,wd.05/bias0,clip1. Single-class collision is not fitted or ranked. Exact physical-unit selection windows are.05 success,.05 supported collision,.05 progress,.02m path,1s terminal duration,E<D<A.\n\nResource gate uses timing/storage only before any evaluation result. Final model/predictions/choices are frozen before evaluation. Paired lookup and true fresh-policy online are distinct, with original parent as statistical cluster. Missing/X denominators retained, common-safe-success cost comparisons only. Equal observed success rates do not establish equivalence or benefit. All primary and panoramic movies are native recordings, full1x20Hz copies only annotated, no state rerender/generative media. Human review and scientific claims remain Research pending; formal readiness/Gate unchanged.\n'''
 (out/'method-facts.md').write_text('Updated '+datetime.now(timezone.utc).isoformat()+'\n\n'+facts)
 print('PAPER FIGURES AND MEASURED METHOD FACTS',flush=True)
if __name__=='__main__':main()
