"""Descriptive final counts with unrun/unknown coverage retained."""
import argparse,csv,os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/share/personal/chensiyu/haokaijiang/MobiWAM/cache/obc-pi05-v1/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();out=a.run/'paper-evidence'
 rows=list(csv.DictReader((out/'final-method-summary.csv').open()));fig,axes=plt.subplots(2,1,figsize=(10,6),gridspec_kw={'height_ratios':[1.4,1.]})
 for ax,scope,methods in zip(axes,['paired_lookup','actual_online'],[['fixedE','fixedD','fixedA','train-best-fixed','geometry','ridge','Linear','MLP','oracle'],['MLP','geometry','train-best-fixed']]):
  data={x['method']:x for x in rows if x['evaluation']==scope};x=np.arange(len(methods));success=np.array([int(data[m]['safe_success']) for m in methods]);unknown=np.array([int(data[m]['unknown_safe_success']) for m in methods]);total=np.array([int(data[m]['denominator']) for m in methods]);failure=total-success-unknown
  ax.bar(x,success,color='#287d57',label='Safety-qualified success');ax.bar(x,failure,bottom=success,color='#b95b50',label='Known failure / X')
  ax.bar(x,unknown,bottom=success+failure,color='#dedede',edgecolor='#888888',hatch='//',label='Unrun / unresolved')
  ax.set_xticks(x,methods,rotation=20,ha='right');ax.set_yticks([0,1,2]);ax.set_ylim(0,2.55);ax.set_ylabel('Parents, full denominator')
  ax.set_title('Same frozen paired outcomes' if scope=='paired_lookup' else 'Separately executed real online episodes',fontsize=10)
  for i,m in enumerate(methods):
   if int(data[m]['unrun']):ax.text(i,2.08,str(data[m]['unrun'])+' unrun',ha='center',fontsize=8)
 axes[0].legend(loc='upper left',ncol=3,fontsize=8)
 fig.suptitle('Two prospective known-development parents, one family; partial E/D scope',fontsize=12)
 fig.tight_layout();fig.savefig(out/'final-method-counts.png',dpi=180);fig.savefig(out/'final-method-counts.svg');plt.close(fig)
if __name__=='__main__':main()
