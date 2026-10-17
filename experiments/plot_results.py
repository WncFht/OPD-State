"""Plot the exact shared-parameter model; no synthetic LLM measurements.

Semantic ledger: each curve is the probability of action 1 for one initial
training-state weight w in {0.1,0.5,0.9}; color keeps that meaning in both
panels. Left: fixed state measures; right: all measures switch to w=0.7.
Geometry ledger: x=gradient update count (0..100); y=probability (0..1),
identical axes in both panels. All numerical values come from curves.csv.
"""
import csv
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SKILL=ROOT.parent/'.agents/skills/figure-viz'
sys.path.insert(0,str(SKILL/'assets/mpl'))
import fvstyle as fv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    fv.apply_rc('paper',fontsize=9)
    rows=list(csv.DictReader((ROOT/'results/curves.csv').open()))
    colors=fv.series_colors('paper',3)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.25),sharex=True,sharey=True)
    for ax,experiment,title in zip(axes,['projection','handoff'],
        ['(a) Different state weights','(b) Common state weight after switch']):
        for w,color in zip([.1,.5,.9],colors):
            r=[r for r in rows if r['experiment']==experiment and r['arm']==f'w={w}' and int(r['step'])<=100]
            ax.plot([float(v['x']) for v in r],[float(v['y']) for v in r],color=color,label=f'Initial w = {w}')
        ax.set_title(title)
        ax.set_xlabel('Gradient updates')
        ax.set_xlim(0,100); ax.set_ylim(0,1)
        ax.set_xticks([0,50,100]); ax.set_yticks([0,.25,.5,.75,1])
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Shared action probability')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.075),ncol=3,frameon=False)
    fig.text(.5,.02,'Exact constructed model; not a language-model training result.',ha='center',
             color=fv.NEUTRALS['vink'],fontsize='small')
    fig.tight_layout(rect=(0,.18,1,1))
    fig.canvas.draw()
    problems=fv.audit(fig)
    (ROOT/'results/figure-audit.json').write_text(json.dumps(problems,indent=2)+'\n')
    errors=[p for p in problems if fv.problem_severity(p)=='error']
    if errors:
        raise RuntimeError(errors)
    for ext in ['png','pdf','svg']:
        fig.savefig(ROOT/f'figures/2026-10-17-状态权重与切换平台.{ext}',dpi=200)
    svg=ROOT/'figures/2026-10-17-状态权重与切换平台.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(json.dumps(problems,indent=2))

if __name__=='__main__':
    main()
