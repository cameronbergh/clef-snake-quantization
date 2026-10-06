"""Reproduce the frozen prospective analysis and figures; CPU only, no inference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from .audit import audit, sha


def analyze(dataset, output):
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise ValueError('Use a new output directory; preserve published analyses')
    evidence = audit(dataset)
    import numpy as np
    conditions = ['bf16', 'mlx-affine8-g64', 'mlx-affine4-g64']
    games = evidence['games']
    rows = []
    for seed_round in range(1, 31):
        block = [next(g for g in games if g['seed_round']==seed_round and g['variant']==v) for v in conditions]
        assert len({g['seed'] for g in block}) == 1
        rows.append({'seed_round':seed_round,'seed':block[0]['seed'],
                     'scores':[g['score'] for g in block],'steps':[g['steps'] for g in block],
                     'attempts':[g['attempts'] for g in block],'ends':[g['end_reason'] for g in block]})
    scores = np.asarray([r['scores'] for r in rows], dtype=np.int64)
    d = scores[:,1:]-scores[:,[0]]
    # One shared default-int64 sign matrix, in frozen Q8/Q4 comparison order.
    signs = np.random.Generator(np.random.PCG64(2026100602)).choice([-1,1],size=(100000,30))
    observed, null = np.abs(d.sum(axis=0)), np.abs(signs@d)
    p = (1+np.count_nonzero(null>=observed,axis=0))/100001
    for j in range(2):
        if np.all(d[:,j]==0):p[j]=1.0
    order = np.argsort(p,kind='stable')
    adjusted = np.empty(2)
    adjusted[order] = np.minimum(1,np.maximum.accumulate(p[order]*np.array([2,1])))
    indices = np.random.Generator(np.random.PCG64(2026100601)).integers(0,30,size=(100000,30))
    means = scores[indices].mean(axis=1)
    ci = np.quantile(means[:,1:]-means[:,[0]],[.025,.975],axis=0,method='linear')
    comparisons = []
    for j,v in enumerate(conditions[1:]):
        comparisons.append({'condition':v,'reference':'bf16','paired_mean_food_difference':float(d[:,j].mean()),
                            'marginal_95_percentile_ci':[float(ci[0,j]),float(ci[1,j])],
                            'wins':int((d[:,j]>0).sum()),'ties':int((d[:,j]==0).sum()),'losses':int((d[:,j]<0).sum()),
                            'raw_two_sided_sign_flip_p':float(p[j]),'holm_adjusted_p':float(adjusted[j]),
                            'satisfies_frozen_improvement_rule':bool(d[:,j].mean()>0 and adjusted[j]<=.05)})
    result = {'schema_version':1,'analysis_kind':'prospective_frozen_seed_level_v1',
              'dataset':'data/2026-10-06-kev4b-campaign-v1','paired_seeds':30,'conditions':conditions,
              'draws':100000,'sign_seed':2026100602,'bootstrap_seed':2026100601,'quantile_method':'linear',
              'sign_flip_assumption':'Sign exchangeability/symmetry; not an exact mean-only-null test without it.',
              'intervals_are_simultaneous':False,'comparisons':comparisons,'rows':rows,
              'technical_errors':[],'not_started_games':[],
              'interpretation':'Neither Q8 nor Q4 meets the pre-specified improvement criterion. This does not establish equivalence or refute the CLEF Q2 observation.',
              'implementation':'PCG64 choice([-1,1], size=(100000,30)), shared default-int64 signs; integer sums implement the exact absolute-mean threshold. PCG64 integers(0,30,size=(100000,30)), shared whole-row bootstrap.'}
    output.mkdir(parents=True)
    (output/'uncertainty.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors, labels = ['#374151','#2563eb','#c65b16'], ['BF16','Q8 affine','Q4 affine']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.hashsalt':'kev-campaign-v1-20261006'})
    for mobile in (False, True):
        fig, axes = plt.subplots(2 if mobile else 1,1 if mobile else 2,figsize=(7.2,10) if mobile else (13,6.6))
        fig.subplots_adjust(left=.13 if mobile else .07,right=.96,bottom=.20,top=.84,hspace=.55,wspace=.32)
        for row in scores:axes[0].plot(range(3),row,color='#9ca3af',alpha=.25,lw=.75,zorder=1)
        for j in range(3):
            # Position offsets identify all 30 seed rows; they are not additional observations.
            axes[0].scatter(j+np.linspace(-.10,.10,30),scores[:,j],color=colors[j],s=27,alpha=.75,zorder=2)
            mean = scores[:,j].mean()
            axes[0].scatter(j,mean,color=colors[j],marker='D',s=95,edgecolor='white',lw=1.2,zorder=3)
            axes[0].annotate(f'{mean:.1f}',(j,mean),xytext=(10,8),textcoords='offset points',weight='bold',color=colors[j])
        axes[0].set(xticks=range(3),xticklabels=labels,ylabel='Food before collision / 500-move horizon',ylim=(0,38),xlim=(-.3,2.4))
        axes[0].set_title('A · All 30 paired seed scores',loc='left',pad=12)
        axes[0].grid(axis='y',alpha=.18)
        for j,c in enumerate(comparisons):
            mean = c['paired_mean_food_difference'];lo,hi = c['marginal_95_percentile_ci']
            axes[1].errorbar(mean,j,xerr=[[mean-lo],[hi-mean]],fmt='o',color=colors[j+1],capsize=5,lw=2,ms=7)
            axes[1].annotate(f'{mean:+.2f}  [{lo:+.2f}, {hi:+.2f}]\nHolm p = {c["holm_adjusted_p"]:.3f}',
                             (mean,j),xytext=(0,14),textcoords='offset points',ha='center',fontsize=10)
        axes[1].axvline(0,color='#6b7280',ls='--',lw=1)
        axes[1].set(yticks=[0,1],yticklabels=labels[1:],ylim=(1.7,-.8),xlim=(-5.5,3.5),xlabel='Paired mean food difference versus BF16')
        axes[1].set_title('B · Effects and marginal 95% intervals',loc='left',pad=12)
        axes[1].grid(axis='x',alpha=.18)
        fig.suptitle('Kev 4B · prospective Snake campaign',y=.96,fontsize=17,weight='bold')
        caption = ('30 fresh paired seeds · 90 games · native MLX · unchanged FP32 head\n'
                   '100,000 paired bootstrap draws; two sign-flip tests with Holm correction.\n'
                   'Neither condition met the improvement criterion; equivalence is not established.\n'
                   'All 90 games collided before the cap. This does not refute CLEF Q2.\n'
                   'Dataset: 2026-10-06-kev4b-campaign-v1')
        fig.text(.13 if mobile else .07,.035,caption,fontsize=9.3,color='#374151',linespacing=1.6)
        name = 'kev-snake-results-mobile' if mobile else 'kev-snake-results'
        fig.savefig(output/(name+'.png'),dpi=190,metadata={'Title':'Kev 4B prospective Snake campaign'})
        fig.savefig(output/(name+'.svg'),metadata={'Date':None,'Title':'Kev 4B prospective Snake campaign'})
        plt.close(fig)
    provenance = {'schema_version':1,'python':sys.version.split()[0],'numpy':np.__version__,'matplotlib':matplotlib.__version__,
                  'dataset_manifest_sha256':sha(dataset/'SHA256SUMS'),'analysis_plan_sha256':sha(dataset/'analysis-plan.json'),
                  'analysis_source_sha256':sha(Path(__file__)),
                  'outputs_sha256':{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    assert not any(n==p or n.startswith(p+'.') for n in sys.modules for p in ('torch','mlx','transformers','kev'))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    result=analyze(args.dataset,args.output)
    result.pop('rows')
    print(json.dumps(result,indent=2))
