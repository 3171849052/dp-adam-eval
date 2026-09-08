"""Per-seed summaries, sample SD and paired contrasts; no significance tests."""
import pandas as pd
from expv1.common import METHODS, write_csv
from expv1.validate_expv1 import cli, validate, load

METRICS = ('relmse_noisy','relmse_filtered','cosine_noisy','cosine_filtered',
           'signal_retention','noise_retention','snr_gain_db','mse_reduction')


def aggregate(frame, groups):
    numeric = [c for c in frame.select_dtypes('number').columns if c not in ['seed',*groups]]
    rows=[]
    for keys, part in frame.groupby(groups):
        keys = keys if isinstance(keys,tuple) else (keys,)
        row=dict(zip(groups,keys))
        for col in numeric:
            v=part[col].dropna()
            row[col+'_mean']=float(v.mean()) if len(v) else None
            row[col+'_sd']=float(v.std(ddof=1)) if len(v)>1 else None
        rows.append(row)
    return rows


def summarize(c,runs,output):
    validate(c,runs,output)
    rows, layers=[] ,[]
    for seed in c['seeds']:
        for method in METHODS:
            root=runs/f'seed{seed}'/method
            summary=load(root/'summary.json')
            total=summary['completed_steps']
            tr=pd.read_csv(root/'train_metrics.csv')
            late=tr[tr.step+1>total/2]
            row={k:v for k,v in summary.items() if k not in ('fingerprint','final_model_hash')}
            row.update({f'late_{m}':float(late[m].median()) for m in METRICS})
            rows.append(row)
            la=pd.read_csv(root/'layer_metrics.csv')
            for layer, frame in la[la.step+1>total/2].groupby('layer'):
                lr=dict(seed=seed,method=method,layer=layer)
                for col in ('kappa','H_q50','H_q10','H_q90',*METRICS):
                    lr[col]=float(frame[col].median()) if frame[col].notna().any() else None
                lr['relmse_improvement']=float((frame.relmse_noisy-frame.relmse_filtered).median())
                lr['cosine_improvement']=float((frame.cosine_filtered-frame.cosine_noisy).median())
                layers.append(lr)
    frame=pd.DataFrame(rows)
    write_csv(output/'summary_runs.csv',rows)
    write_csv(output/'summary_methods.csv',aggregate(frame,['method']))
    write_csv(output/'summary_layers.csv',layers)
    write_csv(output/'summary_filter.csv',aggregate(pd.DataFrame(layers),['method','layer']))
    contrasts=[]
    cols=['final_accuracy','best_accuracy','late_mean_accuracy','accuracy_auc','final_test_loss',*[f'late_{m}' for m in METRICS]]
    for a,b in [('dp_scalar_wiener','dp_sgd'),('dp_fisher_wiener','dp_sgd'),('dp_fisher_wiener','dp_scalar_wiener')]:
        fa,fb=[frame[frame.method==m].set_index('seed') for m in (a,b)]
        for col in cols:
            delta=fa[col]-fb[col]
            direction='negative difference improves' if 'relmse' in col or 'loss' in col else 'positive difference improves'
            if 'retention' in col:
                direction='descriptive; interpret signal and noise jointly'
            for seed,value in delta.items():
                contrasts.append(dict(contrast=f'{a} - {b}',metric=col,seed=seed,difference=float(value),
                    mean=None,sd=None,improvement_direction=direction))
            contrasts.append(dict(contrast=f'{a} - {b}',metric=col,seed='aggregate',difference=None,
                mean=float(delta.mean()),sd=float(delta.std(ddof=1)) if len(delta)>1 else None,improvement_direction=direction))
    write_csv(output/'paired_contrasts.csv',contrasts)
    print('Summary tables generated')


if __name__=='__main__':
    summarize(*cli())
