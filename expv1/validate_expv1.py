"""Strict artifact and paired-stream validation."""
import argparse
import json
import math
from pathlib import Path
import pandas as pd
from expv1.common import ROOT, METHODS, LAYERS, read_config, fingerprint, provenance, require_pinned, output_path, save_json


def load(path):
    return json.loads(Path(path).read_text())


def cli():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--runs', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    return read_config(a.config), output_path(a.runs), output_path(a.output)


def validate(c, runs, output, require_tests=True):
    runs, output = output_path(runs), output_path(output)
    fp, current = fingerprint(c), provenance()
    require_pinned(current, c['smoke'])
    evidence = ROOT/'runs/test_evidence.json'
    if require_tests:
        proof = load(evidence)
        assert proof['passed'] and proof['provenance'] == current, 'Run current isolation tests first'
    count = 0
    for seed in c['seeds']:
        records = {}
        for method in METHODS:
            root = runs/f'seed{seed}'/method
            cfg, meta, summary, pairing = [load(root/f'{f}.json') for f in ('config','metadata','summary','pairing')]
            assert cfg == c and meta['fingerprint'] == summary['fingerprint'] == fp
            assert meta['provenance'] == current and meta['complete']
            require_pinned(meta['provenance'], c['smoke'])
            assert output_path(meta['output']) == root.resolve()
            total = (c['train_subset'] or 60000)//c['batch_size']*c['epochs']
            assert meta['total_steps'] == summary['completed_steps'] == total
            assert meta['sample_rate'] == c['batch_size']/(c['train_subset'] or 60000)
            assert summary['parameters_finite'] and all(r['parameters_finite'] for r in pairing['private'])
            assert all(math.isfinite(v) for v in summary.values() if isinstance(v,(int,float)))
            assert 0 < summary['epsilon_spent'] <= c['epsilon']+.02
            assert meta['diagnostics_enabled'] and meta['eigen_budget'] is None
            frames = {n: pd.read_csv(root/f'{n}_metrics.csv') for n in ('train','layer','eigenbin','refresh')}
            tr, la, eb, re = [frames[n] for n in ('train','layer','eigenbin','refresh')]
            assert tr.step.tolist() == list(range(total))
            assert len(la) == total*4 and set(la.layer)==set(LAYERS)
            assert not la.duplicated(['step','layer']).any()
            for name, frame in frames.items():
                if frame.empty:
                    continue
                assert set(frame.config_fingerprint)=={fp} and set(frame.method)=={method} and set(frame.seed)=={seed}
                for col in frame.select_dtypes(include='number'):
                    missing = frame[col].isna()
                    allowed = col in ('test_loss','test_accuracy') or (method=='dp_sgd' and col=='kappa')
                    assert allowed or not missing.any(), (method,name,col,'missing')
                    assert frame.loc[~missing,col].map(math.isfinite).all(), (name,col)
                for col in frame.columns:
                    if col.startswith('H_'):
                        assert frame[col].between(0,1).all()
                    if col in ('signal_retention','noise_retention'):
                        assert (frame[col]>=0).all()
            expected_eval = [s for s in range(total) if (s+1)%c['eval_interval']==0 or s+1==total]
            assert tr.loc[tr.test_accuracy.notna(),'step'].tolist()==expected_eval
            refresh_steps = list(range(0,total,c['K'])) if method!='dp_sgd' else []
            assert re.step.tolist()==refresh_steps
            assert summary['number_of_refreshes']==len(refresh_steps)
            assert [r['step'] for r in pairing['synthetic']]==refresh_steps
            assert [r['step'] for r in pairing['private']]==list(range(total))
            if method=='dp_sgd':
                assert summary['active_state_bytes']==summary['total_refresh_time']==0
                for frame in (tr,la):
                    assert (frame.snr_gain_db==0).all() and (frame.mse_reduction==0).all()
                    assert (frame.signal_retention==1).all() and (frame.noise_retention==1).all()
                    assert (frame.relmse_filtered==frame.relmse_noisy).all()
                    assert (frame.cosine_filtered==frame.cosine_noisy).all()
            if method=='dp_fisher_wiener':
                assert len(eb)==len(refresh_steps)*4*10
                assert set(eb.step)==set(refresh_steps) and set(eb.layer)==set(LAYERS)
                for _, frame in eb.groupby(['step','layer']):
                    assert sorted(frame.bin.tolist())==list(range(10))
                    assert frame['count'].max()-frame['count'].min()<=1
            else:
                assert eb.empty
            records[method]=(meta,pairing)
            count += 1
        base, pair = records['dp_sgd']
        for meta, pairing in records.values():
            for k in ('noise_multiplier','sample_rate','total_steps','initial_model_hash'):
                assert meta[k]==base[k]
            for a,b in zip(pair['private'],pairing['private']):
                for k in ('batch_indices','noise_rng_before','noise_rng_after'):
                    assert a[k]==b[k], (seed,k)
        assert records['dp_scalar_wiener'][1]['synthetic']==records['dp_fisher_wiener'][1]['synthetic']
    result = dict(passed=True, runs=count, fingerprint=fp, provenance=current,
                  diagnostics_isolation_tests=load(evidence) if require_tests else 'internal test')
    save_json(output/'validation.json', result)
    return result


if __name__=='__main__':
    c,r,o = cli()
    try:
        result = validate(c,r,o)
    except Exception as exc:
        save_json(o/'validation.json', dict(passed=False,error=str(exc)))
        raise
    print(f'Validation passed: {result["runs"]} runs')
