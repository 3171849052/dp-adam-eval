"""Analyze identical checkpoints with fixed Private/Public/three Pink probes."""
import csv
import json
import shutil
import tempfile
import torch
from common import (parse, run_root, fingerprint, provenance, save_torch, save_json,
                    probes, device, set_seed, LAYERS)
from fisher_utils import collect, empirical_summary, empirical_inner
from kfac_utils import spectrum
from metrics import (alignment, kfac_alignment, matrix_alignment,
                     diagonal_metrics, vector_alignment)


def analyze(c):
    root = run_root(c)
    manifest_path = root / 'checkpoints/manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError('Run exp1/train_reference.py with this config first')
    manifest = json.loads(manifest_path.read_text())
    if manifest['fingerprint'] != fingerprint(c) or manifest['upstream'] != provenance():
        raise ValueError('Trajectory config/upstream source mismatch; use original code/config or a fresh run')
    sources = probes(c)
    set_seed(c['seed'])
    results = root / 'results'
    results.mkdir(exist_ok=True)
    # A failed rerun must not leave a stale completion marker for partial CSVs.
    (results / 'analysis.json').unlink(missing_ok=True)
    rows = []
    for checkpoint in manifest['checkpoints']:
        percent = checkpoint['percent']
        state = torch.load(root / 'checkpoints' / checkpoint['file'], weights_only=True)
        if state['fingerprint'] != fingerprint(c) or state['step'] != checkpoint['step']:
            raise ValueError('Checkpoint metadata mismatch')
        full = percent in (0, 50, 100)
        def record(layer, source, family, values):
            for metric, value in values.items():
                rows.append({'checkpoint': percent, 'step': checkpoint['step'], 'layer': layer,
                             'source': source, 'family': family, 'metric': metric, 'value': value})
        with tempfile.TemporaryDirectory(prefix='gram_', dir=results) as temp:
            private, private_paths = collect(state['model'], sources['private'], c, device(c),
                                             f'{temp}/private' if full else None)
            private_full = {layer: empirical_summary(private_paths[layer], c['gram_chunk'])
                            for layer in LAYERS} if full else {}
            for source, samples in sources.items():
                print(f'Analyzing checkpoint={percent}% source={source}', flush=True)
                if source == 'private':
                    stats, paths, summaries = private, private_paths, private_full
                else:
                    stats, paths = collect(state['model'], samples, c, device(c), f'{temp}/source' if full else None)
                    summaries = {layer: empirical_summary(paths[layer], c['gram_chunk'])
                                 for layer in LAYERS} if full else {}
                for layer in LAYERS:
                    s, p = stats[layer], private[layer]
                    record(layer, source, 'diagonal', diagonal_metrics(s['v_direct'], p['v_direct'], c['eps_num']))
                    for variant in ('raw', 'repo'):
                        sf, pf = s['factors'][variant], p['factors'][variant]
                        record(layer, source, f'kfac_{variant}', kfac_alignment(sf, pf))
                        for factor in ('A', 'G'):
                            record(layer, source, f'factor_{factor}_{variant}', matrix_alignment(sf[factor], pf[factor]))
                        # Approximation within each source, plus virtual preconditioning
                        # of Private using each source's KFAC-derived diagonal.
                        record(layer, source, f'kfac_direct_diagonal_{variant}',
                               vector_alignment(s[f'v_kfac_{variant}'], s['v_direct']))
                        record(layer, source, f'kfac_diagonal_{variant}',
                               diagonal_metrics(s[f'v_kfac_{variant}'], p['v_direct'], c['eps_num']))
                    if full:
                        summary, oracle = summaries[layer], private_full[layer]
                        inner = oracle['norm_sq'] if source == 'private' else empirical_inner(
                            paths[layer], private_paths[layer], c['gram_chunk'])
                        record(layer, source, 'empirical', alignment(inner, summary['norm_sq'], oracle['norm_sq'],
                                                                    summary['trace'], oracle['trace']))
                        s['spectra'] = {'empirical': summary['spectrum'],
                                        'kfac_raw': spectrum(s['factors']['raw']),
                                        'kfac_repo': spectrum(s['factors']['repo'])}
                        s['empirical_rank_tolerance'] = summary['rank_tolerance']
                save_torch(results / f'checkpoint_{percent:03d}' / f'{source}.pt',
                           {'checkpoint': checkpoint, 'source': source, 'm_diag': c['m_diag'],
                            'm_full': c['m_full'] if full else None, 'layers': stats})
                if source != 'private' and full:
                    shutil.rmtree(f'{temp}/source')
        # Atomic partial progress; plots require the final metadata file below.
        output = results / 'metrics.csv'
        tmp = output.with_suffix('.csv.tmp')
        with tmp.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['checkpoint', 'step', 'layer', 'source', 'family', 'metric', 'value'])
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(output)
    save_json(results / 'analysis.json', {'config': c, 'fingerprint': fingerprint(c), 'upstream': provenance(),
        'rows': len(rows), 'parameter_order': 'output-major augmented rows: weight row then bias',
        'raw_kfac': 'upstream spatial-averaged covariances, sum-loss deltas, eps=0',
        'repo_kfac': 'mean-loss equivalent at configured batch_size; covariance eps=1e-5',
        'inverse_sqrt_damping_not_in_fisher': 0.001,
        'full_subset': 'first m_full of each fixed diagnostic probe',
        'synthetic_generation_device': 'cpu',
        'diagnostic_tf32': False,
        'note': 'Private diagnostic statistics are unnoised research diagnostics.'})
    print(f'Saved {len(rows)} metric rows to {results}', flush=True)


if __name__ == '__main__':
    analyze(parse(__doc__))
