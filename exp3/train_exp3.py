"""Run one or all paired methods. Full experiments require explicit invocation."""
import argparse
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from exp3.common import (ROOT, METHODS, LAYERS, read_config, fingerprint, provenance,
                         save_json, write_csv, datasets, set_seed, SimpleCNN, RNGStream, digest)
from exp3.preconditioners import synthetic_samples, refresh, apply, state_bytes
from exp3.geometry import diagnose, oracle_compare, stale
from exp3.cost import CostTracker
from exp3.audit_upstream import require_pinned
from exp3.metrics import before_clip, after_noise, norm_metrics
from dp_kfac.privacy import clip_and_noise_gradients, _compute_per_sample_norms_squared


class Indexed(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        x, y = self.data[i]
        return x, y, i


def make_optimizer(model, c):
    if c["optimizer"] != "SGD" or c["learning_rate"] != .1 or c["momentum"] != 0:
        raise ValueError("Exp3 requires plain SGD(lr=0.1, momentum=0)")
    return torch.optim.SGD(model.parameters(), lr=.1, momentum=0)


def private_update(model, active, optimizer, rng, sigma, c, b):
    # The order is shared by all methods, including KFAC.
    apply(model, active)
    row = before_clip(model, c, b)
    row.update(norm_metrics(_compute_per_sample_norms_squared(list(model.parameters()), b, next(model.parameters()).device).sqrt(), c["eps_num"]))
    audit = {"noise_rng_before": rng.audit()}
    with rng.use():
        clip_and_noise_gradients(model, sigma, c["max_grad_norm"], b, store_summed_grad=True)
    audit["noise_rng_after"] = rng.audit()
    row.update(after_noise(model, sigma, c, b))
    row["diagnostic_snr"] = row["clipped_aggregate_norm"]/(row["expected_noise_norm"]+c["eps_num"])
    optimizer.step()
    return row, audit


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    loss = correct = count = 0
    for x, y in loader:
        y = y.to(dev)
        out = model(x.to(dev))
        loss += float(F.cross_entropy(out, y, reduction="sum"))
        correct += int((out.argmax(1) == y).sum())
        count += len(y)
    model.train()
    return dict(test_loss=loss/count, test_accuracy=correct/count)


def train(c, seed, method, output, data_override=None):
    if method == "dp_kfc_pink_matched":
        method = "dp_kfc"
    if method not in METHODS or seed not in c["seeds"]:
        raise ValueError("Unknown method/seed")
    source_provenance = provenance()
    require_pinned(source_provenance, c["smoke"])
    root = Path(output).resolve() / f"seed{seed}" / method
    root.mkdir(parents=True, exist_ok=False)
    save_json(root / "config.json", c)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if c["device"] == "auto" else torch.device(c["device"])
    torch.set_num_threads(c["threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    source, test = datasets(c) if data_override is None else data_override
    n = c["train_subset"] or len(source)
    if not c["batch_size"] <= n <= len(source) or not 1 <= c["M_oracle"] <= n:
        raise ValueError("Invalid training/oracle subset")
    data = Subset(source, range(n))
    if c["test_subset"] is not None:
        if not 1 <= c["test_subset"] <= len(test):
            raise ValueError("Invalid test subset")
        test = Subset(test, range(c["test_subset"]))
    loader = DataLoader(Indexed(data), batch_size=c["batch_size"], shuffle=True, drop_last=True,
                        num_workers=0, generator=torch.Generator().manual_seed(seed+1))
    test_loader = DataLoader(test, batch_size=c["batch_size"], shuffle=False, num_workers=0,
                             generator=torch.Generator().manual_seed(seed+2))
    total = len(loader)*c["epochs"]
    q = c["batch_size"]/n
    sigma = get_noise_multiplier(target_epsilon=c["epsilon"], target_delta=c["delta"], sample_rate=q, steps=total, accountant="rdp")
    set_seed(seed)
    model = GradSampleModule(SimpleCNN().to(dev), loss_reduction="sum")
    optimizer = make_optimizer(model, c)
    accountant = RDPAccountant()
    syn_rng, noise_rng = RNGStream(seed+3, dev), RNGStream(seed+4, dev)
    stale_rng = RNGStream(seed+c["stale_seed_offset"], dev)
    # Deliberately independent of the experimental seed.
    ids = torch.randperm(n, generator=torch.Generator().manual_seed(c["oracle_seed"]))[:c["M_oracle"]]
    meta = dict(seed=seed, method=method, mechanism="dp_kfc_pink_matched" if method == "dp_kfc" else method,
                fingerprint=fingerprint(c), provenance=source_provenance, device=str(dev), torch=str(torch.__version__),
                noise_multiplier=sigma, sample_rate=q, total_steps=total, initial_model_hash=digest(model.parameters()),
                oracle_indices=ids.tolist(), oracle_seed=c["oracle_seed"],
                rng_seeds=dict(init=seed, loader=seed+1, test=seed+2, synthetic=seed+3, noise=seed+4,
                               stale=seed+c["stale_seed_offset"]),
                parameter_order=list(dict(model._module.named_parameters())),
                optimizer=dict(name="SGD", lr=.1, momentum=0), accountant="rdp",
                sampling="shuffle/drop_last; unchanged exp2/upstream RDP convention (not Poisson)",
                clipping="upstream min(1,C/(norm+1e-6)) after preconditioning",
                diagnostics="private, non-DP diagnostic outputs; separate model, never fed to training",
                spectrum="FP64 Gram; positive eig > max(M,d)*FP64_eps*max_eig; no dxd Fisher",
                state_bytes_definition="active diagonal P or active KFAC inverse roots; discarded factors excluded",
                kfac_loss_reduction="mean", covariance_ridge=1e-5, inverse_root_damping=c["damping"],
                cost_definition="core_wall_time=wall_time-diagnostic_seconds; includes evaluation and audit; core CUDA peak is allocated memory outside diagnostic segments")
    meta.update({k:v for k,v in source_provenance.items() if k.startswith("upstream_")})
    save_json(root / "metadata.json", meta)
    rows, oracle_rows, refresh_rows, audits, syn_audits = [], [], [], [], []
    stale_audits = []
    active = None
    step = 0
    refresh_times = []
    tracker = CostTracker(dev)
    sync = tracker.sync
    try:
        for epoch in range(1, c["epochs"]+1):
            iterator = iter(loader)
            for _ in range(len(loader)):
                if step % c["K"] == 0:
                    old = active
                    if method != "dp_sgd":
                        sync()
                        t = time.perf_counter()
                        samples, syn_audit = synthetic_samples(c, dev, syn_rng)
                        candidate = refresh(model._module.state_dict(), samples, c, dev, method)
                        sync()
                        elapsed = time.perf_counter()-t
                        refresh_times.append(elapsed)
                        syn_audits.append(dict(step=step, **syn_audit))
                        active = candidate
                        del samples, candidate
                        with tracker.diagnostics():
                            probes, probe_audit = synthetic_samples(c, dev, stale_rng, budget=c["M_stale"])
                            stale_audits.append(dict(step=step, **probe_audit))
                            transforms = {"new": active}
                            if old is not None:
                                transforms["old"] = old
                            geom = diagnose(model._module.state_dict(), probes, transforms, c, dev, root)
                            for layer in LAYERS:
                                refresh_rows.append(dict(step=step, layer=layer, M_syn=c["M_syn"], M_stale=c["M_stale"], refresh_seconds=elapsed,
                                                         preconditioner_state_bytes=state_bytes(active),
                                                         **stale(geom.get("old", {}).get(layer), geom["new"][layer])))
                            del probes, transforms, geom
                    if c["oracle_enabled"]:
                        with tracker.diagnostics():
                            values = [data[int(i)] for i in ids]
                            samples = torch.stack([s[0] for s in values]), torch.tensor([s[1] for s in values])
                            transforms = {"raw": None} if active is None else {"raw": None, "new": active}
                            if old is not None:
                                transforms["old"] = old
                            geom = diagnose(model._module.state_dict(), samples, transforms, c, dev, root)
                            for layer in LAYERS:
                                # Identity old is N/A at step zero, identity thereafter.
                                previous = geom["raw"][layer] if method == "dp_sgd" and step > 0 else geom.get("old", {}).get(layer)
                                oracle_rows.append(dict(step=step, layer=layer, **oracle_compare(geom["raw"][layer], geom.get("new", geom["raw"])[layer], previous)))
                            del samples, values, transforms, geom, previous
                    del old
                    print(f"seed={seed} {method} diagnostic/refresh step={step}", flush=True)
                x, y, indices = next(iterator)
                model.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x.to(dev)), y.to(dev), reduction="sum")
                loss.backward()
                row, audit = private_update(model, active, optimizer, noise_rng, sigma, c, len(x))
                accountant.step(noise_multiplier=sigma, sample_rate=q)
                step += 1
                row.update(step=step, epoch=epoch, train_loss=float(loss.detach())/len(x), test_loss=None, test_accuracy=None)
                if step % c["eval_interval"] == 0 or step == total:
                    row.update(evaluate(model, test_loader, dev))
                rows.append(row)
                audits.append(dict(step=step, batch_indices=indices.tolist(), model_hash=digest(model.parameters()), **audit))
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("Nonfinite model parameters")
        sync()
        cost = dict(**tracker.finish(), total_refresh_time=sum(refresh_times),
                    mean_refresh_time=sum(refresh_times)/len(refresh_times) if refresh_times else 0.,
                    number_of_refreshes=len(refresh_times),
                    preconditioner_state_bytes=state_bytes(active))
        acc = [r["test_accuracy"] for r in rows if r["test_accuracy"] is not None]
        late = [r["test_accuracy"] for r in rows if r["step"] > total/2 and r["test_accuracy"] is not None]
        summary = dict(**cost, final_accuracy=acc[-1], best_accuracy=max(acc), late_mean_accuracy=sum(late)/len(late),
                       final_test_loss=rows[-1]["test_loss"],
                       epsilon_spent=accountant.get_epsilon(c["delta"]), noise_multiplier=sigma, completed_steps=step,
                       final_model_hash=digest(model.parameters()), fingerprint=fingerprint(c), seed=seed, method=method)
        meta.update(summary, complete=True)
        save_json(root / "metadata.json", meta)
        save_json(root / "summary.json", summary)
    finally:
        model.remove_hooks()
        for name, values in (("train", rows), ("oracle", oracle_rows), ("refresh", refresh_rows)):
            write_csv(root / f"{name}_metrics.csv", [dict(r, seed=seed, method=method, config_fingerprint=fingerprint(c)) for r in values],
                      fields=None if values else ["step", "layer", "seed", "method", "config_fingerprint"])
        save_json(root / "pairing.json", dict(private=audits, synthetic=syn_audits, stale=stale_audits, oracle_indices=ids.tolist()))
    print(f"Completed seed={seed} {method}: {step} steps, accuracy={acc[-1]:.4f}", flush=True)
    return meta


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/full.json"))
    parser.add_argument("--output", default=str(ROOT / "runs"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method", choices=[*METHODS, "dp_kfc_pink_matched", "all"], default="all")
    a = parser.parse_args()
    c = read_config(a.config)
    for seed in c["seeds"] if a.seed is None else [a.seed]:
        for method in METHODS if a.method == "all" else [a.method]:
            train(c, seed, method, a.output)


if __name__ == "__main__":
    main()
