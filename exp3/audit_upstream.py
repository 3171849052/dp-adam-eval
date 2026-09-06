"""Read-only checkout/public audit. Never fetch into or modify upstream."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[2] / "DP-KFC"
EXPECTED_UPSTREAM_REPO = "https://github.com/molinamarcvdb/DP-KFC.git"
PUBLIC = EXPECTED_UPSTREAM_REPO
PINNED_COMMIT = "eb31b9aeb2280642684f4cedfa65cc02b76c76cd"


def git(*args):
    return subprocess.check_output(["git", "--no-optional-locks", "-C", str(REPO), *args], text=True).strip()


def canonicalize_repo_url(url):
    """Normalize the supported HTTPS repo spellings for provenance checks."""
    normalized = str(url).rstrip("/")
    return normalized[:-4] if normalized.endswith(".git") else normalized


def checkout_info():
    status = git("status", "--porcelain", "--untracked-files=all")
    origin_url = git("remote", "get-url", "origin")
    return dict(upstream_repo=EXPECTED_UPSTREAM_REPO, upstream_origin=origin_url,
                upstream_git_commit=git("rev-parse", "HEAD"),
                upstream_git_dirty=bool(status), upstream_git_remote=git("remote", "-v"),
                upstream_git_status=status.splitlines())


def require_clean(info, smoke):
    if not smoke and info["upstream_git_dirty"]:
        raise ValueError("Formal Exp3 requires a clean upstream checkout; dirty upstream is allowed only for smoke")


def require_pinned(info, smoke):
    """Require the public checkout pin whenever Exp3 is in formal mode."""
    if smoke:
        return
    require_clean(info, smoke)
    if info.get("upstream_git_commit") != PINNED_COMMIT:
        raise ValueError(
            f"Formal Exp3 requires upstream commit {PINNED_COMMIT} (exact pin); "
            f"found {info.get('upstream_git_commit')}"
        )
    if canonicalize_repo_url(info.get("upstream_origin")) != canonicalize_repo_url(EXPECTED_UPSTREAM_REPO):
        raise ValueError(
            f"Formal Exp3 requires origin {EXPECTED_UPSTREAM_REPO}; "
            f"found {info.get('upstream_origin')}"
        )


def audit():
    info = checkout_info()
    remote = subprocess.check_output(["git", "ls-remote", PUBLIC, "HEAD"], text=True).split()[0]
    info["expected_upstream_repo"] = EXPECTED_UPSTREAM_REPO
    info["origin_matches_expected"] = (
        canonicalize_repo_url(info["upstream_origin"]) == canonicalize_repo_url(EXPECTED_UPSTREAM_REPO)
    )
    info["public_head"] = remote
    info["pinned_commit"] = PINNED_COMMIT
    info["local_head_matches_pinned"] = info["upstream_git_commit"] == PINNED_COMMIT
    info["public_head_matches_pinned"] = remote == PINNED_COMMIT
    info["local_head_matches_public"] = remote == info["upstream_git_commit"]
    info["files"] = {}
    for p in sorted((REPO / "src/dp_kfac").glob("*.py")):
        rel = p.relative_to(REPO).as_posix()
        base = subprocess.check_output(["git", "-C", str(REPO), "show", f"HEAD:{rel}"])
        local = p.read_bytes()
        info["files"][rel] = dict(checkout_sha256=hashlib.sha256(local).hexdigest(),
                                head_sha256=hashlib.sha256(base).hexdigest(), differs_from_head=local != base)
    selections = {"optimizer.py": ["generate_pink_noise", "DPKFACOptimizer.compute_preconditioner_from_noise"],
                  "covariance.py": ["compute_linear_covariances", "compute_conv2d_covariances", "compute_covariances", "accumulate_covariances", "compute_inverse_sqrt"],
                  "precondition.py": ["precondition_per_sample_gradients"], "recorder.py": ["KFACRecorder"],
                  "trainer.py": ["set_seed"],
                  "privacy.py": ["_compute_per_sample_norms_squared", "_compute_clip_factors", "clip_and_noise_gradients"]}
    def node(source, name):
        body = ast.parse(source).body
        for part in name.split("."):
            result = next(n for n in body if getattr(n, "name", None) == part)
            body = result.body
        return ast.dump(result, include_attributes=False)
    info["callable_ast_matches_head"] = {}
    for filename, names in selections.items():
        local = (REPO / "src/dp_kfac" / filename).read_text()
        base = git("show", f"HEAD:src/dp_kfac/{filename}")
        for name in names:
            info["callable_ast_matches_head"][f"{filename}:{name}"] = node(local, name) == node(base, name)
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", default="exp3/upstream_audit.json")
    args = parser.parse_args()
    result = audit()
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "files"}, indent=2))
