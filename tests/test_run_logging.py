from datetime import datetime
from pathlib import Path

from dp_wiener_mnist.config import Config
from dp_wiener_mnist.run_logging import (
    METRICS_FIELDS,
    MetricsCSVWriter,
    create_run_directory,
    format_run_name,
    format_tmux_session_name,
    run_paths_from_directory,
)


def test_run_name_format_and_collision(tmp_path):
    c = Config()
    now = datetime(2026, 9, 9, 12, 34, 56)
    first_name = format_run_name(c, now)
    assert first_name == "0909-123456_simple_cnn_dp_sgd_s42_ep5_lr0.1_eps1_d1e-5_beta1_K50_M2560"
    assert "/" not in first_name and " " not in first_name

    first = create_run_directory(c, root=tmp_path, now=now)
    second = create_run_directory(c, root=tmp_path, now=now)
    assert first.directory.name == first_name
    assert second.directory.name.startswith("0909-123457_")
    assert format_tmux_session_name(first.directory).startswith("dp_wiener_mnist_")
    assert all(ch.isalnum() or ch in "_-" for ch in format_tmux_session_name(first.directory))
    first.config.write_text("{}\n")
    assert run_paths_from_directory(first.directory) == first


def test_selected_scalar_changes_name():
    c = Config()
    fields = [
        ("algorithm", "dp_fisher_wiener"),
        ("seed", 7),
        ("model.name", "other_model"),
        ("training.epochs", 2),
        ("training.learning_rate", 0.2),
        ("privacy.epsilon", 2.0),
        ("privacy.delta", 2e-5),
        ("wiener.beta", 2),
        ("wiener.refresh_interval", 25),
        ("wiener.synthetic_samples", 1280),
    ]
    baseline = format_run_name(c, datetime(2026, 1, 1))
    for path, value in fields:
        c = Config()
        target = c
        pieces = path.split(".")
        for piece in pieces[:-1]:
            target = getattr(target, piece)
        setattr(target, pieces[-1], value)
        assert format_run_name(c, datetime(2026, 1, 1)) != baseline, path


def test_omitted_settings_do_not_change_name():
    baseline = format_run_name(Config(), datetime(2026, 1, 1))
    fields = [
        ("data.dataset", "other_dataset"),
        ("data.batch_size", 128),
        ("data.eval_batch_size", 128),
        ("training.optimizer", "other_optimizer"),
        ("training.momentum", 0.1),
        ("training.weight_decay", 0.1),
        ("privacy.max_grad_norm", 2.0),
        ("privacy.accountant", "other_accountant"),
        ("privacy.sampling", "other_sampling"),
        ("privacy.accounting_convention", "other_convention"),
        ("privacy.poisson_sampling", False),
        ("wiener.covariance_ridge", 2e-5),
        ("wiener.synthetic_distribution", "other_distribution"),
        ("runtime.device", "cpu"),
        ("runtime.gpu", 1),
        ("runtime.threads", 2),
        ("runtime.deterministic", False),
    ]
    for path, value in fields:
        c = Config()
        target = c
        pieces = path.split(".")
        for piece in pieces[:-1]:
            target = getattr(target, piece)
        setattr(target, pieces[-1], value)
        assert format_run_name(c, datetime(2026, 1, 1)) == baseline, path


def test_metrics_writer_header_and_row(tmp_path):
    path = Path(tmp_path) / "metrics.csv"
    writer = MetricsCSVWriter(path)
    writer.append({field: None for field in METRICS_FIELDS})
    assert len(path.read_text().splitlines()) == 2
