"""Strict dataclass/YAML configuration for one experiment."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import math
import yaml


@dataclass
class ModelConfig:
    name: str = "simple_cnn"


@dataclass
class DataConfig:
    dataset: str = "mnist"
    root: str = "data"
    batch_size: int = 256
    eval_batch_size: int = 256
    train_subset: int | None = None
    test_subset: int | None = None
    num_workers: int = 0


@dataclass
class TrainingConfig:
    epochs: int = 5
    learning_rate: float = 0.1
    optimizer: str = "sgd"
    momentum: float = 0.0
    weight_decay: float = 0.0


@dataclass
class PrivacyConfig:
    epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    accountant: str = "rdp"
    sampling: str = "poisson"
    accounting_convention: str = "poisson_rdp"
    poisson_sampling: bool = True


@dataclass
class WienerConfig:
    beta: int = 1
    refresh_interval: int = 50
    synthetic_samples: int = 2560
    covariance_ridge: float = 1e-5
    synthetic_distribution: str = "pink_noise"


@dataclass
class RuntimeConfig:
    device: str = "auto"
    gpu: int = 0
    threads: int = 4
    deterministic: bool = True


@dataclass
class OutputConfig:
    root: str = "outputs"


@dataclass
class Config:
    algorithm: str = "dp_sgd"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    wiener: WienerConfig = field(default_factory=WienerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        if not isinstance(raw, dict):
            raise ValueError("Configuration must be a mapping")
        default = cls()
        if set(raw) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown top-level configuration keys")
        for key, value in raw.items():
            old = getattr(default, key)
            if hasattr(old, "__dataclass_fields__"):
                if not isinstance(value, dict) or set(value) - {
                    f.name for f in fields(old)
                }:
                    raise ValueError(f"Invalid section {key}")
                value = type(old)(**value)
            setattr(default, key, value)
        default.validate()
        return default

    def validate(self) -> None:
        """Reject unsupported semantics, wrong types, and non-finite numbers."""
        baseline = Config().to_dict()

        def types(actual, expected, prefix=""):
            for key, value in actual.items():
                ref = expected[key]
                path = prefix + key
                if isinstance(ref, dict):
                    types(value, ref, path + ".")
                elif ref is None:
                    if value is not None and (type(value) is not int or value < 1):
                        raise ValueError(path)
                elif type(ref) is float:
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError(path)
                elif type(value) is not type(ref):
                    raise ValueError(path)

        types(self.to_dict(), baseline)
        fixed = [
            (self.model.name, "simple_cnn"),
            (self.data.dataset, "mnist"),
            (self.training.optimizer, "sgd"),
            (self.training.momentum, 0),
            (self.training.weight_decay, 0),
            (self.privacy.accountant, "rdp"),
            (self.privacy.sampling, "poisson"),
            (self.privacy.accounting_convention, "poisson_rdp"),
            (self.privacy.poisson_sampling, True),
            (self.wiener.beta, 1),
            (self.wiener.covariance_ridge, 1e-5),
            (self.wiener.synthetic_distribution, "pink_noise"),
            (self.runtime.deterministic, True),
        ]
        if any(a != b for a, b in fixed):
            raise ValueError("Unsupported algorithm semantics")
        if self.algorithm not in ("dp_sgd", "dp_fisher_wiener"):
            raise ValueError("Unsupported algorithm")
        positive = [
            self.training.epochs,
            self.training.learning_rate,
            self.data.batch_size,
            self.data.eval_batch_size,
            self.privacy.epsilon,
            self.privacy.max_grad_norm,
            self.wiener.refresh_interval,
            self.wiener.synthetic_samples,
            self.runtime.threads,
        ]
        if any(v <= 0 for v in positive) or not 0 < self.privacy.delta < 1:
            raise ValueError("Expected positive values and 0 < delta < 1")
        if (
            not 0 <= self.seed < 2**32 - 4
            or self.runtime.gpu < 0
            or self.data.num_workers < 0
        ):
            raise ValueError("Invalid seed, GPU or workers")
        if self.wiener.synthetic_samples % self.data.batch_size:
            raise ValueError("synthetic_samples must be divisible by batch_size")
        if (
            self.data.train_subset is not None
            and self.data.train_subset < self.data.batch_size
        ):
            raise ValueError("Training subset must be at least batch_size")
        if self.runtime.device not in ("auto", "cpu", "cuda", "cuda:0"):
            raise ValueError("Device must be auto, cpu, cuda or cuda:0")
        if not self.output.root.strip() or not self.data.root.strip():
            raise ValueError("Empty output/data path")

    def fisher_config(self) -> dict:
        return dict(
            batch_size=self.data.batch_size, M_syn=self.wiener.synthetic_samples
        )


def load_config(path: str | Path) -> Config:
    return Config.from_dict(yaml.safe_load(Path(path).read_text()))
