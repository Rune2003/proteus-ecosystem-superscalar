import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

ROOT = Path(__file__).parent.parent.parent

BUILD_DIR = ROOT / "builds"
BENCHMARK_DIR = ROOT / "benchmarks"
RESULTS_DIR = ROOT / "results"
INTERFACES_DIR = ROOT / "cpu-interfaces"


@dataclass
class Benchmark:
    name: str
    path: Path


BENCHMARKS = [
    Benchmark(name=filename.stem, path=BENCHMARK_DIR / filename)
    for filename in BENCHMARK_DIR.glob("*.bin")
]


@dataclass
class Core:
    id: str
    name: str
    core: str
    hw_def_path: Path
    sim_path: Path
    clk_signal: str
    signals: dict[str, str | int] = field(default_factory=dict)

    @staticmethod
    def _flatten_signals(raw_signals: dict, prefix: str = "") -> dict[str, str | int]:
        """Recursively flattens categorized dictionaries and applies string prefixes."""
        flat = {}
        for key, value in raw_signals.items():
            if isinstance(value, dict):
                flat.update(Core._flatten_signals(value, prefix))
            elif isinstance(value, str):
                # Automatically join with a dot if a prefix exists and it's not an absolute path
                if prefix and not value.startswith("TOP"):
                    flat[key] = f"{prefix}.{value}"
                else:
                    flat[key] = value
            else:
                # Keep integer flags (0, -1) as-is
                flat[key] = value
        return flat

    @classmethod
    def _load_and_resolve(cls, config_path: Path) -> tuple[dict, dict[str, str | int]]:
        """Recursively loads JSON configs, inheriting top-level properties and merging signals."""
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        base_signals = {}
        if "base_config" in data:
            # Resolve base_config path relative to the current configuration file
            base_path = config_path.parent / data["base_config"]
            base_data, base_signals = cls._load_and_resolve(base_path)

            # Inherit top-level properties if not explicitly overridden by the child
            for key in ("clk_signal", "signal_prefix"):
                if key not in data and key in base_data:
                    data[key] = base_data[key]

        prefix = data.get("signal_prefix", "")
        raw_signals = data.get("signals", {})
        child_signals = cls._flatten_signals(raw_signals, prefix)

        # Python dictionary union (|): child_signals overwrite base_signals on key conflicts
        merged_signals = base_signals | child_signals
        return data, merged_signals

    @classmethod
    def from_json(cls, config_path: Path) -> Self:
        data, resolved_signals = cls._load_and_resolve(config_path)

        core_id = data["id"]
        return cls(
            id=core_id,
            name=data["name"],
            core=data["core"],
            hw_def_path=BUILD_DIR / core_id / "Core.v",
            sim_path=BUILD_DIR / core_id / "sim",
            clk_signal=data["clk_signal"],
            signals=resolved_signals,
        )


# Ensure the configurations directory exists
INTERFACES_DIR.mkdir(parents=True, exist_ok=True)

# Dynamically load all .json configurations
CORES = [
    Core.from_json(config_file)
    for config_file in sorted(INTERFACES_DIR.glob("*.json"))
]
