import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

from .config import Core, Benchmark, CORES, BENCHMARKS, RESULTS_DIR
from .metric import WaveformReader, Metric, METRICS


def _collect_one(args) -> tuple[str, str, dict[str, float]]:
    core, benchmark, metrics = args
    scores = {}
    try:
        reader = WaveformReader(core, benchmark)
        for m in metrics:
            scores[m.name] = m.fn(reader)
    except BaseException as e:
        print(f"Warning: skipping {benchmark.name} on {core.name}: {e}")
    return core.id, benchmark.name, scores


def collect(cores: list[Core], benchmarks: list[Benchmark],
            metrics: list[Metric], path: Path, max_workers: int | None = None) -> None:
    total = len(cores) * len(benchmarks)

    args = [
        (core, benchmark, metrics)
        for core, benchmark in itertools.product(cores, benchmarks)
        if not all(_already_saved(core.id, benchmark.name, m, path) for m in metrics)
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_collect_one, a) for a in args]

        with tqdm(total=total, unit="bench") as bar:
            # Account for tasks that were skipped because they already exist
            skipped_count = total - len(args)
            if skipped_count > 0:
                bar.update(skipped_count)

            for future in as_completed(futures):
                core_id, benchmark_name, scores = future.result()
                bar.set_postfix(core=core_id, bench=benchmark_name)
                for m in metrics:
                    if m.name in scores:
                        _append(scores[m.name], core_id, benchmark_name, m, path)
                bar.update()


def _append(value: float, core_id: str, benchmark_name: str,
            metric: Metric, path: Path) -> None:
    p = path / metric.id
    p.mkdir(parents=True, exist_ok=True)
    f = p / f"{core_id}.json"

    if f.exists():
        with open(f) as fp:
            data = json.load(fp)
    else:
        data = {"metric": metric.name, "core": core_id, "scores": {}}

    data["scores"][benchmark_name] = value

    with open(f, "w") as fp:
        json.dump(data, fp, indent=2)


def _already_saved(core_id: str, benchmark_name: str,
                   metric: Metric, path: Path) -> bool:
    # Fixed: Use metric.id instead of metric.name to match _append and save
    f = path / metric.id / f"{core_id}.json"
    if not f.exists():
        return False
    try:
        with open(f) as fp:
            return benchmark_name in json.load(fp).get("scores", {})
    except (json.JSONDecodeError, OSError):
        return False


def save(raw: list[tuple[str, float]], core: Core, metric: Metric, path: Path) -> None:
    p = path / metric.id
    print(metric.id)
    p.mkdir(parents=True, exist_ok=True)
    with open(p / f"{core.id}.json", "w") as f:
        json.dump({"metric": metric.name, "core": core.id, "scores": dict(raw)}, f, indent=2)


def load(cores: list[Core], metric: Metric, path: Path) -> dict[str, dict[str, float]]:
    raw = {}
    for core in cores:
        f = path / metric.id / f"{core.id}.json"
        if f.exists():
            with open(f) as fp:
                data = json.load(fp)
                # If it's a standard metric, extract the "scores" dictionary.
                # If it's a single metric, pass the whole dictionary so plot_single can access "score".
                if "scores" in data:
                    raw[core.id] = data["scores"]
                else:
                    raw[core.id] = data
        else:
            from tqdm import tqdm
            tqdm.write(f"Warning: missing {f}")
    return raw


def main():
    parser = argparse.ArgumentParser(description="Collect Proteus simulation metrics from waveform files.")
    parser.add_argument(
        "-j", "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes to use (defaults to available CPU cores)."
    )
    args = parser.parse_args()

    collect(CORES, BENCHMARKS, METRICS, RESULTS_DIR / "metrics", max_workers=args.workers)


if __name__ == "__main__":
    main()