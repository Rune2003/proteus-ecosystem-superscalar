import argparse
import concurrent.futures
import subprocess
from pathlib import Path
from tqdm import tqdm

from .config import Benchmark, Core, CORES, BENCHMARKS, RESULTS_DIR


def results_path(core: Core, benchmark: Benchmark) -> Path:
    return RESULTS_DIR / core.id / benchmark.name


def sim_path(core: Core, benchmark: Benchmark) -> Path:
    return results_path(core, benchmark) / (benchmark.name + ".fst")


def has_valid_run(core: Core, benchmark: Benchmark) -> bool:
    result_path = results_path(core, benchmark)

    if not result_path.exists() or not result_path.is_dir():
        return False

    if not (sim_path(core, benchmark)).exists():
        return False

    if (result_path / "log.txt").exists():
        with open(result_path / "log.txt", "r") as file:
            if "RET=0" in file.read():
                return True
            else:
                return False

    return False


def run_simulation(core: Core, benchmark: Benchmark) -> str:
    """Runs a single simulation using the subprocess module and returns the result message."""
    result_path = results_path(core, benchmark)
    result_path.mkdir(parents=True, exist_ok=True)

    command = [
        str(core.sim_path),
        "--dump-fst",
        str(result_path / (benchmark.name + ".fst")),
        str(benchmark.path)
    ]

    try:
        # Executes the simulation command (renamed 'log' to 'process' to avoid shadowing logging.log)
        process = subprocess.run(command, check=True, text=True, capture_output=True)

        with open(result_path / "log.txt", "w") as file:
            file.write(process.stdout)

        if "RET=0" not in process.stdout:
            return f"Error: running {benchmark.name} on {core.name} failed. (RET=1)\nDetails: {process.stdout}"

        return f"Success: {core.name} completed {benchmark.name}"

    except subprocess.CalledProcessError as error:
        # Handles any errors that occur during execution
        return f"Error: running {benchmark.name} on {core.name} failed.\nDetails: {error.stderr}"


def simulate(cores: list[Core], benchmarks: list[Benchmark], max_workers: int | None = None) -> None:
    """Runs all combinations of cores and benchmarks in parallel."""

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        # 1. Queue up all the tasks
        for core in cores:
            for benchmark in benchmarks:
                if has_valid_run(core, benchmark):
                    # We can use standard print here because the progress bar hasn't started yet
                    print(f"Skipped: {core.name} already finished {benchmark.name} before")
                else:
                    futures.append(executor.submit(run_simulation, core, benchmark))

        # 2. Process tasks as they complete and update the progress bar
        # If there are no futures (e.g., all were skipped), tqdm will just instantly complete
        if futures:
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Simulations"):
                result_message = future.result()
                # Use tqdm.write instead of print to avoid breaking the progress bar visually
                if result_message:
                    tqdm.write(result_message)


def main():
    parser = argparse.ArgumentParser(description="Run Proteus core benchmark simulations.")
    parser.add_argument(
        "-j", "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes to use (defaults to available CPU cores)."
    )
    args = parser.parse_args()

    # simulate(list(filter(lambda c: c.name == "Scalar", CORES)), list(filter(lambda b: b.name == "sglib-combined", BENCHMARKS)), max_workers=args.workers)
    simulate(CORES, BENCHMARKS, max_workers=args.workers)


if __name__ == "__main__":
    main()