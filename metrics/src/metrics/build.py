import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm
from .config import Core, CORES

# --- Configuration Paths ---
SIM_DIR = Path("/ecosystem/simulation")
BUILD_SRC_DIR = SIM_DIR / "build"
# ---------------------------


def run_build(core: Core) -> str:
    """Runs the make command for a single core and copies artifacts to its build directory."""
    # Ensure target directory exists (e.g., builds/<core.id>/)
    dest_dir = core.hw_def_path.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    log_file = dest_dir / "build.log"

    command = [
        "make",
        "-C", str(SIM_DIR),
        f"CORE={core.core}",
        "DEV=0"
    ]
    print(" ".join(command))

    try:
        # Run the build command and capture stdout/stderr
        process = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True
        )

        # Write log file for debugging
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(process.stdout)
            if process.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(process.stderr)

        # Verify artifacts were generated
        src_sim = BUILD_SRC_DIR / "sim"
        src_core_v = BUILD_SRC_DIR / "Core.v"

        if not src_sim.exists() or not src_core_v.exists():
            return f"Error: Build succeeded for {core.name}, but expected artifacts were missing in {BUILD_SRC_DIR}"

        # Copy artifacts using copy2 to preserve metadata and executable file permissions
        shutil.copy2(src_sim, core.sim_path)
        shutil.copy2(src_core_v, core.hw_def_path)

        return f"Success: Built {core.name} ({core.id}) -> {dest_dir}"

    except subprocess.CalledProcessError as error:
        # Save compiler/verilog errors to the log file
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(error.stdout or "")
            if error.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(error.stderr)

        return f"Error: Building {core.name} ({core.id}) failed. Check log at: {log_file}"
    except Exception as e:
        return f"Error: Unexpected failure while building {core.name}: {e}"


def build(cores: list[Core]) -> None:
    """Builds all configured cores sequentially to avoid shared build directory collisions."""
    print(f"Starting build for {len(cores)} core configuration(s)...")

    # Sequential execution prevents multiple Make processes from clobbering BUILD_SRC_DIR simultaneously
    for core in tqdm(cores, desc="Building Cores", unit="core"):
        result_message = run_build(core)
        if result_message:
            tqdm.write(result_message)


def main():
    build(CORES)

if __name__ == "__main__":
    main()