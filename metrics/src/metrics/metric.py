import pywellen
from dataclasses import dataclass
from typing import Callable, Literal
from .config import Core, Benchmark, CORES, BENCHMARKS
from .simulate import sim_path

NumberType = Literal["int", "float", "percentage"]


class SignalWrapper:
    """Adapts pywellen Var/Signal objects to a standardized interface (.all_changes and .value_at_time)."""

    def __init__(self, var):
        self.var = var
        self._sig = None

    def all_changes(self):
        # 'tv' stands for Time-Value transition pairs in pywellen
        return self.var.tv() if callable(self.var.tv) else self.var.tv

    def value_at_time(self, time: int):
        if self._sig is None:
            self._sig = self.var.signal() if callable(self.var.signal) else self.var.signal

        if hasattr(self._sig, "value_at"):
            return self._sig.value_at(time)
        elif hasattr(self._sig, "value_at_time"):
            return self._sig.value_at_time(time)
        else:
            # Fallback: find the value from the transition history
            val = 0
            for t, v in self.all_changes():
                if t <= time:
                    val = v
                else:
                    break
            return val


class WaveformReader:
    def __init__(self, core: Core, benchmark: Benchmark):
        self.core = core
        self.waveform = pywellen.Waveform(str(sim_path(core, benchmark)))

        # 1. Build a signal lookup map once from all_vars for fast O(1) access
        self._signal_map = {}
        raw_vars = self.waveform.all_vars() if callable(self.waveform.all_vars) else self.waveform.all_vars

        if isinstance(raw_vars, dict):
            self._signal_map.update(raw_vars)
            var_list = raw_vars.values()
        else:
            var_list = raw_vars

        for var in var_list:
            if hasattr(var, "full_name"):
                self._signal_map[var.full_name] = var
            if hasattr(var, "name"):
                self._signal_map[var.name] = var

        # 2. Retrieve the clock signal using _signal() (which wraps it in SignalWrapper)
        clock_signal = self._signal(core.clk_signal)

        # 3. Automatic fallback: If exact match fails, search common clock names
        if clock_signal is None:
            common_clocks = ["clk", "clock", "TOP.clk", "TOP.clock", "io_clk", "TOP.io_clk"]
            for candidate in common_clocks:
                clock_signal = self._signal(candidate)
                if clock_signal is not None:
                    break

        # 4. Deep fallback: Search for ANY signal ending in clk or clock
        if clock_signal is None:
            for name in self._signal_map.keys():
                if name.lower().endswith("clk") or name.lower().endswith("clock"):
                    clock_signal = self._signal(name)
                    if clock_signal is not None:
                        break

        # 5. If still not found, raise a helpful error showing only clock-like candidates
        if clock_signal is None:
            clk_candidates = [k for k in self._signal_map.keys() if "clk" in k.lower() or "clock" in k.lower()]
            raise RuntimeError(
                f"Could not find clock signal '{core.clk_signal}' in {core.name}. "
                f"Clock candidates found in waveform: {clk_candidates if clk_candidates else 'None found!'}"
            )

        clock_changes = [t for (t, v) in clock_signal.all_changes() if v == 1]

        self.clock_period = clock_changes[1] - clock_changes[0]
        self.clock_offset = clock_changes[0] % self.clock_period  # e.g. 5
        self.max_clock = clock_changes[-1]
        self.clock_count = len(clock_changes)

    def _signal(self, name: str):
        path = self.core.signals.get(name, name)
        if isinstance(path, int):
            return path

        var = self._signal_map.get(path)
        if var is None:
            return None

        # Return our wrapper so .all_changes() and .value_at_time() work seamlessly
        return SignalWrapper(var)

    def has_signal(self, name: str) -> bool:
        """Returns True if the signal exists in the FST or config and is not disabled (-1)."""
        s = self._signal(name)
        return s is not None and s != -1

    def final_value(self, signal: str) -> int:
        try:
            s = self._signal(signal)
            if s is None or s == -1:
                return 0
            if isinstance(s, int):
                return s
            return int(s.value_at_time(self.max_clock))
        except Exception as e:
            print(f"Warning: final_value({signal}) on {self.core.name}: {e}")
            return 0

    def count_high(self, signal: str) -> int:
        try:
            s = self._signal(signal)
            if s is None or s == -1:
                return 0
            if isinstance(s, int):
                return s * self.clock_count

            count = 0
            val = 0
            prev_t = None

            for t, v in s.all_changes():
                snapped = ((t - self.clock_offset) // self.clock_period) * self.clock_period + self.clock_offset
                if val == 1 and prev_t is not None:
                    count += (snapped - prev_t) // self.clock_period
                val = v
                prev_t = snapped

            if val == 1 and prev_t is not None:
                count += (self.max_clock - prev_t) // self.clock_period

            return count

        except Exception as e:
            raise RuntimeError(f"count_high({signal}) on {self.core.name}: {e}") from e

    def count_high_and(self, signal_a: str, signal_b: str) -> int:
        try:
            s_a = self._signal(signal_a)
            s_b = self._signal(signal_b)

            if s_a is None or s_a == -1 or s_b is None or s_b == -1:
                return 0

            if isinstance(s_a, int) and isinstance(s_b, int):
                return int(s_a and s_b) * self.clock_count
            if isinstance(s_a, int):
                return s_a * self.count_high(signal_b)
            if isinstance(s_b, int):
                return s_b * self.count_high(signal_a)

            changes_a = list(s_a.all_changes())
            changes_b = list(s_b.all_changes())

            def snap(t):
                return ((t - self.clock_offset) // self.clock_period) * self.clock_period + self.clock_offset

            events = sorted([(snap(t), v, 0) for t, v in changes_a] +
                            [(snap(t), v, 1) for t, v in changes_b])

            count = 0
            val_a, val_b = 0, 0
            prev_t = None

            for t, v, src in events:
                if t != prev_t:
                    if val_a == 1 and val_b == 1 and prev_t is not None:
                        count += (t - prev_t) // self.clock_period
                    prev_t = t
                if src == 0:
                    val_a = v
                else:
                    val_b = v

            if val_a == 1 and val_b == 1 and prev_t is not None:
                count += (self.max_clock - prev_t) // self.clock_period

            return count

        except Exception as e:
            raise RuntimeError(f"count_high_and({signal_a}, {signal_b}) on {self.core.name}: {e}") from e

    def high_rate(self, signal: str) -> float:
        return self.count_high(signal) / self.clock_count

    def high_rate_and(self, signal_a: str, signal_b: str) -> float:
        return self.count_high_and(signal_a, signal_b) / self.clock_count

    def debug_signal(self, signal: str, cycles: int = 20) -> None:
        s = self._signal(signal)
        if s is None or isinstance(s, int):
            print(f"\n=== {signal}: Not a valid waveform signal (value={s}) ===")
            return
        print(f"\n=== {signal} ===")
        print(f"clock_period: {self.clock_period}, max_clock: {self.max_clock}, clock_count: {self.clock_count}")
        print(f"First {cycles} changes:")
        for t, v in list(s.all_changes())[:cycles]:
            print(f"  t={t:8d}  cycle={t // self.clock_period:6d}  v={v}")


@dataclass
class Metric:
    id: str
    name: str
    fn: Callable
    number_type: NumberType = "float"


def ipc(r: WaveformReader):
    return r.final_value("instret") / r.clock_count


def count(r: WaveformReader):
    return r.final_value("instret")


def bpr(r: WaveformReader):
    total = r.final_value("branch_mispredictions") + r.final_value("branch_predictions")
    return r.final_value("branch_mispredictions") / total if total > 0 else 0.0


def ipc_fetch_0(r: WaveformReader): return r.high_rate("fetch_0_isDone")


def ipc_fetch_1(r: WaveformReader): return r.high_rate("fetch_1_isDone")


def ipc_fetch_2(r: WaveformReader): return r.high_rate("fetch_2_isDone")


def ipc_fetch_3(r: WaveformReader): return r.high_rate("fetch_3_isDone")


def ipc_issue_0(r: WaveformReader): return r.high_rate("issue_0_isDone")


def ipc_issue_1(r: WaveformReader): return r.high_rate("issue_1_isDone")


def ipc_ret_0(r: WaveformReader): return r.high_rate_and("ret_0_isDone", "ret_0_canCommit")


def ipc_ret_1(r: WaveformReader): return r.high_rate_and("ret_1_isDone", "ret_1_canCommit")


def d_cache_misses(r: WaveformReader):
    return r.final_value("d_cache_misses")


def d_cache_miss_rate(r: WaveformReader):
    total = r.final_value("d_cache_misses") + r.final_value("d_cache_hits")
    return r.final_value("d_cache_misses") / total if total > 0 else 0.0


def i_cache_misses(r: WaveformReader):
    return r.final_value("i_cache_misses")


def i_cache_miss_rate(r: WaveformReader):
    total = r.final_value("i_cache_misses") + r.final_value("i_cache_hits")
    return r.final_value("i_cache_misses") / total if total > 0 else 0.0


def psf_mispredictions(r: WaveformReader):
    return r.final_value("psf_mispredictions")


def psf_misprediction_rate(r: WaveformReader):
    if not r.has_signal("psf_attempts"):
        attempts = r.final_value("psf_mispredictions") + r.final_value("psf_predictions")
    else:
        attempts = r.final_value("psf_attempts")
    return (r.final_value("psf_mispredictions") / attempts) if attempts > 0 else 0.0


def ssb_mispredictions(r: WaveformReader):
    return r.final_value("ssb_mispredictions")


def ssb_misprediction_rate(r: WaveformReader):
    total = r.final_value("ssb_mispredictions") + r.final_value("ssb_predictions")
    return r.final_value("ssb_mispredictions") / total if total > 0 else 0.0


def flushes(r: WaveformReader): return r.final_value("flushes")


def soft_flushes(r: WaveformReader): return r.final_value("soft_flushes")


def total_flushes(r: WaveformReader): return r.final_value("flushes") + r.final_value("soft_flushes")


def flush_rate(r: WaveformReader): return r.high_rate("flush")


def soft_flush_rate(r: WaveformReader): return r.high_rate("soft_flush")


def total_flush_rate(r: WaveformReader): return r.high_rate("flush") + r.high_rate("soft_flush")


IPC = Metric("ipc", "IPC", ipc, number_type="float")
INST = Metric("instructions", "# Instructions", count, number_type="int")
BPR = Metric("bpr", "Branch Prediction Rate", bpr, number_type="percentage")

IPC_FETCH = [Metric("ipc_fetch_0", "IPC Fetch Stage 1", ipc_fetch_0, number_type="float"),
             Metric("ipc_fetch_1", "IPC Fetch Stage 2", ipc_fetch_1, number_type="float"),
             Metric("ipc_fetch_2", "IPC Fetch Stage 3", ipc_fetch_2, number_type="float"),
             Metric("ipc_fetch_3", "IPC Fetch Stage 4", ipc_fetch_3, number_type="float")]

IPC_ISSUE = [Metric("ipc_issue_0", "IPC Decode Stage 1", ipc_issue_0, number_type="float"),
             Metric("ipc_issue_1", "IPC Decode Stage 2", ipc_issue_1, number_type="float")]

IPC_RETIREMENT = [Metric("ipc_ret_0", "IPC Retirement Stage 1", ipc_ret_0, number_type="float"),
                  Metric("ipc_ret_1", "IPC Retirement Stage 2", ipc_ret_1, number_type="float")]

D_CACHE_MISSES = Metric("d_cache_misses", "# Data Cache Misses", d_cache_misses, number_type="int")
D_CACHE_MISS_RATE = Metric("d_cache_miss_rate", "Data Cache Miss Rate", d_cache_miss_rate, number_type="percentage")

I_CACHE_MISSES = Metric("i_cache_misses", "# Instruction Cache Misses", i_cache_misses, number_type="int")
I_CACHE_MISS_RATE = Metric("i_cache_miss_rate", "Instruction Cache Miss Rate", i_cache_miss_rate,
                           number_type="percentage")

PSF_MISPREDICTIONS = Metric("psf_mispredictions", "PSF Mispredictions", psf_mispredictions, number_type="int")
PSF_MISPREDICTION_RATE = Metric("psf_misprediction_rate", "PSF Misprediction Rate", psf_misprediction_rate,
                                number_type="percentage")

SSB_MISPREDICTIONS = Metric("ssb_mispredictions", "SSB Mispredictions", ssb_mispredictions, number_type="int")
SSB_MISPREDICTION_RATE = Metric("ssb_misprediction_rate", "SSB Misprediction Rate", ssb_misprediction_rate,
                                number_type="percentage")

FLUSHES = Metric("flushes", "# ROB Flushes", flushes, number_type="int")
SOFT_FLUSHES = Metric("soft_flushes", "# ROB Soft Flushes", soft_flushes, number_type="int")
TOTAL_FLUSHES = Metric("total_flushes", "# ROB Flushes", total_flushes, number_type="int")

FLUSH_RATE = Metric("flush_rate", "ROB Flush Rate", flush_rate, number_type="percentage")
SOFT_FLUSH_RATE = Metric("soft_flush_rate", "ROB Soft Flush Rate", soft_flush_rate, number_type="percentage")
TOTAL_FLUSH_RATE = Metric("total_flush_rate", "ROB Flush Rate", total_flush_rate, number_type="percentage")

SIMPLE_METRICS = [
    IPC, INST, BPR,
    D_CACHE_MISSES, D_CACHE_MISS_RATE,
    I_CACHE_MISSES, I_CACHE_MISS_RATE,
    PSF_MISPREDICTIONS, PSF_MISPREDICTION_RATE,
    SSB_MISPREDICTIONS, SSB_MISPREDICTION_RATE,
    FLUSHES, SOFT_FLUSHES, FLUSH_RATE, SOFT_FLUSH_RATE, TOTAL_FLUSHES, TOTAL_FLUSH_RATE
]

STACKED_METRICS = []
STACKED_METRICS.extend(IPC_FETCH)
STACKED_METRICS.extend(IPC_ISSUE)
STACKED_METRICS.extend(IPC_RETIREMENT)

METRICS = []
METRICS.extend(SIMPLE_METRICS)
METRICS.extend(STACKED_METRICS)