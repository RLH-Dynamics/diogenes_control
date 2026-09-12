"""Process tuning and loop pacing for a soft-real-time control loop.

Two things live here:

  * `configure_realtime()` -- the cheap process-level settings that reduce
    timing jitter. Call once at startup.
  * `LoopPacer` -- absolute-deadline pacing. The previous loop slept for
    `DT - elapsed`, which accumulates its own overhead on every cycle; measured
    logs showed a systematic ~57 us overshoot (197.7 Hz instead of 200 Hz).
    Pacing against an absolute schedule removes that bias entirely.
"""

import gc
import os
import sys
import time


def configure_realtime(fifo_priority: int = 80, switch_interval: float = 0.0005,
                       freeze_gc: bool = True, verbose: bool = True):
    """Apply process-level jitter reductions. Safe to call without privileges."""
    # CPython's default thread switch interval is 5 ms -- a quarter of a 50 Hz
    # cycle. With an IMU reader thread running, that is enough to delay the
    # control thread by a visible fraction of its budget.
    sys.setswitchinterval(switch_interval)

    if freeze_gc:
        # Move everything allocated during import into a permanent generation,
        # then stop cyclic collection. Reference counting still frees objects,
        # so steady-state loop allocations are reclaimed as usual; this only
        # removes the unpredictable mark-and-sweep pauses.
        gc.collect()
        gc.freeze()
        gc.disable()

    scheduled = False
    try:
        param = os.sched_param(fifo_priority)
        os.sched_setscheduler(0, os.SCHED_FIFO, param)
        scheduled = True
    except (PermissionError, OSError, AttributeError):
        pass

    if verbose:
        if scheduled:
            print(f"[INFO] Real-time scheduling active (SCHED_FIFO {fifo_priority}).")
        else:
            print("[INFO] Running at normal scheduling priority. For tighter "
                  "timing, launch under 'sudo chrt -f 80 ...'.")
    return scheduled


def restore_gc():
    """Re-enable cyclic garbage collection. Call during teardown."""
    gc.enable()


class LoopPacer:
    """Paces a loop against an absolute schedule, with an optional spin tail.

    `sleep()` gives up the CPU for the bulk of the wait and then busy-waits the
    final `spin_margin` seconds, which is where `time.sleep()`'s granularity
    would otherwise show up as jitter.
    """

    def __init__(self, period: float, spin_margin: float = 0.0003):
        self.period = period
        self.spin_margin = spin_margin
        self.next_deadline = None

        # Overrun accounting, so callers can report timing honestly.
        self.cycles = 0
        self.overruns = 0
        self.worst_overrun_s = 0.0

    def start(self):
        self.next_deadline = time.perf_counter() + self.period

    def sleep(self) -> float:
        """Wait until the next deadline. Returns the overrun in seconds (0 if none)."""
        if self.next_deadline is None:
            self.start()

        now = time.perf_counter()
        overrun = 0.0

        if now > self.next_deadline:
            overrun = now - self.next_deadline
            self.overruns += 1
            self.worst_overrun_s = max(self.worst_overrun_s, overrun)
            # Re-base rather than trying to catch up: chasing a missed deadline
            # only compresses the following cycles and makes things worse.
            self.next_deadline = now + self.period
        else:
            target = self.next_deadline
            coarse = target - self.spin_margin
            remaining = coarse - now
            if remaining > 0:
                time.sleep(remaining)
            while time.perf_counter() < target:
                pass
            self.next_deadline = target + self.period

        self.cycles += 1
        return overrun

    def summary(self) -> str:
        if self.cycles == 0:
            return "no cycles run"
        pct = 100.0 * self.overruns / self.cycles
        return (f"{self.cycles} cycles, {self.overruns} overruns ({pct:.2f}%), "
                f"worst {self.worst_overrun_s * 1000:.2f} ms")
