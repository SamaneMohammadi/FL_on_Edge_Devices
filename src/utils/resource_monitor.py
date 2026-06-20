"""
Per-client resource monitoring - the numbers behind Table II (CPU time, RAM %)
and the training-time / exchange-time figures.

This is intentionally simple: we sample psutil around the local training call so
each device records its own user/system CPU time and memory footprint. On the
real testbed these differ a lot between HW T1 (Raspberry Pi 3) and HW T5
(Raspberry Pi 4 8GB), which is the whole point.
"""

import time
import psutil


class ResourceMonitor:
    def __init__(self):
        self.proc = psutil.Process()

    def snapshot_start(self):
        self._cpu_start = self.proc.cpu_times()
        self._wall_start = time.time()

    def snapshot_end(self):
        cpu_end = self.proc.cpu_times()
        wall = time.time() - self._wall_start
        mem = self.proc.memory_info()
        return {
            "wall_time_s": wall,
            "cpu_user_s": cpu_end.user - self._cpu_start.user,
            "cpu_system_s": cpu_end.system - self._cpu_start.system,
            "ram_rss_mb": mem.rss / (1024 * 1024),
            "ram_percent": psutil.virtual_memory().percent,
        }
