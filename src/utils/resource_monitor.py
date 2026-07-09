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
