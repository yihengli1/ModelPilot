import os
import psutil

# MEMORY DEBUGGING
_proc = psutil.Process(os.getpid())


def mem_mb():
    return _proc.memory_info().rss / (1024 * 1024)


def log_mem(msg):
    print(f"[MEM] {msg}: {mem_mb():.1f} MB")
