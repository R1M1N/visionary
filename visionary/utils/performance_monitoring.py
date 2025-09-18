
"""
Performance Monitoring for Visionary

Adds FPS monitoring, memory usage tracking,
processing pipeline profiling, and real-time metrics display.
"""

import time
import psutil
import threading
import matplotlib.pyplot as plt
from collections import deque

class PerformanceMonitor:
    def __init__(self, max_samples: int = 100):
        self.max_samples = max_samples
        self.frame_times = deque(maxlen=max_samples)
        self.start_time = None
        self.end_time = None
        self.memory_usage = deque(maxlen=max_samples)
        self.process = psutil.Process()

    def start_frame(self):
        self.start_time = time.time()

    def end_frame(self):
        self.end_time = time.time()
        frame_time = self.end_time - self.start_time
        self.frame_times.append(frame_time)
        mem_usage = self.process.memory_info().rss / (1024 * 1024)  # MB
        self.memory_usage.append(mem_usage)

    def get_fps(self) -> float:
        if not self.frame_times:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    def get_memory_usage(self) -> float:
        if not self.memory_usage:
            return 0.0
        return sum(self.memory_usage) / len(self.memory_usage)

    def profile(self, func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function {func.__name__} took {(end - start):.4f} seconds")
        return result

    def display_real_time(self):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
        fps_data = deque(maxlen=self.max_samples)
        mem_data = deque(maxlen=self.max_samples)

        def update(frame):
            fps = self.get_fps()
            mem = self.get_memory_usage()

            fps_data.append(fps)
            mem_data.append(mem)

            ax1.clear()
            ax2.clear()

            ax1.plot(list(fps_data))
            ax2.plot(list(mem_data))

            ax1.set_title('FPS (Frames Per Second)')
            ax2.set_title('Memory Usage (MB)')

            ax1.set_ylim(0, max(30, max(fps_data) + 5))
            ax2.set_ylim(0, max(100, max(mem_data) + 50))

            ax1.set_xlabel('Time (frames)')
            ax2.set_xlabel('Time (frames)')

            ax1.set_ylabel('FPS')
            ax2.set_ylabel('Memory (MB)')

            plt.tight_layout()

        ani = animation.FuncAnimation(fig, update, interval=1000)
        plt.show()
