import time
from contextlib import contextmanager
from typing import Generator, Dict, Any

class ExecutionTimer:
    def __init__(self):
        self.timings: Dict[str, float] = {}

    @contextmanager
    def measure(self, label: str) -> Generator[None, None, None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timings[label] = round(elapsed, 4)

    def get_time(self, label: str) -> float:
        return self.timings.get(label, 0.0)

    def summary(self) -> Dict[str, float]:
        return dict(self.timings)
