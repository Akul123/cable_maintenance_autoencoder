from dataclasses import dataclass
from collections import deque
import time

@dataclass
class MemoryConfig:
    ewma_alpha: float = 0.05
    warn_rate_1h: float = 0.05
    crit_rate_1h: float = 0.15
    warn_streak: int = 3
    crit_streak: int = 6

class LongTermMemory:
    def __init__(self, cfg: MemoryConfig):
        self.cfg = cfg
        self.ewma_error = None
        self.streak = 0
        self.last_anom_ts = None
        self.win_10m = deque()  # (ts, is_anom)
        self.win_1h = deque()   # (ts, is_anom)

    def _push(self, win: deque, ts: float, is_anom: int, horizon_sec: float):
        win.append((ts, is_anom))
        cutoff = ts - horizon_sec
        while win and win[0][0] < cutoff:
            win.popleft()

    @staticmethod
    def _rate(win: deque) -> float:
        if not win:
            return 0.0
        return sum(v for _, v in win) / float(len(win))

    def update(self, err: float, is_anom: bool, ts: float | None = None):
        ts = time.time() if ts is None else ts
        is_anom_i = 1 if is_anom else 0

        if self.ewma_error is None:
            self.ewma_error = err
        else:
            a = self.cfg.ewma_alpha
            self.ewma_error = a * err + (1.0 - a) * self.ewma_error

        self.streak = self.streak + 1 if is_anom else 0
        if is_anom:
            self.last_anom_ts = ts

        self._push(self.win_10m, ts, is_anom_i, 600.0)
        self._push(self.win_1h, ts, is_anom_i, 3600.0)

        rate_10m = self._rate(self.win_10m)
        rate_1h = self._rate(self.win_1h)
        time_since_last = None if self.last_anom_ts is None else (ts - self.last_anom_ts)

        level = "normal"
        if rate_1h >= self.cfg.crit_rate_1h or self.streak >= self.cfg.crit_streak:
            level = "critical"
        elif rate_1h >= self.cfg.warn_rate_1h or self.streak >= self.cfg.warn_streak:
            level = "warning"

        return {
            "ewma_error": float(self.ewma_error),
            "anom_rate_10m": float(rate_10m),
            "anom_rate_1h": float(rate_1h),
            "streak_len": int(self.streak),
            "time_since_last_anom_s": None if time_since_last is None else float(time_since_last),
            "long_term_level": level,
        }
