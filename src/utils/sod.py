"""
Send-on-Delta (SoD) 与双预测的最小实现

功能要点：
- SoD 门控：仅当新读数与上次发送值差异超过阈值时触发“发送”。
- 阈值两种模式：
  - fixed: 白天/夜间使用固定阈值 (delta_day/delta_night)
  - adaptive: 使用滚动窗口标准差 σ，阈值 = k * σ（根据小时选择白天/夜间的最小下限）
- 轻量状态：按节点维护最近观测队列与上次发送值。

建议：温度等单通道触发即可；后续可扩展到多通道或加权融合。
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional, Tuple
import numpy as np


@dataclass
class SoDConfig:
    mode: str = "adaptive"        # "fixed" | "adaptive"
    delta_day: float = 0.5         # 固定模式：白天阈值
    delta_night: float = 0.2       # 固定模式：夜间阈值
    k: float = 1.5                 # 自适应模式：阈值系数
    window: int = 24               # 自适应模式：滚动窗口长度
    min_day: float = 0.2           # 自适应模式：白天最小阈值下限
    min_night: float = 0.1         # 自适应模式：夜间最小阈值下限


class SoDController:
    def __init__(self, config: Optional[SoDConfig] = None):
        self.config: SoDConfig = config or SoDConfig()
        self.history: Deque[float] = deque(maxlen=self.config.window)
        self.last_sent_value: Optional[float] = None

    @staticmethod
    def _is_day(hour: int) -> bool:
        return 6 <= (hour % 24) <= 18

    def _compute_delta(self, hour: int) -> float:
        if self.config.mode == "fixed":
            return self.config.delta_day if self._is_day(hour) else self.config.delta_night

        # adaptive 模式
        if len(self.history) < max(3, self.config.window // 4):
            # 历史不足时回退到固定阈值，避免冷启动误判
            return self.config.delta_day if self._is_day(hour) else self.config.delta_night

        sigma = float(np.std(self.history))
        raw_delta = self.config.k * sigma
        min_floor = self.config.min_day if self._is_day(hour) else self.config.min_night
        return max(raw_delta, min_floor)

    def update_and_should_send(self, value: float, hour: int) -> Tuple[bool, float]:
        """
        更新历史并判定是否触发发送。

        返回: (should_send, used_delta)
        """
        self.history.append(value)
        used_delta = self._compute_delta(hour)

        if self.last_sent_value is None:
            # 首个样本始终发送
            self.last_sent_value = value
            return True, used_delta

        if abs(value - self.last_sent_value) >= used_delta:
            self.last_sent_value = value
            return True, used_delta

        return False, used_delta

    def reset(self) -> None:
        self.history.clear()
        self.last_sent_value = None


