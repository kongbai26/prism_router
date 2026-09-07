"""全局运行状态 — 替代 __init__.py 中的可变全局变量"""

from __future__ import annotations


class ServiceStatus:
    """服务组件状态跟踪（分类器 / 改写器）"""

    def __init__(self, failure_threshold: int = 3):
        self.ok: bool = True
        self.consecutive_failures: int = 0
        self.failure_threshold: int = failure_threshold

    def set_ok(self, ok: bool) -> None:
        self.ok = ok

    def is_ok(self) -> bool:
        return self.ok

    def record_failure(self, threshold: int | None = None) -> bool:
        """记录失败，返回是否应禁用"""
        effective_threshold = threshold if threshold is not None else self.failure_threshold
        self.consecutive_failures += 1
        if self.consecutive_failures >= effective_threshold:
            self.ok = False
            return True
        return False

    def record_success(self) -> None:
        """成功，重置失败计数"""
        self.consecutive_failures = 0
        self.ok = True


# 模块级单例
classifier_status = ServiceStatus(failure_threshold=3)
rewriter_status = ServiceStatus(failure_threshold=3)


# ── 对外接口（保持向后兼容） ──


def set_classifier_ok(ok: bool) -> None:
    classifier_status.set_ok(ok)


def is_classifier_ok() -> bool:
    return classifier_status.is_ok()


def record_classifier_failure(threshold: int | None = None) -> bool:
    return classifier_status.record_failure(threshold)


def record_classifier_success() -> None:
    classifier_status.record_success()


def set_rewriter_ok(ok: bool) -> None:
    rewriter_status.set_ok(ok)


def is_rewriter_ok() -> bool:
    return rewriter_status.is_ok()
