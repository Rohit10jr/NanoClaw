"""Cron service for scheduled agent tasks. (nanobot cron/)

Jobs live in ``<workspace>/cron/jobs.json``. A user job is bound to the chat
session it was created in (the ``cron`` tool) and runs there as a normal turn
(:func:`nanoclaw.cron.bound_runner.run_bound_cron_job`). Protected system jobs
(``payload.kind == "system_event"``) are registered by the host with
``CronService.register_system_job``.
"""

from typing import TYPE_CHECKING, Any

from nanoclaw.cron.types import CronJob, CronSchedule

if TYPE_CHECKING:
    from nanoclaw.cron.service import CronService

__all__ = ["CronService", "CronJob", "CronSchedule"]

_LAZY = {"CronService": ".service"}


def __getattr__(name: str) -> Any:
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    mod = import_module(module_path, __name__)
    val = getattr(mod, name)
    globals()[name] = val
    return val
