"""Ingest a LeRobot dataset (v2 parquet) into a vlatrust :class:`TraceSet`.

LeRobot v2 stores one row per timestep with an ``action`` vector, an
``observation.state`` (robot joint state), and an ``episode_index`` that groups
rows into trajectories. We read the parquet with ``pyarrow`` (**lazy import** —
not a core dependency) and assemble one :class:`Trace` per episode.

Crucially, a recorded dataset carries **no policy confidence** (token entropy is
internal to the policy at inference time and is not stored), so every ingested
trace is tagged :data:`ConfidenceSource.NONE`. The abstention/calibration axis
then reports N/A by design — this is the honest boundary between "we have a real
robot trajectory" and "we have the policy's self-reported confidence". The
physics gate (joint-limit validity) *can* still be evaluated on the real joint
states, which is the part of the claim a recorded dataset can falsify on CPU.
"""

from __future__ import annotations

import numpy as np

from ..core.types import ConfidenceSource, Step, Trace, TraceSet

__all__ = ["traceset_from_lerobot_parquet", "traceset_from_rows"]


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ModuleNotFoundError as e:  # pragma: no cover - exercised only without pyarrow
        raise RuntimeError(
            "Reading LeRobot parquet needs pyarrow: install `vlatrust[lerobot]` "
            "(or `pip install pyarrow`)."
        ) from e
    return pq


def traceset_from_rows(
    *,
    actions: np.ndarray,
    episode_index: np.ndarray,
    states: np.ndarray | None = None,
    success_by_episode: dict[int, bool] | None = None,
    joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
    task: str = "lerobot",
) -> TraceSet:
    """Assemble a TraceSet from per-timestep arrays grouped by ``episode_index``.

    ``actions`` is ``(n_steps, action_dim)``; ``episode_index`` is ``(n_steps,)``.
    If ``joint_limits=(low, high)`` is given, a step whose ``observation.state``
    falls outside the limits is marked ``physically_valid=False`` (the only real
    signal the physics gate can read from a recorded dataset).
    """
    actions = np.asarray(actions, dtype=float)
    episode_index = np.asarray(episode_index)
    states_arr = None if states is None else np.asarray(states, dtype=float)

    traces: list[Trace] = []
    for ep in dict.fromkeys(episode_index.tolist()):  # preserve first-seen order
        sel = np.flatnonzero(episode_index == ep)
        steps: list[Step] = []
        for i in sel:
            valid = True
            if joint_limits is not None and states_arr is not None:
                low, high = joint_limits
                st = states_arr[i]
                valid = bool(np.all(st >= low) and np.all(st <= high))
            steps.append(Step(action=actions[i], physically_valid=valid))
        success = None if success_by_episode is None else success_by_episode.get(int(ep))
        traces.append(
            Trace(
                steps=tuple(steps),
                confidence_source=ConfidenceSource.NONE,  # datasets have no policy confidence
                success=success,
                task=task,
                trace_id=f"{task}-ep{int(ep)}",
            )
        )
    return TraceSet(tuple(traces))


def traceset_from_lerobot_parquet(
    path,
    *,
    action_key: str = "action",
    episode_key: str = "episode_index",
    state_key: str | None = "observation.state",
    success_key: str | None = None,
    joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
    task: str = "lerobot",
) -> TraceSet:
    """Load a LeRobot v2 parquet file into a TraceSet (lazy pyarrow).

    ``success_key``, if given, names a per-row column whose value on the *last*
    row of an episode is taken as that episode's success label; absent, traces
    are left unlabelled (``success=None``).
    """
    pq = _require_pyarrow()
    table = pq.read_table(str(path))
    cols = table.column_names

    def _col(name: str) -> np.ndarray:
        return np.array(table.column(name).to_pylist(), dtype=object)

    if action_key not in cols or episode_key not in cols:
        raise KeyError(
            f"parquet missing required columns {action_key!r}/{episode_key!r}; has {cols}"
        )

    actions = np.array([np.asarray(a, dtype=float) for a in table.column(action_key).to_pylist()])
    episode_index = np.asarray(table.column(episode_key).to_pylist())
    states = None
    if state_key is not None and state_key in cols:
        states = np.array([np.asarray(s, dtype=float) for s in table.column(state_key).to_pylist()])

    success_by_episode: dict[int, bool] | None = None
    if success_key is not None and success_key in cols:
        succ = _col(success_key)
        success_by_episode = {}
        for ep in dict.fromkeys(episode_index.tolist()):
            last = np.flatnonzero(episode_index == ep)[-1]
            success_by_episode[int(ep)] = bool(succ[last])

    return traceset_from_rows(
        actions=actions,
        episode_index=episode_index,
        states=states,
        success_by_episode=success_by_episode,
        joint_limits=joint_limits,
        task=task,
    )
