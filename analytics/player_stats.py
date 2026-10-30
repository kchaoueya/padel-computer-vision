"""
Player kinematics (speed, acceleration) and per-player match statistics.

Input: the dataframe returned by DataAnalytics.into_dataframe(), which holds
player positions in meters (player{id}_x, player{id}_y) for each frame.

Why this module exists
----------------------
Differentiating raw positions twice amplifies noise enormously. A position
error of only 4 cm (1 pixel of the projected 2D court) turns into ~1.3 m/s of
speed noise at 30 fps, and into tens of m/s^2 of acceleration noise. So we:

1. remove detection outliers (ID swaps, box glitches),
2. fill short gaps by interpolation (and never differentiate across long gaps),
3. fit a local polynomial around every frame (Savitzky-Golay) and take the
   derivative of the *fit* instead of differencing neighbouring samples.
"""

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


PLAYER_IDS = (1, 2, 3, 4)
MS_TO_KMH = 3.6


@dataclass
class KinematicsConfig:
    """Tunable parameters. Distances in meters, speeds in m/s, time in seconds."""

    # --- cleaning -----------------------------------------------------------
    outlier_distance_m: float = 1.0   # distance from local median => outlier
    outlier_window_s: float = 0.3
    max_gap_s: float = 0.5            # shorter gaps are interpolated

    # --- smoothing / differentiation ---------------------------------------
    # Acceleration is a *second* derivative, so it needs a longer window than
    # speed to keep noise under control (tuned on simulated trajectories).
    smooth_window_s: float = 0.5
    accel_window_s: float = 1.0
    polyorder: int = 2

    # --- physical sanity limits (beyond = tracking error) -------------------
    max_speed_ms: float = 9.0
    max_accel_ms2: float = 10.0

    # --- speed zones (m/s): upper bound of each zone ------------------------
    walk_max_ms: float = 1.5    # ~5.4 km/h
    jog_max_ms: float = 3.0     # ~10.8 km/h
    run_max_ms: float = 4.5     # ~16.2 km/h (above = sprint)

    # --- events -------------------------------------------------------------
    moving_speed_ms: float = 0.5
    accel_threshold_ms2: float = 2.0
    event_min_duration_s: float = 0.3
    sprint_min_duration_s: float = 0.5

    # --- court depth zones, measured from the net (meters) ------------------
    net_zone_m: float = 3.0
    back_zone_m: float = 7.0


def _odd(n: int, minimum: int = 3) -> int:
    n = max(int(n), minimum)
    return n if n % 2 == 1 else n + 1


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of consecutive True values."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(int))
    return list(zip(np.where(diff == 1)[0], np.where(diff == -1)[0]))


def count_events(mask: np.ndarray, min_len: int) -> int:
    """Number of runs of True lasting at least min_len frames."""
    return sum(1 for s, e in _runs(mask) if e - s >= min_len)


def _clean_positions(x, y, fps, cfg):
    """Remove outliers and fill short gaps."""
    sx, sy = pd.Series(x, dtype=float), pd.Series(y, dtype=float)

    win = _odd(round(cfg.outlier_window_s * fps), minimum=5)
    med_x = sx.rolling(win, center=True, min_periods=1).median()
    med_y = sy.rolling(win, center=True, min_periods=1).median()
    bad = (np.hypot(sx - med_x, sy - med_y) > cfg.outlier_distance_m).to_numpy()
    sx[bad] = np.nan
    sy[bad] = np.nan

    # Long gaps stay NaN: differentiating across them would invent motion.
    limit = max(int(round(cfg.max_gap_s * fps)), 1)
    sx = sx.interpolate(method="linear", limit=limit, limit_area="inside")
    sy = sy.interpolate(method="linear", limit=limit, limit_area="inside")
    return sx.to_numpy(), sy.to_numpy()


def _smooth_derivatives(x, y, fps, cfg):
    """Savitzky-Golay position / velocity / acceleration per valid segment."""
    n = len(x)
    dt = 1.0 / fps
    out = {k: np.full(n, np.nan) for k in ("x", "y", "vx", "vy", "ax", "ay")}

    min_win = cfg.polyorder + 2
    vel_win = _odd(round(cfg.smooth_window_s * fps), minimum=min_win)
    acc_win = _odd(round(cfg.accel_window_s * fps), minimum=min_win)
    min_acc_len = max(int(round(0.7 * cfg.accel_window_s * fps)), min_win)
    valid = ~(np.isnan(x) | np.isnan(y))

    def fit_window(wanted, length):
        w = min(wanted, length if length % 2 == 1 else length - 1)
        return w if w > cfg.polyorder else 0

    for start, end in _runs(valid):
        length = end - start
        w_vel = fit_window(vel_win, length)
        if w_vel == 0:
            continue
        w_acc = fit_window(acc_win, length) if length >= min_acc_len else 0

        for axis, arr in (("x", x), ("y", y)):
            seg = arr[start:end]
            out[axis][start:end] = savgol_filter(
                seg, w_vel, cfg.polyorder, deriv=0, delta=dt, mode="interp")
            out["v" + axis][start:end] = savgol_filter(
                seg, w_vel, cfg.polyorder, deriv=1, delta=dt, mode="interp")
            if w_acc:
                out["a" + axis][start:end] = savgol_filter(
                    seg, w_acc, cfg.polyorder, deriv=2, delta=dt, mode="interp")
    return out


def add_player_kinematics(
    df: pd.DataFrame,
    fps: float,
    cfg: Optional[KinematicsConfig] = None,
    player_ids: Iterable[int] = PLAYER_IDS,
) -> pd.DataFrame:
    """
    Add smoothed kinematics columns for each player. Existing columns are kept.

    New columns (per player id):
        player{id}_x_s, _y_s             smoothed position (m)
        player{id}_vx_s, _vy_s           smoothed velocity components (m/s)
        player{id}_speed_ms / _speed_kmh speed
        player{id}_accel_ms2             signed accel along the direction of
                                         motion (+ speeding up, - braking)
        player{id}_accel_total_ms2       magnitude of full acceleration vector
                                         (includes direction changes)
    """
    cfg = cfg or KinematicsConfig()
    out = df.copy()
    new_cols = {}

    for pid in player_ids:
        kx, ky = f"player{pid}_x", f"player{pid}_y"
        if kx not in out.columns or ky not in out.columns:
            continue

        x = pd.to_numeric(out[kx], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(out[ky], errors="coerce").to_numpy(dtype=float)
        x, y = _clean_positions(x, y, fps, cfg)
        d = _smooth_derivatives(x, y, fps, cfg)

        speed = np.hypot(d["vx"], d["vy"])
        a_total = np.hypot(d["ax"], d["ay"])

        # Tangential acceleration = projection of a onto the direction of v.
        with np.errstate(invalid="ignore", divide="ignore"):
            a_tan = (d["vx"] * d["ax"] + d["vy"] * d["ay"]) / speed
        a_tan = np.where(speed < cfg.moving_speed_ms, 0.0, a_tan)
        a_tan = np.where(np.isnan(speed) | np.isnan(a_total), np.nan, a_tan)

        impossible = (speed > cfg.max_speed_ms) | (a_total > cfg.max_accel_ms2)
        for arr in (speed, a_tan, a_total):
            arr[impossible] = np.nan

        new_cols[f"player{pid}_x_s"] = d["x"]
        new_cols[f"player{pid}_y_s"] = d["y"]
        new_cols[f"player{pid}_vx_s"] = d["vx"]
        new_cols[f"player{pid}_vy_s"] = d["vy"]
        new_cols[f"player{pid}_speed_ms"] = speed
        new_cols[f"player{pid}_speed_kmh"] = speed * MS_TO_KMH
        new_cols[f"player{pid}_accel_ms2"] = a_tan
        new_cols[f"player{pid}_accel_total_ms2"] = a_total

    out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)
    out.attrs["fps"] = fps
    return out


def summarize_players(df, fps, cfg=None, player_ids=PLAYER_IDS) -> pd.DataFrame:
    """One row per player with headline numbers. `df` from add_player_kinematics()."""
    cfg = cfg or KinematicsConfig()
    dt = 1.0 / fps
    rows = []

    for pid in player_ids:
        if f"player{pid}_speed_ms" not in df.columns:
            continue

        speed = df[f"player{pid}_speed_ms"].to_numpy(dtype=float)
        acc = df[f"player{pid}_accel_ms2"].to_numpy(dtype=float)
        y = df[f"player{pid}_y_s"].to_numpy(dtype=float)

        ok = ~np.isnan(speed)
        if not ok.any():
            continue

        sp, ac = speed[ok], acc[ok]
        ac_valid = ac[~np.isnan(ac)]
        moving = sp >= cfg.moving_speed_ms
        ev_min = max(int(round(cfg.event_min_duration_s * fps)), 1)
        sp_min = max(int(round(cfg.sprint_min_duration_s * fps)), 1)

        depth = np.abs(y[ok])
        vd = ~np.isnan(depth)
        n_depth = max(int(vd.sum()), 1)

        rows.append({
            "player_id": pid,
            "tracked_time_s": int(ok.sum()) * dt,
            "distance_m": float(sp.sum() * dt),
            "avg_speed_kmh": float(sp.mean() * MS_TO_KMH),
            "avg_speed_moving_kmh": float(sp[moving].mean() * MS_TO_KMH) if moving.any() else 0.0,
            "max_speed_kmh": float(sp.max() * MS_TO_KMH),
            "max_accel_ms2": float(ac_valid.max()) if ac_valid.size else 0.0,
            "max_decel_ms2": float(-ac_valid.min()) if ac_valid.size else 0.0,
            "accel_events": count_events(np.nan_to_num(ac) >= cfg.accel_threshold_ms2, ev_min),
            "decel_events": count_events(np.nan_to_num(ac) <= -cfg.accel_threshold_ms2, ev_min),
            "sprints": count_events(sp >= cfg.run_max_ms, sp_min),
            "net_pct": float(100 * (depth[vd] < cfg.net_zone_m).sum() / n_depth),
            "mid_pct": float(100 * ((depth[vd] >= cfg.net_zone_m) & (depth[vd] <= cfg.back_zone_m)).sum() / n_depth),
            "back_pct": float(100 * (depth[vd] > cfg.back_zone_m).sum() / n_depth),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("player_id").round(2)


def speed_zone_table(df, fps, cfg=None, player_ids=PLAYER_IDS) -> pd.DataFrame:
    """Time and distance per speed zone. Long format."""
    cfg = cfg or KinematicsConfig()
    dt = 1.0 / fps
    edges = [0.0, cfg.walk_max_ms, cfg.jog_max_ms, cfg.run_max_ms, np.inf]
    k = MS_TO_KMH
    labels = [
        f"Walk (<{cfg.walk_max_ms*k:.0f} km/h)",
        f"Jog ({cfg.walk_max_ms*k:.0f}-{cfg.jog_max_ms*k:.0f} km/h)",
        f"Run ({cfg.jog_max_ms*k:.0f}-{cfg.run_max_ms*k:.0f} km/h)",
        f"Sprint (>{cfg.run_max_ms*k:.0f} km/h)",
    ]
    rows = []
    for pid in player_ids:
        col = f"player{pid}_speed_ms"
        if col not in df.columns:
            continue
        speed = df[col].dropna()
        if speed.empty:
            continue
        zones = pd.cut(speed, bins=edges, labels=labels, right=False)
        for label in labels:
            m = (zones == label).to_numpy()
            rows.append({
                "player_id": pid, "zone": label,
                "time_s": float(m.sum() * dt),
                "time_pct": float(100 * m.sum() / len(speed)),
                "distance_m": float(speed.to_numpy()[m].sum() * dt),
            })
    return pd.DataFrame(rows).round(2)
