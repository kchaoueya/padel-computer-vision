"""Draw live speed / acceleration next to each player on the output video."""

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm


def _color_for_speed(kmh: float) -> tuple[int, int, int]:
    """BGR color: green (slow) -> yellow -> red (sprint)."""
    t = float(np.clip(kmh / 18.0, 0.0, 1.0))
    return (0, int(255 * (1 - max(0.0, t - 0.5) * 2)), int(255 * min(1.0, t * 2)))


def draw_stats_overlay(video_in, video_out, players_results, df, player_ids=(1, 2, 3, 4)) -> None:
    """
    Second pass over an already written video: speed/accel tag above each
    tracked player plus a small scoreboard (current and max speed).

    players_results: players tracker `.results` (indexable by frame, iterable of Player)
    df: dataframe returned by add_player_kinematics()
    """
    info = sv.VideoInfo.from_video_path(video_in)
    out = cv2.VideoWriter(
        video_out, cv2.VideoWriter_fourcc(*"mp4v"), float(info.fps), info.resolution_wh,
    )
    cols = {
        pid: (df[f"player{pid}_speed_kmh"].to_numpy(), df[f"player{pid}_accel_ms2"].to_numpy())
        for pid in player_ids
        if f"player{pid}_speed_kmh" in df.columns
    }
    max_so_far = {pid: 0.0 for pid in cols}
    n_rows = len(df)

    for i, frame in enumerate(tqdm(sv.get_video_frames_generator(video_in), total=info.total_frames)):
        if i < n_rows:
            for pid, (speed, _) in cols.items():
                if not np.isnan(speed[i]):
                    max_so_far[pid] = max(max_so_far[pid], float(speed[i]))

            if i < len(players_results):
                for player in players_results[i]:
                    if player.id not in cols:
                        continue
                    speed, acc = cols[player.id]
                    if np.isnan(speed[i]):
                        continue
                    x1, y1 = (int(v) for v in player.xyxy[:2])
                    y1 = max(y1, 50)
                    a = acc[i]
                    color = _color_for_speed(speed[i])
                    cv2.rectangle(frame, (x1, y1 - 46), (x1 + 190, y1), (30, 30, 30), -1)
                    cv2.putText(frame, f"P{player.id}  {speed[i]:.1f} km/h", (x1 + 6, y1 - 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    if not np.isnan(a):
                        cv2.putText(frame, f"{a:+.1f} m/s2", (x1 + 6, y1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            h = 28 * (len(cols) + 1)
            cv2.rectangle(frame, (10, 70), (330, 70 + h), (30, 30, 30), -1)
            cv2.putText(frame, "Player  now    max (km/h)", (18, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            for k, (pid, (speed, _)) in enumerate(cols.items()):
                now = "  -  " if np.isnan(speed[i]) else f"{speed[i]:5.1f}"
                cv2.putText(frame, f"P{pid}     {now}   {max_so_far[pid]:5.1f}",
                            (18, 120 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        out.write(frame)
    out.release()
