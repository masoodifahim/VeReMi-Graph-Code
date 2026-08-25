
from __future__ import annotations

from pathlib import Path
import tarfile
import json
import re
import math
import pandas as pd
import numpy as np

LOG_RE = re.compile(
    r"JSONlog-(?P<id1>\d+)-(?P<id2>\d+)-A(?P<aflag>\d+)\.json$"
)


def read_json_lines(tar: tarfile.TarFile, member: tarfile.TarInfo):
    f = tar.extractfile(member)
    if f is None:
        return []
    rows = []
    for raw_line in f:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _vec_component(v, idx, default=np.nan):
    try:
        return float(v[idx])
    except Exception:
        return default


def parse_archive(archive_path: Path):
    """
    Parse one original VeReMi archive.

    Returns
    -------
    logs_meta : DataFrame
        Metadata for every JSONlog file.
    local_states : DataFrame
        Type-2 local vehicle states.
    received : DataFrame
        Type-3 received messages with receiver provenance.
    ground_truth : DataFrame
        Type-4 ground-truth message records.
    """
    log_meta = []
    local_frames = []
    recv_frames = []
    gt = None

    with tarfile.open(archive_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        ground_truth_member = None

        parsed_logs = []
        for m in members:
            short = Path(m.name).name

            if short == "GroundTruthJSONlog.json":
                ground_truth_member = m
                continue

            match = LOG_RE.search(short)
            if match:
                parsed_logs.append({
                    "member": m,
                    "file_name": short,
                    "id1": int(match.group("id1")),
                    "id2": int(match.group("id2")),
                    "A_flag": int(match.group("aflag")),
                })

        if ground_truth_member is None:
            raise ValueError(
                f"GroundTruthJSONlog.json not found in {archive_path.name}"
            )

        gt_rows = read_json_lines(tar, ground_truth_member)
        gt = pd.DataFrame(gt_rows)

        for info in parsed_logs:
            rows = read_json_lines(tar, info["member"])
            if not rows:
                continue

            df = pd.DataFrame(rows)
            if "type" not in df.columns:
                continue

            type2 = df[df["type"] == 2].copy()
            if not type2.empty:
                type2["archive_name"] = archive_path.name
                type2["log_file"] = info["file_name"]
                type2["id1"] = info["id1"]
                type2["receiver_id"] = info["id2"]
                type2["A_flag"] = info["A_flag"]
                local_frames.append(type2)

            type3 = df[df["type"] == 3].copy()
            if not type3.empty:
                type3["archive_name"] = archive_path.name
                type3["receiver_log_file"] = info["file_name"]
                type3["receiver_id"] = info["id2"]
                type3["id1"] = info["id1"]
                type3["A_flag"] = info["A_flag"]
                recv_frames.append(type3)

            log_meta.append({
                "archive_name": archive_path.name,
                "file_name": info["file_name"],
                "id1": info["id1"],
                "receiver_id": info["id2"],
                "A_flag": info["A_flag"],
                "total_records": len(df),
                "type2_records": int((df["type"] == 2).sum()),
                "type3_records": int((df["type"] == 3).sum()),
            })

    logs_meta = pd.DataFrame(log_meta)
    local_states = (
        pd.concat(local_frames, ignore_index=True)
        if local_frames else pd.DataFrame()
    )
    received = (
        pd.concat(recv_frames, ignore_index=True)
        if recv_frames else pd.DataFrame()
    )

    return logs_meta, local_states, received, gt


def flatten_vector_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten position/speed/noise vector columns into numeric columns."""
    out = df.copy()

    vector_specs = {
        "pos": ("pos_x", "pos_y", "pos_z"),
        "noise": ("local_noise_x", "local_noise_y", "local_noise_z"),
        "pos_noise": ("pos_noise_x", "pos_noise_y", "pos_noise_z"),
        "spd": ("spd_x", "spd_y", "spd_z"),
        "spd_noise": ("spd_noise_x", "spd_noise_y", "spd_noise_z"),
    }

    for source, names in vector_specs.items():
        if source not in out.columns:
            continue
        for idx, name in enumerate(names):
            out[name] = out[source].apply(
                lambda v, i=idx: _vec_component(v, i)
            )

    return out


def match_ground_truth(
    received: pd.DataFrame,
    ground_truth: pd.DataFrame,
    time_tolerance: float = 1e-6,
):
    """
    Match type-3 received messages to type-4 ground-truth records using
    (sender, messageID), then validate sendTime == ground-truth time.
    """
    required_recv = {"sender", "messageID", "sendTime", "rcvTime"}
    required_gt = {"sender", "messageID", "time", "attackerType"}

    if not required_recv.issubset(received.columns):
        raise ValueError(
            f"Received data missing: {required_recv - set(received.columns)}"
        )
    if not required_gt.issubset(ground_truth.columns):
        raise ValueError(
            f"Ground truth missing: {required_gt - set(ground_truth.columns)}"
        )

    gt_cols = ["sender", "messageID", "time", "attackerType"]
    for c in ["pos", "pos_noise", "spd", "spd_noise"]:
        if c in ground_truth.columns:
            gt_cols.append(c)

    gt_small = ground_truth[gt_cols].copy()

    # uniqueness audit
    dup_count = int(
        gt_small.duplicated(subset=["sender", "messageID"]).sum()
    )

    merged = received.merge(
        gt_small,
        on=["sender", "messageID"],
        how="left",
        suffixes=("_received", "_groundtruth"),
        indicator=True,
    )

    merged["groundtruth_matched"] = merged["_merge"].eq("both")
    merged["send_gt_time_difference"] = (
        pd.to_numeric(merged["sendTime"], errors="coerce")
        - pd.to_numeric(merged["time"], errors="coerce")
    ).abs()

    merged["groundtruth_time_valid"] = (
        merged["groundtruth_matched"]
        & (merged["send_gt_time_difference"] <= time_tolerance)
    )

    merged["communication_delay"] = (
        pd.to_numeric(merged["rcvTime"], errors="coerce")
        - pd.to_numeric(merged["sendTime"], errors="coerce")
    )

    return merged, dup_count


def build_edge_table(matched: pd.DataFrame) -> pd.DataFrame:
    """
    Build directed sender -> receiver communication edges.

    Important:
    The attack label belongs naturally to the transmitted message/edge,
    therefore `attackerType` is retained as an edge-level target.
    """
    x = flatten_vector_columns(matched)

    cols = [
        "archive_name",
        "sender",
        "receiver_id",
        "messageID",
        "sendTime",
        "rcvTime",
        "communication_delay",
        "RSSI",
        "attackerType",
        "A_flag",
        "receiver_log_file",
        "groundtruth_matched",
        "groundtruth_time_valid",
        "pos_x_received",
        "pos_y_received",
        "pos_noise_x_received",
        "pos_noise_y_received",
        "spd_x_received",
        "spd_y_received",
        "spd_noise_x_received",
        "spd_noise_y_received",
    ]

    # Handle flattened names regardless of merge suffix.
    candidate_map = {
        "pos_x_received": ["pos_received", "pos_x"],
        "pos_y_received": ["pos_received", "pos_y"],
        "spd_x_received": ["spd_received", "spd_x"],
        "spd_y_received": ["spd_received", "spd_y"],
    }

    # Explicitly construct core table first.
    out = pd.DataFrame({
        "archive_name": x["archive_name"],
        "sender": pd.to_numeric(x["sender"], errors="coerce"),
        "receiver": pd.to_numeric(x["receiver_id"], errors="coerce"),
        "messageID": pd.to_numeric(x["messageID"], errors="coerce"),
        "sendTime": pd.to_numeric(x["sendTime"], errors="coerce"),
        "rcvTime": pd.to_numeric(x["rcvTime"], errors="coerce"),
        "communication_delay": pd.to_numeric(
            x["communication_delay"], errors="coerce"
        ),
        "RSSI": pd.to_numeric(x.get("RSSI"), errors="coerce"),
        "attackerType": pd.to_numeric(
            x["attackerType"], errors="coerce"
        ),
        "A_flag": pd.to_numeric(x["A_flag"], errors="coerce"),
        "receiver_log_file": x["receiver_log_file"],
        "groundtruth_matched": x["groundtruth_matched"],
        "groundtruth_time_valid": x["groundtruth_time_valid"],
    })

    # Add reported message state from the received record if present.
    for src, dest in [
        ("pos_received", "reported_pos"),
        ("spd_received", "reported_spd"),
        ("pos_noise_received", "reported_pos_noise"),
        ("spd_noise_received", "reported_spd_noise"),
    ]:
        if src in x.columns:
            out[dest + "_x"] = x[src].apply(lambda v: _vec_component(v, 0))
            out[dest + "_y"] = x[src].apply(lambda v: _vec_component(v, 1))
        else:
            # If no suffix was introduced by merge.
            raw = src.replace("_received", "")
            if raw in x.columns:
                out[dest + "_x"] = x[raw].apply(lambda v: _vec_component(v, 0))
                out[dest + "_y"] = x[raw].apply(lambda v: _vec_component(v, 1))

    out = out.dropna(
        subset=["sender", "receiver", "sendTime", "rcvTime"]
    ).copy()

    out["sender"] = out["sender"].astype("int64")
    out["receiver"] = out["receiver"].astype("int64")
    out["messageID"] = out["messageID"].astype("int64")

    return out.reset_index(drop=True)


def build_node_state_table(local_states: pd.DataFrame) -> pd.DataFrame:
    """
    Build time-stamped node-state observations from type-2 local records.
    """
    if local_states.empty:
        return pd.DataFrame()

    x = flatten_vector_columns(local_states)

    out = pd.DataFrame({
        "archive_name": x["archive_name"],
        "vehicle_id": pd.to_numeric(
            x["receiver_id"], errors="coerce"
        ),
        "time": pd.to_numeric(x["rcvTime"], errors="coerce"),
        "A_flag": pd.to_numeric(x["A_flag"], errors="coerce"),
    })

    for c in [
        "pos_x", "pos_y", "pos_z",
        "spd_x", "spd_y", "spd_z",
        "local_noise_x", "local_noise_y", "local_noise_z",
        "spd_noise_x", "spd_noise_y", "spd_noise_z",
    ]:
        if c in x.columns:
            out[c] = pd.to_numeric(x[c], errors="coerce")

    out = out.dropna(subset=["vehicle_id", "time"]).copy()
    out["vehicle_id"] = out["vehicle_id"].astype("int64")

    if {"spd_x", "spd_y"}.issubset(out.columns):
        out["speed_magnitude"] = np.sqrt(
            out["spd_x"] ** 2 + out["spd_y"] ** 2
        )

    return out.sort_values(
        ["vehicle_id", "time"]
    ).reset_index(drop=True)


def assign_snapshot_ids(
    edges: pd.DataFrame,
    window_size: float,
) -> pd.DataFrame:
    """
    Assign each received message edge to a fixed temporal graph window.
    Snapshot IDs are local to each archive.
    """
    out = edges.copy()

    t0 = float(out["sendTime"].min())
    out["snapshot_id"] = np.floor(
        (out["sendTime"] - t0) / window_size
    ).astype("int64")
    out["snapshot_start"] = (
        t0 + out["snapshot_id"] * window_size
    )
    out["snapshot_end"] = (
        out["snapshot_start"] + window_size
    )

    return out


def archive_summary(
    archive_path: Path,
    logs_meta: pd.DataFrame,
    local_states: pd.DataFrame,
    received: pd.DataFrame,
    ground_truth: pd.DataFrame,
    matched: pd.DataFrame,
    edges: pd.DataFrame,
    duplicate_gt_keys: int,
):
    vehicles = set(edges["sender"].unique()) | set(edges["receiver"].unique())

    return {
        "archive_name": archive_path.name,
        "log_files": len(logs_meta),
        "A0_files": int((logs_meta["A_flag"] == 0).sum()),
        "A1_files": int((logs_meta["A_flag"] == 1).sum()),
        "local_state_rows": len(local_states),
        "received_message_rows": len(received),
        "ground_truth_rows": len(ground_truth),
        "communication_edges": len(edges),
        "unique_vehicles": len(vehicles),
        "unique_senders": edges["sender"].nunique(),
        "unique_receivers": edges["receiver"].nunique(),
        "gt_duplicate_sender_message_keys": duplicate_gt_keys,
        "gt_match_percent": float(
            matched["groundtruth_matched"].mean() * 100
        ),
        "gt_time_valid_percent": float(
            matched["groundtruth_time_valid"].mean() * 100
        ),
        "median_comm_delay": float(
            edges["communication_delay"].median()
        ),
    }
