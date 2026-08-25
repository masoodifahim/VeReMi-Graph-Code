
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import networkx as nx


def _safe_num(s):
    return pd.to_numeric(s, errors="coerce")


def add_snapshot_to_nodes(nodes: pd.DataFrame, edges: pd.DataFrame, window_size: float):
    """
    Assign local type-2 state records to the same temporal grid used by edges.
    Uses the archive's minimum edge sendTime as t0.
    """
    out = nodes.copy()
    if out.empty or edges.empty:
        return out

    t0 = float(edges["sendTime"].min())
    out["snapshot_id"] = np.floor(
        (_safe_num(out["time"]) - t0) / window_size
    ).astype("Int64")

    return out


def latest_node_state_per_snapshot(nodes_with_snapshot: pd.DataFrame):
    """
    Keep the latest local state for each vehicle within each snapshot.
    This is the snapshot-level node state representation.
    """
    if nodes_with_snapshot.empty:
        return pd.DataFrame()

    x = nodes_with_snapshot.dropna(
        subset=["snapshot_id", "vehicle_id", "time"]
    ).copy()

    x = x.sort_values(
        ["snapshot_id", "vehicle_id", "time"]
    )

    latest = (
        x.groupby(
            ["snapshot_id", "vehicle_id"],
            as_index=False,
            sort=False
        )
        .tail(1)
        .reset_index(drop=True)
    )

    # Mobility-derived attributes supported by the original raw VeReMi data.
    if {"spd_x", "spd_y"}.issubset(latest.columns):
        latest["speed_magnitude"] = np.sqrt(
            latest["spd_x"] ** 2 + latest["spd_y"] ** 2
        )
        latest["motion_angle"] = np.arctan2(
            latest["spd_y"], latest["spd_x"]
        )

    # Temporal changes across graph snapshots for the same vehicle.
    latest = latest.sort_values(
        ["vehicle_id", "snapshot_id"]
    ).reset_index(drop=True)

    g = latest.groupby("vehicle_id", sort=False)

    latest["delta_speed_magnitude"] = (
        g["speed_magnitude"].diff()
        if "speed_magnitude" in latest.columns
        else np.nan
    )

    if {"pos_x", "pos_y"}.issubset(latest.columns):
        dx = g["pos_x"].diff()
        dy = g["pos_y"].diff()
        latest["snapshot_displacement"] = np.sqrt(dx ** 2 + dy ** 2)

    if "motion_angle" in latest.columns:
        raw = g["motion_angle"].diff()
        latest["delta_motion_angle"] = np.arctan2(
            np.sin(raw), np.cos(raw)
        )

    return latest


def add_receiver_state_to_edges(
    edges: pd.DataFrame,
    node_states: pd.DataFrame,
):
    """
    Attach the latest receiver-local state available at or before rcvTime.

    This is causal and operationally observable. We deliberately DO NOT
    attach sender ground-truth state, because that would introduce information
    unavailable to a real receiver.
    """
    if edges.empty:
        return edges.copy()

    e = edges.copy()
    n = node_states.copy()

    e["rcvTime"] = _safe_num(e["rcvTime"])
    e["receiver"] = _safe_num(e["receiver"]).astype("Int64")

    n["time"] = _safe_num(n["time"])
    n["vehicle_id"] = _safe_num(n["vehicle_id"]).astype("Int64")

    keep = [
        c for c in [
            "vehicle_id", "time",
            "pos_x", "pos_y",
            "spd_x", "spd_y",
            "speed_magnitude"
        ]
        if c in n.columns
    ]

    n = n[keep].dropna(
        subset=["vehicle_id", "time"]
    ).copy()

    # merge_asof requires global sort by the 'on' key.
    e = e.sort_values(["rcvTime", "receiver"]).reset_index(drop=True)
    n = n.sort_values(["time", "vehicle_id"]).reset_index(drop=True)

    merged = pd.merge_asof(
        e,
        n,
        left_on="rcvTime",
        right_on="time",
        left_by="receiver",
        right_by="vehicle_id",
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_receiver_state")
    )

    rename = {}
    for c in ["pos_x", "pos_y", "spd_x", "spd_y", "speed_magnitude"]:
        if c in merged.columns:
            rename[c] = f"receiver_{c}"

    merged = merged.rename(columns=rename)

    merged["receiver_state_age"] = (
        merged["rcvTime"] - merged["time"]
    )

    return merged


def engineer_edge_features(edges_with_receiver_state: pd.DataFrame):
    """
    Create message/edge features using only information available to the receiver
    plus past communication history.

    `attackerType` remains the target and is never used to compute predictors.
    """
    x = edges_with_receiver_state.copy()

    # Reported sender-state features already reconstructed from type-3 messages.
    if {"reported_spd_x", "reported_spd_y"}.issubset(x.columns):
        x["reported_sender_speed_magnitude"] = np.sqrt(
            x["reported_spd_x"] ** 2 + x["reported_spd_y"] ** 2
        )

    if {
        "reported_pos_x", "reported_pos_y",
        "receiver_pos_x", "receiver_pos_y"
    }.issubset(x.columns):
        x["reported_distance_to_receiver"] = np.sqrt(
            (x["reported_pos_x"] - x["receiver_pos_x"]) ** 2
            + (x["reported_pos_y"] - x["receiver_pos_y"]) ** 2
        )

    if {
        "reported_spd_x", "reported_spd_y",
        "receiver_spd_x", "receiver_spd_y"
    }.issubset(x.columns):
        x["relative_velocity_magnitude"] = np.sqrt(
            (x["reported_spd_x"] - x["receiver_spd_x"]) ** 2
            + (x["reported_spd_y"] - x["receiver_spd_y"]) ** 2
        )

    if "communication_delay" not in x.columns:
        x["communication_delay"] = (
            _safe_num(x["rcvTime"]) - _safe_num(x["sendTime"])
        )

    # Pair-history features: strictly past rows only.
    x = x.sort_values(
        ["sender", "receiver", "rcvTime", "messageID"]
    ).reset_index(drop=True)

    pair_group = x.groupby(
        ["sender", "receiver"],
        sort=False
    )

    x["prior_pair_message_count"] = pair_group.cumcount()
    x["time_since_previous_pair_message"] = pair_group["rcvTime"].diff()

    # Sender-wide and receiver-wide historical activity.
    x = x.sort_values(
        ["sender", "rcvTime", "messageID"]
    ).reset_index(drop=True)
    x["prior_sender_message_count"] = (
        x.groupby("sender", sort=False).cumcount()
    )

    x = x.sort_values(
        ["receiver", "rcvTime", "messageID"]
    ).reset_index(drop=True)
    x["prior_receiver_message_count"] = (
        x.groupby("receiver", sort=False).cumcount()
    )

    return x.sort_values(
        ["snapshot_id", "rcvTime", "messageID"]
    ).reset_index(drop=True)


def topology_features_for_snapshot(
    snapshot_edges: pd.DataFrame,
    active_nodes=None
):
    """
    Compute node-level and graph-level topology features
    for one temporal snapshot.

    The graph contains:
      1. vehicles with an active local Type-2 state, and
      2. vehicles appearing as sender or receiver.

    Active vehicles with no communication edges are retained
    as isolated graph nodes.
    """

    # --------------------------------------------------------
    # 1. ACTIVE NODE SET
    # --------------------------------------------------------

    if active_nodes is None:
        active_nodes = []

    active_nodes = {
        int(v)
        for v in active_nodes
        if pd.notna(v)
    }


    # --------------------------------------------------------
    # 2. BUILD DIRECTED COMMUNICATION GRAPH
    # --------------------------------------------------------

    dg = nx.DiGraph()

    # Add active vehicles first.
    # This preserves isolated vehicles.
    dg.add_nodes_from(active_nodes)

    if not snapshot_edges.empty:

        pairs = (
            snapshot_edges[
                ["sender", "receiver"]
            ]
            .dropna()
            .drop_duplicates()
            .astype("int64")
        )

        dg.add_edges_from(
            pairs.itertuples(
                index=False,
                name=None
            )
        )


    # Undirected representation for clustering/components.
    ug = dg.to_undirected()

    nodes = sorted(dg.nodes())

    n = len(nodes)

    unique_edges = dg.number_of_edges()


    # --------------------------------------------------------
    # 3. NODE DEGREE FEATURES
    # --------------------------------------------------------

    indeg = dict(dg.in_degree())

    outdeg = dict(dg.out_degree())

    degree = dict(ug.degree())


    # --------------------------------------------------------
    # 4. PAGERANK
    # --------------------------------------------------------

    if n > 0:
        pagerank = nx.pagerank(dg)
    else:
        pagerank = {}


    # --------------------------------------------------------
    # 5. CLUSTERING
    # --------------------------------------------------------

    if n > 0:
        clustering = nx.clustering(ug)
    else:
        clustering = {}


    # --------------------------------------------------------
    # 6. CONNECTED COMPONENTS
    # --------------------------------------------------------

    if n > 0:
        components = list(
            nx.connected_components(ug)
        )
    else:
        components = []

    component_size = {}

    for component in components:

        size = len(component)

        for vehicle in component:
            component_size[vehicle] = size


    # --------------------------------------------------------
    # 7. NODE-LEVEL TOPOLOGY TABLE
    # --------------------------------------------------------

    denom = max(n - 1, 1)

    node_topology = pd.DataFrame({

        "vehicle_id": nodes,

        "in_degree": [
            indeg.get(v, 0)
            for v in nodes
        ],

        "out_degree": [
            outdeg.get(v, 0)
            for v in nodes
        ],

        "degree": [
            degree.get(v, 0)
            for v in nodes
        ],

        "degree_centrality": [
            degree.get(v, 0) / denom
            for v in nodes
        ],

        "pagerank": [
            pagerank.get(v, 0.0)
            for v in nodes
        ],

        "clustering_coefficient": [
            clustering.get(v, 0.0)
            for v in nodes
        ],

        "component_size": [
            component_size.get(v, 1)
            for v in nodes
        ]
    })


    # --------------------------------------------------------
    # 8. GRAPH DENSITY
    # --------------------------------------------------------

    if n > 1:
        density = float(nx.density(dg))
    else:
        density = 0.0


    # --------------------------------------------------------
    # 9. RECIPROCITY
    # --------------------------------------------------------
    # NetworkX reciprocity is undefined for graphs
    # containing zero edges.

    if unique_edges == 0:

        reciprocity = 0.0

    else:

        reciprocity = nx.reciprocity(dg)

        if reciprocity is None:
            reciprocity = 0.0

        reciprocity = float(reciprocity)


    # --------------------------------------------------------
    # 10. AVERAGE DEGREE
    # --------------------------------------------------------

    if degree:

        average_degree = float(
            np.mean(
                list(degree.values())
            )
        )

    else:
        average_degree = 0.0


    # --------------------------------------------------------
    # 11. AVERAGE CLUSTERING
    # --------------------------------------------------------

    if n > 0:

        average_clustering = float(
            nx.average_clustering(ug)
        )

    else:
        average_clustering = 0.0


    # --------------------------------------------------------
    # 12. COMPONENT STATISTICS
    # --------------------------------------------------------

    largest_component_size = max(
        (
            len(component)
            for component in components
        ),
        default=0
    )

    if n > 0:

        largest_component_fraction = (
            largest_component_size / n
        )

    else:
        largest_component_fraction = 0.0


    # --------------------------------------------------------
    # 13. ISOLATED NODES
    # --------------------------------------------------------

    isolated_node_count = sum(
        1
        for vehicle in nodes
        if degree.get(vehicle, 0) == 0
    )


    # --------------------------------------------------------
    # 14. RSSI SUMMARY
    # --------------------------------------------------------

    if (
        not snapshot_edges.empty
        and "RSSI" in snapshot_edges.columns
    ):

        mean_rssi = float(
            pd.to_numeric(
                snapshot_edges["RSSI"],
                errors="coerce"
            ).mean()
        )

    else:
        mean_rssi = np.nan


    # --------------------------------------------------------
    # 15. COMMUNICATION DELAY SUMMARY
    # --------------------------------------------------------

    if (
        not snapshot_edges.empty
        and
        "communication_delay"
        in snapshot_edges.columns
    ):

        median_communication_delay = float(
            pd.to_numeric(
                snapshot_edges[
                    "communication_delay"
                ],
                errors="coerce"
            ).median()
        )

    else:
        median_communication_delay = np.nan


    # --------------------------------------------------------
    # 16. GRAPH-LEVEL FEATURE RECORD
    # --------------------------------------------------------

    graph_stats = {

        "node_count":
            n,

        "isolated_node_count":
            int(isolated_node_count),

        "message_edge_count":
            int(len(snapshot_edges)),

        "unique_directed_edge_count":
            int(unique_edges),

        "density":
            density,

        "reciprocity":
            reciprocity,

        "average_degree":
            average_degree,

        "average_clustering":
            average_clustering,

        "connected_components":
            int(len(components)),

        "largest_component_size":
            int(largest_component_size),

        "largest_component_fraction":
            float(largest_component_fraction),

        "mean_RSSI":
            mean_rssi,

        "median_communication_delay":
            median_communication_delay
    }


    # --------------------------------------------------------
    # 17. ATTACK FRACTION
    # Descriptive only — not a predictive feature.
    # --------------------------------------------------------

    if (
        not snapshot_edges.empty
        and
        "attackerType"
        in snapshot_edges.columns
    ):

        attack_labels = pd.to_numeric(
            snapshot_edges["attackerType"],
            errors="coerce"
        )

        graph_stats[
            "labelled_attack_edge_fraction"
        ] = float(
            (attack_labels != 0).mean()
        )

    else:

        graph_stats[
            "labelled_attack_edge_fraction"
        ] = np.nan


    # --------------------------------------------------------
    # 18. RETURN
    # --------------------------------------------------------

    return node_topology, graph_stats

def compute_archive_topology(
    edges: pd.DataFrame,
    snapshot_nodes: pd.DataFrame
):
    node_parts = []
    graph_rows = []

    snapshot_ids = sorted(
        set(
            edges["snapshot_id"]
            .dropna()
            .astype(int)
        )
        |
        set(
            snapshot_nodes["snapshot_id"]
            .dropna()
            .astype(int)
        )
    )

    for snapshot_id in snapshot_ids:

        snap_edges = edges[
            edges["snapshot_id"] == snapshot_id
        ]

        snap_nodes = snapshot_nodes[
            snapshot_nodes["snapshot_id"] == snapshot_id
        ]

        active_nodes = (
            snap_nodes["vehicle_id"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        nt, gs = topology_features_for_snapshot(
            snap_edges,
            active_nodes
        )

        if not nt.empty:
            nt["snapshot_id"] = int(snapshot_id)
            node_parts.append(nt)

        gs["snapshot_id"] = int(snapshot_id)

        if not snap_edges.empty:
            gs["snapshot_start"] = float(
                snap_edges["snapshot_start"].iloc[0]
            )
            gs["snapshot_end"] = float(
                snap_edges["snapshot_end"].iloc[0]
            )
        else:
            gs["snapshot_start"] = np.nan
            gs["snapshot_end"] = np.nan

        graph_rows.append(gs)

    node_topology = (
        pd.concat(
            node_parts,
            ignore_index=True
        )
        if node_parts
        else pd.DataFrame()
    )

    graph_features = pd.DataFrame(
        graph_rows
    )

    return node_topology, graph_features


def add_lagged_topology(node_topology: pd.DataFrame):
    """
    Produce model-safe lagged topology. These features describe the vehicle's
    topology in the PREVIOUS snapshot, avoiding use of future/current messages
    for a real-time prediction at the start of the current snapshot.
    """
    if node_topology.empty:
        return node_topology.copy()

    x = node_topology.sort_values(
        ["vehicle_id", "snapshot_id"]
    ).copy()

    topology_cols = [
        "in_degree", "out_degree", "degree",
        "degree_centrality", "pagerank",
        "clustering_coefficient", "component_size"
    ]

    g = x.groupby("vehicle_id", sort=False)

    for col in topology_cols:
        x[f"prev_{col}"] = g[col].shift(1)

    return x


def add_lagged_graph_features(graph_features: pd.DataFrame):
    if graph_features.empty:
        return graph_features.copy()

    x = graph_features.sort_values("snapshot_id").copy()

    non_label_cols = [
        "node_count",
        "message_edge_count",
        "unique_directed_edge_count",
        "density",
        "reciprocity",
        "average_degree",
        "average_clustering",
        "connected_components",
        "largest_component_size",
        "largest_component_fraction",
        "mean_RSSI",
        "median_communication_delay",
    ]

    for col in non_label_cols:
        if col in x.columns:
            x[f"prev_{col}"] = x[col].shift(1)

    return x


def merge_node_state_and_topology(
    snapshot_nodes: pd.DataFrame,
    topology: pd.DataFrame,
):
    if snapshot_nodes.empty:
        return topology.copy()
    if topology.empty:
        return snapshot_nodes.copy()

    return snapshot_nodes.merge(
        topology,
        on=["snapshot_id", "vehicle_id"],
        how="outer",
        validate="one_to_one"
    )


def validation_summary(
    node_features: pd.DataFrame,
    edge_features: pd.DataFrame,
    graph_features: pd.DataFrame,
):
    rows = []

    def add_table(name, df):
        numeric = df.select_dtypes(include=[np.number])

        rows.append({
            "table": name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "null_cells": int(df.isna().sum().sum()),
            "infinite_cells": int(
                np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()
            ) if len(numeric.columns) else 0,
        })

    add_table("node_features", node_features)
    add_table("edge_features", edge_features)
    add_table("graph_features", graph_features)

    return pd.DataFrame(rows)
