import numpy as np
from collections import defaultdict, deque
from scipy.spatial import cKDTree


def fit_line_pca(points):
    """
    Fit a 2D line using PCA / total least squares.
    Returns:
        center, direction, slope, intercept, proj_min, proj_max, endpoints
    """
    pts = np.asarray(points, dtype=float)
    center = pts.mean(axis=0)
    X = pts - center

    # PCA in 2D
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    direction = vt[0]
    direction = direction / np.linalg.norm(direction)

    # avoid vertical sign ambiguity
    if direction[0] < 0:
        direction = -direction

    # project onto direction
    t = X @ direction
    tmin, tmax = t.min(), t.max()
    p1 = center + tmin * direction
    p2 = center + tmax * direction

    # slope/intercept
    if abs(direction[0]) < 1e-9:
        slope = np.inf
        intercept = np.nan
    else:
        slope = direction[1] / direction[0]
        intercept = center[1] - slope * center[0]

    return {
        "center": center,
        "direction": direction,
        "slope": slope,
        "intercept": intercept,
        "proj_min": tmin,
        "proj_max": tmax,
        "endpoints": np.vstack([p1, p2]),
        "length": np.linalg.norm(p2 - p1),
    }


def local_tangent_estimates(points, k=10):
    """
    Estimate local tangent angle at each point using PCA of k-nearest neighbors.
    Uses scipy.spatial.cKDTree (no sklearn dependency).
    """
    pts = np.asarray(points, dtype=float)
    k_eff = min(k, len(pts))
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k_eff)
    if idx.ndim == 1:          # k_eff == 1
        idx = idx[:, np.newaxis]

    tangents = []
    for neigh_idx in idx:
        neigh = pts[neigh_idx]
        center = neigh.mean(axis=0)
        X = neigh - center
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        d = vt[0]
        d = d / np.linalg.norm(d)
        if d[0] < 0:
            d = -d
        tangents.append(d)

    tangents = np.asarray(tangents)
    angles = np.arctan2(tangents[:, 1], tangents[:, 0])
    return tangents, angles


def build_negative_slope_graph(points, k=10, d_max=0.05, angle_tol_deg=20):
    """
    Connect nearby points whose local tangents are compatible and whose slope is negative.
    Uses scipy.spatial.cKDTree (no sklearn dependency).
    """
    pts = np.asarray(points, dtype=float)
    tangents, angles = local_tangent_estimates(pts, k=k)

    tree = cKDTree(pts)
    neigh_inds = tree.query_ball_point(pts, r=d_max)

    angle_tol = np.deg2rad(angle_tol_deg)
    graph = defaultdict(set)

    for i, neigh in enumerate(neigh_inds):
        # only keep points whose local tangent slope is negative
        ti = tangents[i]
        if ti[0] == 0:
            continue
        si = ti[1] / ti[0]
        if si >= 0:
            continue

        for j in neigh:
            if i == j:
                continue

            tj = tangents[j]
            if tj[0] == 0:
                continue
            sj = tj[1] / tj[0]
            if sj >= 0:
                continue

            # angle compatibility (up to sign)
            a = abs(angles[i] - angles[j])
            a = min(a, np.pi - a, abs(np.pi - a))
            if a < angle_tol:
                graph[i].add(j)
                graph[j].add(i)

    return graph, tangents, angles


def connected_components(graph, n_points):
    visited = np.zeros(n_points, dtype=bool)
    comps = []

    for i in range(n_points):
        if visited[i]:
            continue
        if i not in graph:
            continue

        q = deque([i])
        visited[i] = True
        comp = []

        while q:
            u = q.popleft()
            comp.append(u)
            for v in graph[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)

        comps.append(comp)

    return comps


def extract_negative_fragments(peaks, k=10, d_max=0.05, angle_tol_deg=20,
                               min_points=6, min_length=0.08):
    """
    Returns fitted negative-slope fragments.
    """
    peaks = np.asarray(peaks, dtype=float)
    graph, tangents, angles = build_negative_slope_graph(
        peaks, k=k, d_max=d_max, angle_tol_deg=angle_tol_deg
    )
    comps = connected_components(graph, len(peaks))

    fragments = []
    for comp in comps:
        if len(comp) < min_points:
            continue
        pts = peaks[comp]
        line = fit_line_pca(pts)
        slope = line["slope"]

        if not np.isfinite(slope):
            continue
        if slope >= 0:
            continue
        if line["length"] < min_length:
            continue

        line["indices"] = comp
        line["points"] = pts
        fragments.append(line)

    return fragments


def point_to_segment_distance(p, a, b):
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-12:
        return np.linalg.norm(p - a)
    t = np.dot(p - a, ab) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj)


def scanned_support_along_segment(scanned, a, b, support_radius=0.03, n_samples=20):
    """
    Sample along candidate bridge and check if scanned cells exist nearby.
    Returns 0.0 immediately if the scanned array is empty.
    """
    scanned = np.asarray(scanned, dtype=float)
    if len(scanned) == 0:
        return 0.0

    ts = np.linspace(0, 1, n_samples)
    samples = a[None, :] * (1 - ts[:, None]) + b[None, :] * ts[:, None]

    tree = cKDTree(scanned)
    count = 0
    for s in samples:
        dist, _ = tree.query(s, k=1)
        if dist <= support_radius:
            count += 1

    return count / n_samples


def endpoint_facing_score(endpoint, tangent, other_endpoint):
    """
    Higher if tangent direction roughly points toward the other endpoint.
    """
    v = other_endpoint - endpoint
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return 0.0
    v = v / nv
    t1 = tangent / (np.linalg.norm(tangent) + 1e-12)
    t2 = -t1
    return max(np.dot(t1, v), np.dot(t2, v))


def _cluster_midpoints(candidates, eps=0.06):
    """
    Simple union-find clustering of bridge midpoints within eps distance.
    Replaces sklearn DBSCAN to remove sklearn dependency.
    """
    n = len(candidates)
    mids = np.array([(c["p1"] + c["p2"]) / 2.0 for c in candidates])

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    tree = cKDTree(mids)
    pairs = tree.query_pairs(r=eps)
    for i, j in pairs:
        union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    return groups


def detect_interdot_bridges(peaks, scanned,
                            d_max=0.05,
                            angle_tol_deg=20,
                            min_points=6,
                            min_length=0.08,
                            min_gap=0.03, max_gap=0.22,
                            support_radius=0.03,
                            min_support=0.5,
                            positive_slope_min=0.1,
                            positive_slope_max=5.0,
                            max_vy_diff=0.3):
    """
    Main routine:
      1) get negative fragments
      2) collect endpoints
      3) pair endpoints into interdot bridge candidates
      4) greedy one-bridge-per-endpoint matching
    """
    fragments = extract_negative_fragments(
        peaks,
        d_max=d_max,
        angle_tol_deg=angle_tol_deg,
        min_points=min_points,
        min_length=min_length,
    )

    # Prepare endpoint list
    endpoints = []
    for fi, frag in enumerate(fragments):
        p1, p2 = frag["endpoints"]
        t = frag["direction"]

        endpoints.append({
            "fragment_id": fi,
            "point": p1,
            "tangent": t,
            "frag": frag,
        })
        endpoints.append({
            "fragment_id": fi,
            "point": p2,
            "tangent": t,
            "frag": frag,
        })

    candidates = []

    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            e1 = endpoints[i]
            e2 = endpoints[j]

            # must come from different fragments
            if e1["fragment_id"] == e2["fragment_id"]:
                continue

            a = e1["point"]
            b = e2["point"]
            gap = np.linalg.norm(b - a)

            if gap < min_gap or gap > max_gap:
                continue

            # Both endpoints must be at a similar Vy level (same honeycomb vertex)
            if abs(a[1] - b[1]) > max_vy_diff:
                continue

            dx = b[0] - a[0]
            dy = b[1] - a[1]
            if abs(dx) < 1e-9:
                continue

            bridge_slope = dy / dx
            if bridge_slope < positive_slope_min or bridge_slope > positive_slope_max:
                continue

            support = scanned_support_along_segment(
                scanned, a, b, support_radius=support_radius
            )
            if support < min_support:
                continue

            face1 = endpoint_facing_score(a, e1["tangent"], b)
            face2 = endpoint_facing_score(b, e2["tangent"], a)

            # penalty if bridge is too parallel to reservoir lines
            s1 = e1["frag"]["slope"]
            s2 = e2["frag"]["slope"]
            slope_sep = abs(bridge_slope - 0.5 * (s1 + s2))

            score = (
                2.0 * support
                + 1.0 * face1
                + 1.0 * face2
                + 0.3 * slope_sep
                - 1.5 * gap
            )

            candidates.append({
                "p1": a,
                "p2": b,
                "slope": bridge_slope,
                "gap": gap,
                "support": support,
                "face1": face1,
                "face2": face2,
                "score": score,
                "frag_ids": (e1["fragment_id"], e2["fragment_id"]),
                "ep_idx": (i, j),
            })

    # merge near-duplicate bridges by clustering midpoints
    if len(candidates) == 0:
        return fragments, []

    groups = _cluster_midpoints(candidates, eps=0.06)

    merged = []
    for indices in groups.values():
        group = [candidates[k] for k in indices]
        best = max(group, key=lambda x: x["score"])
        merged.append(best)

    merged = sorted(merged, key=lambda x: x["score"], reverse=True)

    # Greedy one-bridge-per-endpoint: each fragment endpoint can only be
    # used by one bridge.  Process candidates best-score-first.
    used_endpoints: set = set()
    final = []
    for bridge in merged:
        ei, ej = bridge["ep_idx"]
        if ei not in used_endpoints and ej not in used_endpoints:
            used_endpoints.add(ei)
            used_endpoints.add(ej)
            final.append(bridge)

    return fragments, final
