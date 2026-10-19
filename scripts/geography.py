from pathlib import Path
import sys, pickle, json

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pandas as pd, numpy as np, shapefile, igraph as ig
from scipy.stats import spearmanr
from validate_pipeline import load_master, EX

O = ROOT / "results/paper"
m = load_master()
m = m[~m.index.str[:2].isin(EX)]
parts = pickle.load(open(ROOT / "results/analysis/partitions.pkl", "rb"))
u = parts["core85", 33.471236].astype(str)
c = pd.read_csv(
    ROOT / "data/raw/census/co-est2019-alldata.csv",
    encoding="latin1",
    dtype={"STATE": str, "COUNTY": str},
)
c["geoid"] = c.STATE + c.COUNTY
c = c.set_index("geoid")
# Standard vertex-based queen adjacency on the shared-boundary county shapefile.
from collections import defaultdict

vertices = defaultdict(set)
cent = {}
for sr in shapefile.Reader(
    str(ROOT / "data/reference/maps/UScounties_fixed_46102.shp")
).iterShapeRecords():
    f = sr.record["FIPS"]
    if f in m.index:
        numerator = np.zeros(2)
        denominator = 0.0
        cuts = list(sr.shape.parts) + [len(sr.shape.points)]
        for start, end in zip(cuts[:-1], cuts[1:]):
            xy = np.asarray(sr.shape.points[start:end])
            nxt = np.roll(xy, -1, axis=0)
            cross = xy[:, 0] * nxt[:, 1] - nxt[:, 0] * xy[:, 1]
            numerator += ((xy + nxt) * cross[:, None]).sum(axis=0)
            denominator += 3 * cross.sum()
        cent[f] = tuple(numerator / denominator)
        for point in sr.shape.points:
            vertices[tuple(point)].add(f)
edges = set()
import itertools

for members in vertices.values():
    if len(members) > 1:
        edges.update(itertools.combinations(sorted(members), 2))
adj = ig.Graph.TupleList(list(edges), directed=False)
missing = set(m.index) - set(adj.vs["name"])
adj.add_vertices(sorted(missing))
rows = []
for unit, s in u.groupby(u):
    members = list(s.index)
    rows.append(
        dict(
            unit=unit,
            n_counties=len(members),
            n_states=m.loc[members, "state"].nunique(),
            contiguous=adj.induced_subgraph(members).is_connected(),
            members="; ".join(
                f'{c.loc[f,"CTYNAME"]}, {c.loc[f,"STNAME"]} ({f})' for f in members
            ),
        )
    )
pd.DataFrame(rows).to_csv(O / "M8_units.csv", index=False)
# Geometric county centroids in the supplied longitude/latitude shapefile.
assert set(m.index) == set(cent)
e = pd.read_parquet(ROOT / "data/processed/graphs/edge_tables/graph_2.parquet")
e = e[
    e.geoid_i.isin(m.index)
    & e.geoid_j.isin(m.index)
    & (e.geoid_i != e.geoid_j)
    & (e.weight > 0)
].copy()
a = np.radians(np.array([cent[f] for f in e.geoid_i]))
b = np.radians(np.array([cent[f] for f in e.geoid_j]))
dlon = b[:, 0] - a[:, 0]
dlat = b[:, 1] - a[:, 1]
h = np.sin(dlat / 2) ** 2 + np.cos(a[:, 1]) * np.cos(b[:, 1]) * np.sin(dlon / 2) ** 2
dist = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
w = e.weight.to_numpy()
assert np.isfinite(dist).all()
valid = dist > 0
order = np.argsort(dist)
wm = dist[order][np.searchsorted(np.cumsum(w[order]), w.sum() / 2)]
res = dict(
    phase="2019 mobility",
    support="continental full G2",
    n_edges=len(e),
    spearman_log_weight_log_distance=float(
        spearmanr(np.log(w[valid]), np.log(dist[valid])).statistic
    ),
    flow_weighted_median_km=float(wm),
    unweighted_median_km=float(np.median(dist)),
    contiguous_pct=100 * np.mean([r["contiguous"] for r in rows]),
    cross_state_pct=100 * np.mean([r["n_states"] > 1 for r in rows]),
    n_units=len(rows),
    n_singletons=sum(r["n_counties"] == 1 for r in rows),
    centroid_convention="Area-weighted polygon centroids from supplied shapefile in lon/lat (signed ring areas, holes subtracted); spherical great-circle radius 6371.0088 km; no self edges",
    contiguity_convention="Exact shared polygon vertices (queen); excluded counties cannot bridge components; singleton units contiguous",
)
(O / "M8_geography.json").write_text(json.dumps(res, indent=2))
print(res)
