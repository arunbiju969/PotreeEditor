"""Functional test for the P0 manager fixes: itc dedup, delete-in-streaming,
touched-count-once, registry snapshot/clear, RLock under concurrency."""
import numpy as np, laspy, threading, tempfile, os, sys
from pointcloud_manager import PointCloudManager

d = tempfile.mkdtemp(prefix="p0test_")
src = os.path.join(d, "src.laz")
out = os.path.join(d, "out.laz")

N = 5000
rng = np.random.default_rng(42)
hdr = laspy.LasHeader(point_format=2, version="1.4")
pts = laspy.LasData(hdr)
pts.x = rng.uniform(0, 100, N).astype(np.float64)
pts.y = rng.uniform(0, 100, N).astype(np.float64)
pts.z = rng.uniform(0, 50, N).astype(np.float64)
pts.intensity = rng.integers(0, 1000, N).astype(np.uint16)
pts.classification = np.full(N, 2, dtype=np.uint8)
pts.write(src)

m = PointCloudManager()
m.load_las_header(src)
assert m.las_data is None, "header-only mode"

# 1) assign_itc on a polygon covering x<50,y<50 -> source has NO itc (dup-dim bug)
m.add_staged_edit({"type": "polygon", "action": "assign_itc", "value": 101,
                   "coords": [[0, 0], [50, 0], [50, 50], [0, 50]]})
# 2) mark_noise box, x>90 column
m.add_staged_edit({"type": "box", "action": "mark_noise", "value": None,
                   "min": [90, 0, 0], "max": [100, 100, 50]})
# 3) delete box x in [70,80]
m.add_staged_edit({"type": "box", "action": "delete",
                   "min": [70, 0, 0], "max": [80, 100, 50]})

res = m.commit_staged_edits(src, out)

# Expected ground truth from the QUANTIZED coords as stored, with the inclusive
# raw-integer box semantics the commit uses (floor/ceil on scale boundaries).
s = laspy.read(src)
x, y, z = s.x, s.y, s.z
in_poly = (x <= 50) & (y <= 50)
in_noise = (x >= 90) & (x <= 100) & (z >= 0) & (z <= 50)
in_del = (x >= 70) & (x <= 80) & (z >= 0) & (z <= 50)
exp_deleted = int(in_del.sum())
exp_touched = int((in_poly | in_noise | in_del).sum())
print(f"expected: deleted={exp_deleted} touched(union)={exp_touched}")
print("commit result:", {k: res[k] for k in ("total_points", "modified_points", "edits_count", "success")})

back = laspy.read(out)
assert res["modified_points"] == exp_touched, f"touched {res['modified_points']} != {exp_touched}"
assert len(back) == N - exp_deleted, f"len {len(back)} != {N-exp_deleted}"
bx, by, bz = back.x, back.y, back.z
assert set(np.unique(back.classification)) <= {2, 7}, np.unique(back.classification)
# itc on survivors: poly->101, noise->9999, rest->0
bx_in_poly = (bx <= 50) & (by <= 50)
assert np.all(back.itc[bx_in_poly] == 101), "poly itc"
bx_noise = bx >= 90
assert np.all(back.itc[bx_noise] == 9999), f"noise itc {np.unique(back.itc[bx_noise])}"
assert np.all(back.classification[bx_noise] == 7), "noise class"
rest = ~bx_in_poly & ~bx_noise & ((bx < 70) | (bx > 80))  # survivors outside poly & noise
assert np.all(back.itc[rest] == 0), "rest itc"
print("LAS content checks: OK")

# registry cleared -> second commit raises
assert m.get_staged_edits() == []
try:
    m.commit_staged_edits(src, out)
    raise AssertionError("expected ValueError")
except ValueError:
    print("registry cleared after commit: OK")

# concurrent staging + reading under lock
def churn(i):
    for j in range(50):
        e = m.add_staged_edit({"type": "polygon", "action": "assign_itc", "value": i * 100 + j,
                               "coords": [[0, 0], [1, 0], [1, 1]]})
        m.get_staged_edits()
        m.remove_staged_edit(e["id"])
ts = [threading.Thread(target=churn, args=(i,)) for i in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
assert m.get_staged_edits() == [], m.get_staged_edits()
print("concurrent staging: OK")
print("ALL TESTS PASSED")
