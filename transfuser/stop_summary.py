"""Stops per drive: long waits vs short stops. Usage: python -m transfuser.stop_summary DIR [DIR ...]"""
import csv, json, sys
from collections import Counter
import numpy as np

tot = Counter()
for d in sys.argv[1:]:
    m = json.load(open(f"{d}/drive_metrics.json"))
    r = list(csv.DictReader(open(f"{d}/drive_log.csv")))
    v = np.array([float(x["speed_kmh"]) for x in r])
    tr = [x["light_truth"] for x in r]
    moving = v > 1
    st = ~moving
    e = np.flatnonzero(np.diff(np.r_[0, st.astype(int), 0]))
    seg = [(a, b) for a, b in zip(e[::2], e[1::2]) if b - a >= 4 and a > 40]
    short = [s for s in seg if s[1] - s[0] < 60]
    long_ = [s for s in seg if s[1] - s[0] >= 60]
    false = [s for s in short if Counter(tr[s[0]:s[1]]).most_common(1)[0][0] == "None"]
    dv = np.diff(v) * 20 / 3.6
    hard = int((dv < -6).sum())
    print(f"{d.split('/')[-1]}: dist={m['distance_m']:.0f}m col={m['collisions']} inv={m['lane_invasions']} stopped={m['stopped_pct']}% "
          f"cte_p50={m['cte_p50_m']} cte_p90={m['cte_p90_m']} | short stops={len(short)} (no light {len(false)}) long stops={len(long_)} longest={max([(b-a)/20 for a,b in long_] or [0]):.0f}s hard-brake frames={hard}")
    tot.update(dist=int(m["distance_m"]), col=m["collisions"], inv=m["lane_invasions"], short=len(short), false=len(false), long=len(long_), hard=hard)
print(dict(tot))
