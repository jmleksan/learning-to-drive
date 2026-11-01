"""Per-drive turn outcomes from drive logs. Usage: python -m transfuser.turn_summary DIR [DIR ...]"""
import csv, json, sys
import numpy as np

total = {"taken": 0, "missed": 0, "unfinished": 0}
for d in sys.argv[1:]:
    m = json.load(open(f"{d}/drive_metrics.json"))
    r = list(csv.DictReader(open(f"{d}/drive_log.csv")))
    tpy = np.array([float(x["target_y"]) for x in r])
    cte = np.abs([float(x["cte_m"]) for x in r])
    events, start, side, left_route = [], None, 0, None
    for i in range(len(r)):
        if cte[i] >= 2.5:
            if start is not None:
                events.append((round(start / 20, 1), side, "missed"))
            else:
                events.append((round(i / 20, 1), "?", "missed"))
            left_route = round(i / 20, 1)
            break
        if start is None and abs(tpy[i]) > 6:
            start, side = i, "R" if tpy[i] > 0 else "L"
        elif start is not None and abs(tpy[i]) < 3:
            events.append((round(start / 20, 1), side, "taken"))
            start = None
    else:
        if start is not None:
            events.append((round(start / 20, 1), side, "unfinished"))
    for e in events:
        total[e[2]] += 1
    print(f"{d.split('/')[-1]}: spawn={m['spawn_index']} dist={m['distance_m']:.0f}m col={m['collisions']} inv={m['lane_invasions']} "
          f"stopped={m['stopped_pct']}% left_route_t={left_route} (of {len(r)/20:.0f}s) cte_p50={m['cte_p50_m']}")
    print("   ", events)
print(total)
