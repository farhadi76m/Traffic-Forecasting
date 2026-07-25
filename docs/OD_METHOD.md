# How the OD matrix is built — notes

## 0. The one-line summary

**Neshan is a stopwatch, not a car counter.**
The *shape* of the OD matrix comes from a gravity model over OSM land use.
Neshan only fixes the *total scale* (how many trips/day).

---

## 1. Notation

| Symbol | Meaning | Where it comes from |
|---|---|---|
| `P_i` | trips **produced** by zone i (people leaving home) | OSM: count of residential buildings in zone i |
| `A_j` | trips **attracted** to zone j (jobs, shops, schools) | OSM: count of workplace/retail/school features |
| `c_ij` | travel cost i→j, in **minutes**, on an EMPTY net | SUMO shortest path (`getOptimalPath(fastest=True)`) |
| `f(c)` | deterrence — how much distance discourages a trip | formula below |
| `β` | decay rate, per minute (default 0.12) | assumption (textbook range 0.08–0.15) |
| `T_ij` | trips from i to j — **this is the OD matrix** | output |
| `D` | total trips per day | calibrated against Neshan |

---

## 2. The deterrence function

```
f(c_ij) = exp( −β · c_ij )
```

Meaning: a far zone is exponentially less attractive.
With β = 0.12 per minute:

| travel time | f | interpretation |
|---|---|---|
| 2 min | exp(−0.24) = **0.787** | very attractive |
| 10 min | exp(−1.20) = **0.301** | 2.6× less likely |
| 15 min | exp(−1.80) = **0.165** | 4.8× less likely |

Bigger β → people only make short trips. Smaller β → long trips are common.

---

## 3. The gravity model

### 3a. Simple form (production-constrained) — learn this one

> Every trip leaving zone i is distributed among destinations j,
> in proportion to how attractive j is and how close it is.

```
                  A_j · f(c_ij)
T_ij  =  P_i · ──────────────────
                 Σ_k A_k · f(c_ik)
```

The denominator just makes the shares sum to 1, so `Σ_j T_ij = P_i` exactly.

### 3b. What the code actually uses (doubly-constrained / Furness)

Also forces the **columns** to match the jobs: `Σ_i T_ij = A_j`.
You cannot satisfy both with one division, so you alternate until it settles:

```
T_ij = a_i · P_i · b_j · A_j · f(c_ij)

repeat until converged:
    a_i = P_i / Σ_j ( b_j · A_j · f(c_ij) )      ← fix the rows
    b_j = A_j / Σ_i ( a_i · P_i · f(c_ij) )      ← fix the columns
```

This is IPF / Furness. `furness()` in `gravity_od.py` (~50 iterations).

---

## 4. WORKED EXAMPLE (do this by hand)

Three zones. Zone A is residential; B and C have the jobs.

```
Homes    P_A = 80          Jobs   A_A = 10,  A_B = 50,  A_C = 40
Travel time from A:   c_AA = 2 min,  c_AB = 10 min,  c_AC = 15 min
β = 0.12
```

**Step 1 — deterrence**
```
f(2)  = exp(−0.12 × 2)  = 0.787
f(10) = exp(−0.12 × 10) = 0.301
f(15) = exp(−0.12 × 15) = 0.165
```

**Step 2 — attractiveness × closeness, for each destination**
```
A_A · f = 10 × 0.787 =  7.87
A_B · f = 50 × 0.301 = 15.05      ← B wins: lots of jobs, still reachable
A_C · f = 40 × 0.165 =  6.60      ← C has jobs but is too far
                        ───────
                 sum  = 29.52
```

**Step 3 — shares**
```
A → A :  7.87 / 29.52 = 0.267   (26.7 %)
A → B : 15.05 / 29.52 = 0.510   (51.0 %)
A → C :  6.60 / 29.52 = 0.224   (22.4 %)
```

**Step 4 — trips (multiply by P_A = 80)**
```
T_A→A = 80 × 0.267 = 21.3
T_A→B = 80 × 0.510 = 40.8   ←  the biggest flow
T_A→C = 80 × 0.224 = 17.9
                     ─────
              total = 80.0  ✓  (all 80 people went somewhere)
```

**Lesson:** C has 4× the jobs of A, but only slightly more trips — because it is
15 minutes away. Distance beat job count. That is the gravity model in one line.

---

## 5. Splitting into 24 hours

```
T_ij(h) = D · hourfrac(h) · [ w(h) · G_ij  +  (1 − w(h)) · G_ji ]
```

- `G` = the normalised gravity matrix from §3 (home→work direction)
- `hourfrac(h)` = share of the day's trips in hour h (`HOUR_PROFILE`)
- `w(h)` = share going *outbound* in hour h (`OUTBOUND`)
  - morning (07:00): w ≈ 0.9 → almost all home → work
  - evening (18:00): w ≈ 0.15 → mostly work → home (**the transpose `G_ji`**)

Then Poisson-sample each cell for day-to-day randomness:
```
count_ij(h) ~ Poisson( T_ij(h) )
```

---

## 6. Calibration with Neshan — the only measured step

**Congestion index** (both sides computed on the SAME routes):

```
                 freeflow_time            (empty roads, from SUMO)
     R  =  ─────────────────────────
                  actual_time             (real traffic)
```

- `R_observed` — `actual_time` comes from **Neshan** ← MEASURED
- `R_simulated(D)` — `actual_time` comes from **SUMO** with demand `D`

Find `D` such that:
```
     R_simulated(D)  =  R_observed
```

More cars → more jam → lower R. It is **monotone**, so bisect:

```
Neshan says R_observed = 0.56   (traffic at 56 % of free flow)

D = 100,000  → R_sim = 0.95   too empty  → increase D
D = 316,000  → R_sim = 0.65   still empty → increase D
D = 562,000  → R_sim = 0.44   too jammed → decrease D
D = 421,000  → R_sim = 0.59   ✓ close enough → STOP

Answer: D ≈ 421,000 trips/day
```

**Worked ratio example**
```
Zone 3_1 → zone 1_2
  SUMO, empty roads : 600 s  (10 min)
  Neshan, real now  : 1080 s (18 min)
  R = 600 / 1080 = 0.56
```

---

## 7. Neshan API — exact request

Get a free key: https://platform.neshan.org

```bash
curl -H "Api-Key: $NESHAN_API_KEY" \
  "https://api.neshan.org/v4/direction?type=car&origin=35.7536,51.3242&destination=35.7789,51.3358"
```

- `origin` / `destination` = `lat,lon` (**latitude first**)
- `type=car`
- key goes in the **header**, not the query string

**Response (trimmed):**
```json
{
  "routes": [{
    "legs": [{
      "summary": "بزرگراه همت",
      "distance": { "value": 8450, "text": "۸.۴ کیلومتر" },
      "duration": { "value": 1080, "text": "۱۸ دقیقه" },
      "steps": [ ... ]
    }]
  }]
}
```

The only field that matters:
```
routes[0].legs[0].duration.value  →  1080   (seconds, WITH real traffic)
```

That is `actual_time`. Everything else in the response is ignored.

**Errors:** `{"status":"ERROR","code":480,"message":"API Key not found or is not valid."}`

---

## 8. What is real and what is not

| Quantity | Source | Trust |
|---|---|---|
| `c_ij` travel cost | SUMO net | real |
| `P_i`, `A_j` | OpenStreetMap | real |
| `actual_time` | Neshan | **MEASURED** |
| `D` total trips/day | calibrated | order-of-magnitude |
| `T_ij` individual cell | gravity model | **INFERRED — not measured** |
| `β` | assumption | guess |

**The hard limit:** Neshan gives *time*, not *volume*. Two different demand
patterns can produce identical congestion, and nothing here can tell them apart.
Only real counts (municipal loop detectors, or manual intersection counts fed to
SUMO's `routeSampler.py`) can pin down individual cells.

Use for: "what if we add a road / demand grows 20 %".
Do NOT use for: "exactly how many vehicles cross this link at 08:00".

---

## 9. Running the whole thing on ANOTHER map

Every script takes the net/TAZ as arguments — nothing is Tehran-specific except
the Neshan provider (Iran only; outside Iran swap step 4 for
`fetch_speeds.py --provider tomtom|here`).

```bash
AREA=myarea        # pick a name

# 0. OSM extract -> SUMO net   (bbox order is lon_min,lat_min,lon_max,lat_max!)
wget -O $AREA.osm "https://overpass-api.de/api/map?bbox=51.25,35.68,51.45,35.80"
netconvert --osm-files $AREA.osm -o $AREA.net.xml \
    --geometry.remove --junctions.join --ramps.guess --tls.guess-signals \
    --keep-edges.by-vclass passenger --remove-edges.isolated

# 1. TAZ from real admin boundaries (polygon zones)
#    FIRST on any new map: see which admin_levels exist there (differs per country)
python scripts/build_taz_districts.py $AREA.net.xml --list-levels
python scripts/build_taz_districts.py $AREA.net.xml -l 11 -o $AREA/taz
#    (or a plain grid: python scripts/build_taz.py --net $AREA.net.xml --grid 3 ...)

# 2. zone-to-zone freeflow cost matrix (own net, offline)
python scripts/cost_matrix.py --net $AREA.net.xml \
    --taz $AREA/taz/${AREA}_districts.taz.xml --backend sumo --out $AREA/cost.csv

# 3. gravity OD from real land use (Overpass; local .osm only if building-rich)
python scripts/gravity_od.py --net $AREA.net.xml \
    --taz $AREA/taz/${AREA}_districts.taz.xml --cost $AREA/cost.csv \
    --cache $AREA/weights.npz --osm "" --out-dir $AREA/od --num 20

# 4. measure real congestion (Iran: Neshan; elsewhere: fetch_speeds.py)
python scripts/fetch_neshan.py --net $AREA.net.xml \
    --taz $AREA/taz/${AREA}_districts.taz.xml --poll 3600 --hours 24 \
    --out $AREA/observed.csv

# 5. calibrate total demand against the measurement
python scripts/calibrate_od.py --net $AREA.net.xml \
    --taz $AREA/taz/${AREA}_districts.taz.xml --cost $AREA/cost.csv \
    --observed $AREA/observed.csv --out-dir $AREA/od_calibrated

# 6. picture: zones + land use + flows on the road map
python scripts/plot_od_map.py --taz $AREA/taz/${AREA}_districts.taz.xml \
    --weights $AREA/weights.npz --od $AREA/od_calibrated/od_00.xml \
    --cost $AREA/cost.csv --net $AREA.net.xml --out $AREA/od_map.png
```

Checklist for a new area (the things that actually bite):

1. **bbox order** in the Overpass map URL is `lon,lat,lon,lat` — reversed vs
   most tools.
2. **admin_level differs per country** — always run `--list-levels` first
   (Tehran: 9 = منطقه, 11 = محله; Germany: 9/10; France: 8).
3. **Zones must be routable** — keep the largest-SCC default in
   build_taz_districts, or islands eat your trips silently.
4. **Check the weights** before trusting the OD: a zone showing `1 home` means
   OSM has no tagged buildings there (common outside Europe) — the gravity
   model is blind in that zone.
5. **Provider coverage**: Neshan = Iran. TomTom/HERE = most of the world,
   NOT Iran. Same CSV format either way; calibrate_od reads both.
