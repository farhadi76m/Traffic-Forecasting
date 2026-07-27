# new_test — 2-hour OD for a second Tehran map

A worked example of the OD method on a **new** OSM extract, with the demand
magnitude anchored on **live Neshan measurements** rather than guessed.

| | |
|---|---|
| source extract | `map(2).osm` (13.5 MB, OSM bbox 35.7590–35.8054 N, 51.3379–51.3970 E) |
| network | `new_test.net.xml`, 7.1 × 10.5 km, **6,342 routable car edges** |
| zones | 6 TAZ (3×3 grid, empty southern row dropped) |
| observation | 20 zone pairs, Neshan live, Mon 2026-07-27 09:33 local |
| measured congestion | **0.583** — traffic moving at 58% of free flow |
| output | `output/od_2h/od_*.xml` — tazRelation OD, 07:00–09:00 |

## Reproduce

```bash
conda activate traffic          # its SUMO is 1.27; the system netconvert 1.18
                                # crashes on this extract (junction-angle assert)

# 1. OSM -> car-only SUMO network
cd new_test && netconvert --osm-files "map(2).osm" -o new_test.net.xml \
    --geometry.remove --roundabouts.guess --ramps.guess \
    --junctions.join --tls.guess-signals --tls.discard-simple --tls.join \
    --output.street-names --proj.utm \
    --keep-edges.by-vclass passenger --remove-edges.isolated && cd ..

# 2. zones + zone-to-zone travel-time (impedance) matrix
python scripts/build_taz.py --net new_test/new_test.net.xml --grid 3 \
    --out new_test/output/taz/taz9.xml
python scripts/cost_matrix.py --net new_test/new_test.net.xml \
    --taz new_test/output/taz/taz9.xml --backend sumo \
    --out new_test/output/cost_matrix.csv

# 3. MEASURE real congestion (needs a Tehran IP; free key at platform.neshan.org)
export NESHAN_API_KEY=service....
python scripts/fetch_neshan.py --net new_test/new_test.net.xml \
    --taz new_test/output/taz/taz9.xml --once --pairs 20 \
    --out new_test/output/observed_neshan.csv

# 4. gravity OD from the map's own land use, magnitude bisected until SUMO
#    reproduces the measured congestion; emit only the 07:00-09:00 window
python scripts/calibrate_od.py --net new_test/new_test.net.xml \
    --taz new_test/output/taz/taz9.xml --cost new_test/output/cost_matrix.csv \
    --observed new_test/output/observed_neshan.csv \
    --peak-hours 9 --hours 7,8 --num 5 \
    --osm 'new_test/map(2).osm' --cache new_test/output/osm_weights.npz \
    --out-dir new_test/output/od_2h --work new_test/output/calib_work
```

Always pass `--osm`/`--cache`: their defaults point at the *first* map, and
land use read from the wrong extract lands outside every zone polygon, which
silently floors all weights to 1 and gives a uniform (meaningless) OD shape.

## Honest limits

- The demand was measured at **09:33**, so hour 9 is the anchored hour. The
  07:00–09:00 output is that calibrated daily total re-split by the model's
  hourly profile — the *level* is measured, the peak *shape* is assumed. To
  anchor the peak itself, run the fetcher across it:
  `fetch_neshan.py --poll 3600 --hours 24`.
- Congestion pins demand only loosely: a link at half free-flow can be
  moderately or heavily loaded (both branches of the fundamental diagram).
  This fixes the order of magnitude, not a vehicle count.
- Zone-internal (intrazonal) trips are part of the matrix but barely load the
  network, since od2trips picks origin and destination edges inside the
  same zone.
