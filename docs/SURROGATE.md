# A neural surrogate of SUMO, and the OD inverse problem

**Goal.** Replace SUMO with a network that maps an OD matrix to per-edge, per-hour
traffic, then run it *backwards*: given observed travel times, recover the OD.

**Short answer to "is it logical?"** The forward surrogate is logical and it
works. The inverse is logical *only in a congested network*. In free-flowing
traffic, travel time carries essentially no information about the OD, and we can
now say that with numbers rather than intuition.

---

## 1. What was already there, and why it could not be reused

`train.py` trains an MLP/GRU whose inputs are `(edge, hour)`. **The OD is not an
input.** The model therefore cannot answer "what happens if demand changes" — it
learned one average day and reproduces it. That is the right model for the
forecasting question the repo was built for, and it is why the `table` baseline
(a per-edge/hour historical average) nearly matches it. But it is not a
surrogate, and no amount of retraining makes it one.

Worse, the training data could not have taught it. `generate_od.py` builds every
"day" from the same gravity matrix and adds Poisson jitter:

| | 60 scenarios from `generate_od.py` | 300 scenarios from `sample_od.py` |
|---|---|---|
| OD-shape correlation between any two scenarios | **> 0.997** | **0.68** |
| spread of total daily trips | ±7.6% | ±34% |

Every scenario is effectively **the same OD**. A surrogate trained on that data
learns the mean and nothing about the map from demand to traffic. So the first
new script is `sample_od.py`, which draws each scenario's total demand, per-zone
production/attraction, cell-level flows, 24h shape and commute directionality
from a wide log-normal prior. That prior is also, by construction, the prior of
the inverse problem.

## 2. The regime problem: travel time only moves when the road is full

BPR-style link cost is `t = t0 * (1 + 0.15 * (v/c)^4)`. Below roughly `v/c = 0.6`
that fourth power is flat: travel time barely responds to demand. Measured on the
actual network (`tehran_2026_area.net.xml`, 3,229 edges):

| daily trips | travel time / free-flow (peak) | edges > 1.2x free-flow | network speed | SUMO runtime |
|---|---|---|---|---|
| 12,000 (the repo's setting) | 1.01x | 1.6% | 99.1% of free flow | 5 s |
| 50,000 | 1.04x | 2.7% | 97.7% | 18 s |
| 90,000 | 1.10x | 4.1% | 96.4% | 46 s |
| **110,000** | **5.7x** | **15.3%** | **88.2%** | 247 s |
| 150,000 | 564x (gridlock) | 70.1% | 33.7% | 488 s |

The transition from free-flow to gridlock is abrupt. The experiment therefore
uses two regimes: **light** (~12k trips/day, what the repo already simulated) and
**congested** (~100k/day, clipped below the gridlock cliff).

This is not a quirk of our network. It is the reason every travel-time-based OD
paper in the literature is a congested-network paper — one is literally titled
"...under Congested Network" ([Networks & Spatial Economics
2020](https://link.springer.com/article/10.1007/s11067-020-09496-4)).

## 3. The surrogate

Your GRU, with the one change that matters: **the GRU is driven by the OD**, not
by an edge embedding alone.

```
OD (24, n_zones^2)  ->  GRU over the 24 hours  ->  demand context c_h  (24, H)
(edge embedding, static edge features, c_h)  ->  MLP  ->  counts_h, travel_time_h
```

The hour-to-hour recurrence is doing real work: congestion built up at 08:00
spills into 09:00, and a per-hour feedforward map cannot express that. Two heads
with the right likelihood for each: Poisson NLL on counts, MSE on
`log(travel time / free-flow time)`.

**Every result is reported against an OD-blind baseline** that predicts the
training mean for each `(edge, hour)` and ignores the OD entirely — i.e. exactly
what the old model is. Any model scores a high R2 here just by learning the
average rush hour. **Only the gap over that baseline is evidence the model reads
the OD at all.**

## 4. Results

### Forward: does the surrogate learn OD -> traffic?

Held-out OD matrices the surrogate never saw:

| regime | target | surrogate R2 | OD-blind R2 | gap |
|---|---|---|---|---|
| light | vehicle counts | **0.972** | 0.747 | **+0.225** |
| light | travel time (s) | 0.9966 | 0.9966 | **+0.0000** |
| light | travel time (log ratio) | 0.384 | 0.712 | **-0.328** |

Read the travel-time rows carefully. The R2 of 0.9966 looks superb and means
nothing: the OD-blind baseline scores exactly the same. That 0.9966 is "long
edges take longer" — it is geometry, not traffic. On the part that actually
depends on demand (the ratio to free-flow) the surrogate is **worse than
predicting the mean**, because there is no signal to fit and it fits noise.

Counts, in the same regime and from the same model, are learned well (+0.225 over
blind). **The information is in the counts, not the travel times.**

### Inverse: travel times -> OD

MAP + Laplace through the frozen surrogate, 300 busiest edges as sensors, 864
unknowns (36 OD pairs x 24 h), 5 held-out ODs. Observation noise is calibrated to
the surrogate's own held-out error — see the warning below.

| regime | observed | OD R2 recovered | prior mean alone | OD directions constrained |
|---|---|---|---|---|
| light | travel time | **-2.5** | 0.288 | 102 / 864 |
| light | counts | **0.686** | 0.288 | 220 / 864 |

In the light regime, inverting travel times is **worse than not looking at the
data**. Counts recover the OD properly.

### A trap worth naming: the likelihood must not out-claim the surrogate

Our first run hard-coded a sensor precision of `sigma = 0.05` on log travel time.
Result: **OD R2 = -169**, with total daily trips estimated at 181,000 against a
true 24,500. The inversion was not fitting traffic; it was fitting the
surrogate's own error, which is far larger than 0.05. Setting `sigma` from the
surrogate's measured held-out residuals (0.109) removes the blow-up.

If you tell the sampler the sensors are more precise than the emulator is
accurate, the posterior will confidently measure your emulator instead of your
city. `invert_od.py` now calibrates `sigma` automatically by default.

A related caveat on the identifiability counts: the spectrum of `J'J` is
information about *the surrogate's* input-output map. It is only information
about SUMO when the surrogate is faithful. In the light regime the travel-time
head is **not** faithful (it loses to the blind baseline), so its "102
constrained directions" are largely spurious — the model invented a sensitivity
that SUMO does not have. Always read the spectrum next to the fidelity table.

## 5. How to run it

```bash
# 1. OD matrices from a wide prior (not 60 copies of one matrix)
python scripts/sample_od.py --taz output/taz/taz_grid9.xml \
    --out-dir output_sur/cong/od --num 160 --daily-trips 100000 \
    --sigma-total 0.20 --max-trips 125000

# 2. simulate them (parallel; drops route files and gzips edgedata)
python scripts/run_sim.py --net sumo/tehran_2026_area.net.xml \
    --taz output/taz/taz_grid9.xml --od-dir output_sur/cong/od \
    --out-dir output_sur/cong/sim --jobs 6

# 3. pair OD with counts AND travel times
python scripts/build_surrogate_dataset.py --net sumo/tehran_2026_area.net.xml \
    --od-dir output_sur/cong/od --sim-dir output_sur/cong/sim \
    --out-dir output_sur/cong/dataset

# 4. train the surrogate (reports the OD-blind baseline alongside)
python scripts/train_surrogate.py --data output_sur/cong/dataset/data.npz \
    --out-dir output_sur/cong/model

# 5. invert: observations -> posterior over OD
python scripts/invert_od.py --model output_sur/cong/model/surrogate.pt \
    --data output_sur/cong/dataset/data.npz --observe traveltime
python scripts/invert_od.py --model output_sur/cong/model/surrogate.pt \
    --data output_sur/cong/dataset/data.npz --observe counts

# 6. figures
python scripts/visualize_surrogate.py --root output_sur --out-dir docs/figures
```

## 6. Where this sits in the literature

The forward half is a **surrogate / metamodel for dynamic traffic assignment**;
the inverse half is **dynamic OD estimation (DODE)** to transport researchers and
**simulation-based inference / amortized inference** to ML researchers. The two
literatures barely cite each other.

- [ASNPE, Griesemer, Cao, Cui, Osorio & Liu, NeurIPS 2024](https://arxiv.org/abs/2412.05590)
  ([code](https://github.com/samgriesemer/seqinf)) is the closest work: SUMO,
  5,329 OD pairs, neural posterior estimation with normalizing flows. But it
  observes **counts**, and calls SUMO in the loop rather than learning a surrogate.
- [HSTGSN, Liu & Meidani 2024](https://arxiv.org/html/2408.04131) learns
  OD -> link flow as a surrogate, on DTALite, with a heterogeneous graph and
  "virtual OD edges" — the trick to steal if the zone count grows.
- [Osorio's metamodel line](https://web.mit.edu/osorioc/www/papers/osoODCalib.pdf)
  is the state of the art in DODE, and its "metamodel" is an *analytical physics*
  model, not a neural net — the paper's thesis is that the physics is what buys
  the sample efficiency. Worth knowing before assuming a neural surrogate
  inherits those results.
- [Osorio et al. 2025](https://arxiv.org/abs/2507.00306) argues exactly the
  deployment premise behind this work: travel times are abundant worldwide
  (Google Maps), loop-detector counts are scarce. The motivation for travel times
  is **data availability, not information content**.
- [BO4Mob](https://arxiv.org/html/2510.18824v1) is a sobering SUMO OD-estimation
  benchmark: on its 10,100-OD-pair problem, every method tested failed to beat
  random search.

Nobody appears to have published the exact combination here — a neural surrogate
of mesoscopic SUMO that emits **travel times**, inverted for OD. That is an
opening, but it also means the failure modes above are unmapped.

## 7. What to do next

1. **Use counts as the primary observable** wherever detectors exist; treat
   travel times as a supplement that only sharpens the peak hours.
2. **Reparameterize the OD through a gravity model** (production, attraction,
   distance decay) to cut 864 unknowns to a few dozen. Standard, and it directly
   attacks the identifiability problem rather than working around it.
3. **Validate against `routeSampler.py`**, SUMO's own count-based demand
   generator. If it cannot match the counts, the problem is the network/TAZ, not
   the inference.
4. If travel-time-only inversion is truly required, restrict claims to congested
   peak hours and always publish the spectrum showing which directions were
   resolved.
