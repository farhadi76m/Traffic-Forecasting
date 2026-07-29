#!/usr/bin/env python3
"""Train a neural surrogate of SUMO: OD matrix -> per-edge, per-hour traffic.

    OD (24, zones^2) -> [GRU over the 24 hours] -> context c_h
    (edge embedding, static features, c_h) -> MLP -> counts_h, traveltime_h

Unlike train.py's GRU, this one is driven by the demand, so it can react to it;
the hour-to-hour recurrence carries congestion built up earlier into later hours.
Two heads, each with the right likelihood: Poisson NLL for counts, MSE on
log(t / freeflow) for travel time.

THE number to read is `vs OD-blind baseline`. Any model scores a high R2 here
just by learning the average rush hour; only the gap over the baseline shows the
surrogate actually learned OD -> traffic.

    python scripts/train_surrogate.py --data output_sur/cong/dataset/data.npz \
        --out-dir output_sur/cong/model
"""
import _path  # noqa: F401

from traffic_estimate.cli import add_out_dir, add_seed, make_parser
from traffic_estimate.ml import SurrogateDataset, SurrogateTrainer
from traffic_estimate.ml.models import ODSurrogate  # noqa: F401  (re-export)


def main():
    parser = make_parser(__doc__)
    parser.add_argument("--data", required=True,
                        help="data.npz from build_surrogate_dataset.py")
    add_out_dir(parser)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--scen-batch", type=int, default=8,
                        help="scenarios per step")
    parser.add_argument("--edge-batch", type=int, default=768,
                        help="edges per step")
    parser.add_argument("--w-tt", type=float, default=1.0,
                        help="weight of the travel-time loss")
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--val-frac", type=float, default=0.15)
    add_seed(parser, default=0)
    args = parser.parse_args()

    trainer = SurrogateTrainer(
        SurrogateDataset.load(args.data), hidden=args.hidden,
        emb_dim=args.emb_dim, lr=args.lr, epochs=args.epochs,
        scen_batch=args.scen_batch, edge_batch=args.edge_batch,
        tt_weight=args.w_tt, test_frac=args.test_frac, val_frac=args.val_frac,
        seed=args.seed)
    trainer.fit()
    trainer.save(args.out_dir, trainer.evaluate())


if __name__ == "__main__":
    main()
