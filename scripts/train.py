#!/usr/bin/env python3
"""Train a model that estimates hourly traffic (vehicles entering an edge).

Inputs are only what is known at inference time: the edge identity (learned
embedding), the hour of day, and static edge attributes. Whole days are held
out, so evaluation measures prediction on *unseen days*.

Models:
  mlp    edge embedding + hour embedding + sin/cos + static features -> MLP
  gru    edge embedding initialises a GRU that decodes the 24-hour profile
  table  historical average per (edge, hour) -- the baseline

    python scripts/train.py --data-dir output/dataset --out-dir output/model
"""
import _path  # noqa: F401

from traffic_estimate.cli import add_out_dir, add_seed, make_parser
from traffic_estimate.ml import ForecastDataset, ForecastTrainer
# re-exported for demo.ipynb and any other in-notebook use
from traffic_estimate.ml.metrics import regression_metrics as metrics  # noqa: F401
from traffic_estimate.ml.models import EdgeProfileGRU as GRUNet  # noqa: F401
from traffic_estimate.ml.models import EdgeProfileMLP as MLP  # noqa: F401


def main():
    parser = make_parser(__doc__)
    parser.add_argument("--data-dir", required=True,
                        help="dir with traffic.csv.gz + edges.csv")
    add_out_dir(parser)
    parser.add_argument("--model", choices=["mlp", "gru", "table"], default="mlp")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=32)
    parser.add_argument("--test-scenarios", type=int, default=4)
    parser.add_argument("--val-scenarios", type=int, default=2)
    parser.add_argument("--abs-tol", type=float, default=3.0,
                        help="a prediction counts as accurate within this many "
                             "vehicles, or within --rel-tol, whichever is larger")
    parser.add_argument("--rel-tol", type=float, default=0.2)
    add_seed(parser, default=0)
    args = parser.parse_args()

    trainer = ForecastTrainer(
        ForecastDataset.load(args.data_dir), model=args.model,
        hidden=args.hidden, emb_dim=args.emb_dim, lr=args.lr,
        epochs=args.epochs, batch_size=args.batch_size,
        n_test=args.test_scenarios, n_val=args.val_scenarios,
        abs_tol=args.abs_tol, rel_tol=args.rel_tol, seed=args.seed)
    _, results = trainer.run()
    trainer.save(args.out_dir, results)


if __name__ == "__main__":
    main()
