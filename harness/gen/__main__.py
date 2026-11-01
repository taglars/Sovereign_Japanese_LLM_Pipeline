"""python -m gen <command>. Experiment 3 from the command line."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gen import customers as C  # noqa: E402
from gen import runner as R  # noqa: E402
from gen import templates as T  # noqa: E402

DEFAULT_DB = ROOT / "out" / "exp3" / "generation.db"
DEFAULT_OUT = ROOT / "out" / "exp3"


def _config() -> dict:
    import yaml
    return yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())["exp3"]


def _customers(args) -> list:
    cfg = _config()
    people = C.build(int(args.n or cfg["customers"]), seed=int(cfg["seed"]))
    if args.limit:
        # Smoke runs take one of each family first, so every graph is exercised.
        by_family = {}
        for c in people:
            by_family.setdefault(c.family, []).append(c)
        order = [c for trio in zip(*by_family.values()) for c in trio]
        people = order[:args.limit]
    return people


def cmd_customers(args) -> int:
    people = _customers(args)
    out = Path(args.out or DEFAULT_OUT / "customers.json")
    C.save(people, out)
    print(f"wrote {len(people)} customer(s) to {out}")
    return 0


def cmd_run(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = _config()
    sampling = R.Sampling.from_config(cfg.get("sampling", {}))
    people = _customers(args)
    conditions = tuple(args.conditions.split(",")) if args.conditions else R.CONDITIONS
    table = T.load_table() if "ja" in conditions else None
    gen = R.Generator(args.base_url, args.model or cfg["model"], sampling,
                      timeout=float(cfg.get("request_timeout_seconds", 900)))
    print(f"model      {gen.client.model}")
    print(f"customers  {len(people)}  conditions {','.join(conditions)}")
    print(f"sampling   {sampling}")
    print(f"store      {args.db}")
    gen.client.wait_until_ready(timeout=args.ready_timeout, on_line=print)
    store = R.GenStore(args.db)
    return R.run_all(gen, store, people, table=table, conditions=conditions,
                     workers=int(args.workers or cfg.get("workers", 96)))


def cmd_pending(args) -> int:
    """Prints one number: chains still to generate. Used by chained jobs."""
    table = T.load_table()
    print(R.pending(R.GenStore(args.db), _customers(args), table=table))
    return 0


def cmd_status(args) -> int:
    store = R.GenStore(args.db)
    ctx = store.rows("SELECT kind, COUNT(*), SUM(ok) FROM contexts GROUP BY kind")
    for kind, n, ok in ctx:
        print(f"context {kind:10} {ok or 0}/{n}")
    print(f"{'cond':4} {'task':9} {'n':>5} {'ok':>5} {'in':>11} {'out':>10} {'s/call':>7}")
    for cond, task, n, ok, tin, tout, lat in store.summary():
        print(f"{cond:4} {task:9} {n:5d} {ok or 0:5d} {tin or 0:11,} {tout or 0:10,} "
              f"{lat or 0:7.1f}")
    bad = store.rows("SELECT error, COUNT(*) FROM calls WHERE ok=0 GROUP BY error "
                     "ORDER BY 2 DESC LIMIT 8")
    for error, n in bad:
        print(f"  x{n:<4d} {error[:100]}")
    return 0


def cmd_export(args) -> int:
    store = R.GenStore(args.db)
    result = R.export(store, _customers(args), args.out or DEFAULT_OUT)
    print(f"exported {result['scripts']} script(s); "
          f"{result['incomplete_customers']} customer(s) incomplete, left out")
    return 0


def cmd_metrics(args) -> int:
    from gen import metrics as M
    store = R.GenStore(args.db)
    path = M.write(store, _customers(args), Path(args.out or DEFAULT_OUT) / "metrics.csv")
    print(f"wrote {path}")
    print(M.summary(path))
    return 0


def cmd_review_sheet(args) -> int:
    from gen import review
    print(f"wrote {review.write_sheet()}")
    return 0


def cmd_apply_review(args) -> int:
    from gen import review
    result = review.apply(Path(args.file))
    print(f"applied: {result['changed']} of {result['lines']} line(s) corrected")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gen", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--n", type=int, help="customers (default: config exp3.customers)")
        p.add_argument("--limit", type=int, default=0,
                       help="only the first N, one family at a time (smoke test)")
        p.add_argument("--db", default=str(DEFAULT_DB))
        p.add_argument("--out")

    p = sub.add_parser("customers"); common(p); p.set_defaults(fn=cmd_customers)
    p = sub.add_parser("run"); common(p)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", help="served model name (default: config exp3.model)")
    p.add_argument("--conditions", help="en,ja (default both)")
    p.add_argument("--workers", type=int)
    p.add_argument("--ready-timeout", type=float, default=1800)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("status"); common(p); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("pending"); common(p); p.set_defaults(fn=cmd_pending)
    p = sub.add_parser("export"); common(p); p.set_defaults(fn=cmd_export)
    p = sub.add_parser("metrics"); common(p); p.set_defaults(fn=cmd_metrics)
    p = sub.add_parser("review-sheet"); p.set_defaults(fn=cmd_review_sheet)
    p = sub.add_parser("apply-review"); p.add_argument("file")
    p.set_defaults(fn=cmd_apply_review)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
