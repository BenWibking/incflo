"""Use Optimas to search for one-cycle recurrence on matched grids."""

import argparse
import hashlib
import json
import os
import socket
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from dss_resolution import richardson_return
from dss_vorticity import INPUTS, measure, replay
from mpi4py import MPI
from optimize_dss import DEFAULT_SEARCH_STEPS, MIN_SEARCH_GRID, search, search_seed
from scipy.stats import spearmanr


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def control_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate(args, proposal, stage):
    rank = MPI.COMM_WORLD.rank
    trial = int(proposal["_id"])
    directory = args.out / f"trial-{trial:05}"
    coarse = directory / "coarse"
    control = coarse / "candidate/velocity.bin"
    started = time.time()
    if stage == "coarse":
        directory.mkdir()
        base = search_seed(args.seed, args.coarse_grid)
        # Separate components that are even/odd under a half turn about the
        # vertical axis, including rotation of the Cartesian vector itself.
        rotated = base[:, :, ::-1, ::-1].copy()
        rotated[:2] *= -1
        family = (
            proposal["even_gain"] * (base + rotated)
            + proposal["spiral_gain"] * (base - rotated)
        ) / 2
        with tempfile.TemporaryDirectory(dir=directory) as tmp:
            seed = Path(tmp) / "family.npy"
            np.save(seed, family)
            del base, rotated, family
            inner = argparse.Namespace(
                seed=seed,
                random_seed=args.inner_seed,
                n=args.coarse_grid,
                steps=args.coarse_steps,
                exe=args.exe,
                inputs=INPUTS,
                out=coarse,
                nr=9,
                nz=7,
                harmonics=[19],
                jitter=0.12,
                iterations=args.iterations,
                plots=True,
                richardson_grids=(args.middle_grid, args.fine_grid)
                if args.mode == "richardson"
                else None,
                richardson_steps=(args.middle_steps, args.fine_steps),
                richardson_order=args.order,
                check_direction=args.check_direction and trial == 0,
            )
            with (directory / "optimizer.log").open("w") as log, redirect_stdout(log):
                search(inner)
        run = coarse / "candidate"
        if args.mode == "richardson":
            for name, n in (("middle", args.middle_grid), ("fine", args.fine_grid)):
                (run / f"n{n}").rename(directory / name)
        steps = args.coarse_steps
    else:
        run = directory / stage
        previous = json.loads((directory / "coarse-score.json").read_text())
        if control_hash(control) != previous["control_sha256"]:
            raise ValueError(f"Accepted control changed for trial {trial}")
        steps = args.middle_steps if stage == "middle" else args.fine_steps
        if args.mode == "screen":
            report = json.loads((coarse / "search.json").read_text())
            run.mkdir()
            replay(report, control, args.exe, INPUTS, run, args.fine_grid, steps)
    if (
        os.environ.get("DSS_EXPECT_HIP") == "1"
        and "HIP initialized with 1 device" not in (run / "incflo.log").read_text()
    ):
        raise RuntimeError(f"Expected a single visible HIP device for trial {trial}")
    row = {
        **measure(run, steps),
        "trial": trial,
        "stage": stage,
        "method": proposal["method"],
        "even_gain": proposal["even_gain"],
        "spiral_gain": proposal["spiral_gain"],
        "control_sha256": control_hash(control),
        "host": socket.gethostname(),
        "rank": rank,
        "device": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "start": started,
        "end": time.time(),
    }
    if stage == "fine":
        row["promotion"] = proposal.get("promotion", "all three grids")
    write_json(directory / f"{stage}-score.json", row)
    return row


def evaluate_batch(comm, args, proposals, stage):
    local = [evaluate(args, p, stage) for p in proposals[comm.rank :: comm.size]]
    return [row for group in comm.allgather(local) for row in group]


def select_fine(proposals, coarse, batch, count):
    by_id = {p["_id"]: p for p in proposals}
    ranked = [
        by_id[r["trial"]]
        for r in sorted(coarse, key=lambda r: (r["score"], r["trial"]))
    ]
    if batch == 0 or count == len(ranked):
        return [
            {**p, "promotion": "paired calibration" if batch == 0 else "all"}
            for p in ranked
        ]
    # Keep a coarse outlier in each batch to detect reversals under refinement.
    return [{**p, "promotion": "coarse rank"} for p in ranked[: count - 1]] + [
        {**ranked[-1], "promotion": "coarse worst audit"}
    ]


def score_triplet(directory, coarse, middle, fine, order):
    levels = [coarse, middle, fine]
    if len({r["control_sha256"] for r in levels}) != 1:
        raise ValueError("Three-grid scores have different control hashes")
    grids = [r["grid"] for r in levels]
    metrics = {
        key: richardson_return([r[key] for r in levels], grids, order)[0]
        for key in ("velocity_return", "vorticity_return", "m19_return")
    }
    report = json.loads((directory / "coarse/search.json").read_text())
    if not np.isclose(
        metrics["velocity_return"]["score"],
        report["final_guided_return"],
        rtol=2e-7,
        atol=2e-9,
    ):
        raise RuntimeError(
            "Projected plotfiles disagree with the three-grid optimization objective"
        )
    result = {
        "trial": fine["trial"],
        "score": max(v["score"] for v in metrics.values()),
        "fine_score": fine["score"],
        "control_sha256": fine["control_sha256"],
        "grids": grids,
        "steps": [r["steps"] for r in levels],
        "metrics": metrics,
    }
    write_json(directory / "richardson-score.json", result)
    return result


def run(args):
    comm = MPI.COMM_WORLD
    use_richardson = args.mode == "richardson"
    ranking = "richardson" if use_richardson else "fine"
    if comm.rank == 0:
        import torch
        from gest_api.vocs import VOCS
        from optimas.core import TrialStatus
        from optimas.generators import AxSingleFidelityGenerator

        args.out.mkdir(parents=True, exist_ok=False)
        np.random.seed(args.design_seed)
        torch.manual_seed(args.design_seed)
        torch.set_num_threads(1)
        generator = AxSingleFidelityGenerator(
            vocs=VOCS(
                variables={"even_gain": [0.85, 1.15], "spiral_gain": [0.75, 1.35]},
                objectives={"score": "MINIMIZE"},
            ),
            n_init=args.batch_size,
            save_model=False,
        )
        history = []
        write_json(
            args.out / "config.json",
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            | {"workers": comm.size},
        )

    for batch in range(args.batches):
        proposals = None
        if comm.rank == 0:
            proposals = generator.suggest(args.batch_size)
            for p in proposals:
                p["method"] = generator.ax_client.get_trial(
                    p["_id"]
                ).generation_method_str
            print(
                f"Batch {batch + 1} proposals: {[p['method'] for p in proposals]}",
                flush=True,
            )
        proposals = comm.bcast(proposals, root=0)
        coarse = evaluate_batch(comm, args, proposals, "coarse")
        middle = {}
        if use_richardson:
            middle = {
                r["trial"]: r for r in evaluate_batch(comm, args, proposals, "middle")
            }
            promoted = proposals
        else:
            promoted = select_fine(proposals, coarse, batch, args.fine_per_batch)
        fine = evaluate_batch(comm, args, promoted, "fine")
        if comm.rank == 0:
            observed = {r["trial"]: r for r in fine}
            scores = {}
            for row in coarse:
                trial = row["trial"]
                record = {
                    "batch": batch + 1,
                    "coarse": row,
                    "fine": observed.get(trial),
                }
                if use_richardson:
                    record["middle"] = middle[trial]
                    record["richardson"] = score_triplet(
                        args.out / f"trial-{trial:05}",
                        row,
                        middle[trial],
                        observed[trial],
                        args.order,
                    )
                history.append(record)
                if record[ranking] is not None:
                    scores[trial] = record[ranking]["score"]
            for proposal in proposals:
                trial = proposal["_id"]
                if trial in scores:
                    generator.ingest(
                        [
                            {
                                **proposal,
                                "score": scores[trial],
                                "_status": TrialStatus.COMPLETED,
                            }
                        ]
                    )
                else:
                    generator.ax_client.abandon_trial(
                        trial, reason="Coarse screening; fine score unobserved"
                    )
            paired = [r for r in history if r["fine"] is not None]
            correlation = spearmanr(
                [r["coarse"]["score"] for r in paired],
                [r["fine"]["score"] for r in paired],
            ).statistic
            best = min(
                (r[ranking] for r in paired), key=lambda r: (r["score"], r["trial"])
            )
            write_json(
                args.out / "summary.json",
                {
                    "best": best,
                    "ranking": ranking,
                    "middle_count": len(paired) if use_richardson else 0,
                    "coarse_count": len(history),
                    "fine_count": len(paired),
                    "paired_spearman": float(correlation)
                    if np.isfinite(correlation)
                    else None,
                    "trials": history,
                },
            )
            print(
                f"Batch {batch + 1} complete. Best {ranking} score {best['score']:.6g}, trial {best['trial']}",
                flush=True,
            )

    if comm.rank == 0:
        winner = args.out / f"trial-{best['trial']:05}" / "fine"
        measure(
            winner,
            args.fine_steps,
            save_profile=True,
            title=f"Trial {best['trial']} · {args.fine_grid}³",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument(
        "--seed",
        type=Path,
        help="Override the modeled seed generated on the optimization grid",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--mode", choices=("richardson", "screen"), default="richardson"
    )
    parser.add_argument("--coarse-grid", type=int, default=MIN_SEARCH_GRID)
    parser.add_argument("--middle-grid", type=int, default=160)
    parser.add_argument("--fine-grid", type=int, default=192)
    parser.add_argument("--coarse-steps", type=int, default=DEFAULT_SEARCH_STEPS)
    parser.add_argument("--middle-steps", type=int, default=120)
    parser.add_argument("--fine-steps", type=int, default=144)
    parser.add_argument("--order", type=float, default=2.0)
    parser.add_argument(
        "--check-direction",
        action="store_true",
        help="Check trial zero's combined three-grid gradient",
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--fine-per-batch", type=int, help="Number of fine replays in screened mode"
    )
    parser.add_argument("--inner-seed", type=int, default=19005)
    parser.add_argument("--design-seed", type=int, default=319)
    args = parser.parse_args()
    if args.coarse_grid < MIN_SEARCH_GRID:
        parser.error(
            f"Optimization requires at least {MIN_SEARCH_GRID} cells per dimension"
        )
    if args.mode == "richardson":
        if args.fine_per_batch is not None or not (
            args.coarse_grid < args.middle_grid < args.fine_grid
            and args.coarse_steps * args.middle_grid
            == args.middle_steps * args.coarse_grid
            and args.coarse_steps * args.fine_grid == args.fine_steps * args.coarse_grid
            and np.isfinite(args.order)
            and args.order > 0
        ):
            parser.error(
                "Richardson mode evaluates every trial at three increasing grids with constant steps/grid and positive order"
            )
    elif args.check_direction:
        parser.error("--check-direction requires --mode richardson")
    if args.fine_per_batch is None:
        args.fine_per_batch = (
            args.batch_size if args.mode == "richardson" else min(4, args.batch_size)
        )
    if not (
        (1 if args.mode == "richardson" else 2)
        <= args.fine_per_batch
        <= args.batch_size
        and args.fine_grid > args.coarse_grid
        and args.fine_steps >= args.coarse_steps >= 1
        and min(args.iterations, args.batches) >= 1
    ):
        parser.error(
            "Need positive batch, step and iteration budgets, a finer final grid, "
            "and at least two fine promotions per batch in screened mode"
        )
    args.exe, args.out = args.exe.resolve(), args.out.resolve()
    if args.seed is not None:
        args.seed = args.seed.resolve()
    run(args)
