"""Use Optimas to propose seed families and evaluate their one-cycle recurrence on two grids."""

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
            )
            with (directory / "optimizer.log").open("w") as log, redirect_stdout(log):
                search(inner)
        run = coarse / "candidate"
        steps = args.coarse_steps
    else:
        run = directory / "fine"
        previous = json.loads((directory / "coarse-score.json").read_text())
        if control_hash(control) != previous["control_sha256"]:
            raise ValueError(f"Accepted control changed for trial {trial}")
        report = json.loads((coarse / "search.json").read_text())
        run.mkdir()
        replay(report, control, args.exe, INPUTS, run, args.fine_grid, args.fine_steps)
        steps = args.fine_steps
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
        row["promotion"] = proposal["promotion"]
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


def run(args):
    comm = MPI.COMM_WORLD
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
        promoted = select_fine(proposals, coarse, batch, args.fine_per_batch)
        fine = evaluate_batch(comm, args, promoted, "fine")
        if comm.rank == 0:
            observed = {r["trial"]: r for r in fine}
            for p in proposals:
                trial = p["_id"]
                if trial in observed:
                    generator.ingest(
                        [
                            {
                                **p,
                                "score": observed[trial]["score"],
                                "_status": TrialStatus.COMPLETED,
                            }
                        ]
                    )
                else:
                    generator.ax_client.abandon_trial(
                        trial, reason="Coarse screening; fine score unobserved"
                    )
            history.extend(
                {"batch": batch + 1, "coarse": r, "fine": observed.get(r["trial"])}
                for r in coarse
            )
            paired = [r for r in history if r["fine"] is not None]
            correlation = spearmanr(
                [r["coarse"]["score"] for r in paired],
                [r["fine"]["score"] for r in paired],
            ).statistic
            best = min(paired, key=lambda r: (r["fine"]["score"], r["fine"]["trial"]))[
                "fine"
            ]
            write_json(
                args.out / "summary.json",
                {
                    "best": best,
                    "coarse_count": len(history),
                    "fine_count": len(paired),
                    "paired_spearman": float(correlation)
                    if np.isfinite(correlation)
                    else None,
                    "trials": history,
                },
            )
            print(
                f"Batch {batch + 1} complete. Best fine score {best['score']:.6g}, trial {best['trial']}",
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
    parser.add_argument("--coarse-grid", type=int, default=MIN_SEARCH_GRID)
    parser.add_argument("--fine-grid", type=int, default=192)
    parser.add_argument("--coarse-steps", type=int, default=DEFAULT_SEARCH_STEPS)
    parser.add_argument("--fine-steps", type=int, default=144)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fine-per-batch", type=int, default=4)
    parser.add_argument("--inner-seed", type=int, default=19005)
    parser.add_argument("--design-seed", type=int, default=319)
    args = parser.parse_args()
    if args.coarse_grid < MIN_SEARCH_GRID:
        parser.error(
            f"Optimization requires at least {MIN_SEARCH_GRID} cells per dimension"
        )
    if not (
        2 <= args.fine_per_batch <= args.batch_size
        and args.fine_grid > args.coarse_grid
        and args.fine_steps >= args.coarse_steps >= 1
        and min(args.iterations, args.batches) >= 1
    ):
        parser.error(
            "Need 2 <= fine-per-batch <= batch-size, finer second grid, and positive step and iteration budgets"
        )
    args.exe, args.out = args.exe.resolve(), args.out.resolve()
    if args.seed is not None:
        args.seed = args.seed.resolve()
    run(args)
