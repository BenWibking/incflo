"""Export the two scalar Enzyme pullbacks from a configured upstream-Clang CPU build."""

import argparse
import json
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("build", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
database = json.loads((args.build / "compile_commands.json").read_text())
unit = next(e for e in database if e["file"].endswith("incflo_dss_faces.cpp"))
flags = shlex.split(unit["command"])
compiler = Path(flags[0])
flags = flags[: flags.index("-o")]
with tempfile.TemporaryDirectory() as temp:
    raw = Path(temp) / "faces.ll"
    subprocess.run(
        [
            *flags,
            "-O2",
            "-fno-vectorize",
            "-fno-slp-vectorize",
            "-fno-exceptions",
            "-fno-rtti",
            "-ffp-contract=off",
            "-S",
            "-emit-llvm",
            unit["file"],
            "-o",
            str(raw),
        ],
        cwd=unit["directory"],
        check=True,
    )
    ir = subprocess.check_output(
        [
            str(compiler.parent / "llvm-extract"),
            "-S",
            "--func=incflo_dss_flux_vjp",
            "--func=incflo_dss_normal_vjp",
            "--recursive",
            str(raw),
            "-o",
            "-",
        ],
        text=True,
    )
# These freestanding functions use only native double arithmetic, branches, and
# generic pointers. Remove host-only attributes and optional alias hints before
# the ROCm frontend assembles the IR for its own ABI and LLVM bitcode version.
lines = []
for line in ir.splitlines():
    if line.startswith((";", "source_filename", "target ", "!", "attributes ")):
        continue
    line = re.sub(r", !(tbaa|alias\.scope|noalias) !\d+", "", line)
    line = re.sub(r" #\d+", "", line).replace(" captures(none)", "")
    if line.startswith("define "):
        line = line.replace(") local_unnamed_addr {", ") alwaysinline {")
    if line.startswith("declare ") and "@llvm.fabs.f64(" not in line:
        raise ValueError(f"Nonportable external dependency in Enzyme output: {line}")
    lines.append(line)
args.output.write_text("\n".join(lines).rstrip() + "\n")
