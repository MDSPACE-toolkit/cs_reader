#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


def configure_openmm_plugins() -> None:
    executable_directory = Path(sys.executable).resolve().parent
    plugin_directory = executable_directory / "openmm_plugins"

    if plugin_directory.is_dir():
        os.environ["OPENMM_PLUGIN_DIR"] = str(plugin_directory)


configure_openmm_plugins()

import openmm
from openmm import Platform
from openmm.app import PDBFile
from pdbfixer import PDBFixer


PROGRAM_NAME = "pdb-fixer"


def log(
    message: str,
    level: str = "INFO",
) -> None:
    timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    print(
        f"[{timestamp}] [{PROGRAM_NAME}] [{level}] {message}",
        file=sys.stderr,
        flush=True,
    )


@contextmanager
def logged_step(name: str) -> Iterator[None]:
    log(f"START: {name}")
    start_time = time.perf_counter()

    try:
        yield

    except Exception:
        elapsed = time.perf_counter() - start_time

        log(
            f"FAILED: {name} after {elapsed:.3f} seconds",
            level="ERROR",
        )

        raise

    else:
        elapsed = time.perf_counter() - start_time

        log(
            f"DONE: {name} in {elapsed:.3f} seconds"
        )


def is_compiled_runtime() -> bool:
    return (
        "__compiled__" in globals()
        or getattr(sys, "frozen", False)
    )


def log_environment() -> None:
    log(f"Python executable: {sys.executable}")

    log(
        "Python version: "
        + sys.version.replace("\n", " ")
    )

    log(f"Operating system: {platform.platform()}")
    log(f"Machine architecture: {platform.machine()}")
    log(f"Current directory: {Path.cwd()}")
    log(f"OpenMM version: {openmm.__version__}")

    log(
        "Runtime mode: "
        + (
            "compiled executable"
            if is_compiled_runtime()
            else "Python interpreter"
        )
    )

    relevant_variables = (
        "OPENMM_PLUGIN_DIR",
        "OPENMM_CPU_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "LD_LIBRARY_PATH",
    )

    for variable in relevant_variables:
        value = os.environ.get(
            variable,
            "<unset>",
        )

        log(
            f"Environment {variable}={value}"
        )


def log_openmm_platforms() -> None:
    with logged_step("Discover OpenMM platforms"):
        platform_count = Platform.getNumPlatforms()

        log(
            f"OpenMM platform count: {platform_count}"
        )

        for index in range(platform_count):
            current_platform = Platform.getPlatform(
                index
            )

            log(
                f"OpenMM platform {index}: "
                f"{current_platform.getName()}"
            )

        plugin_failures = Platform.getPluginLoadFailures()

        if plugin_failures:
            for failure in plugin_failures:
                log(
                    f"OpenMM plugin load failure: {failure}",
                    level="WARNING",
                )
        else:
            log(
                "No OpenMM plugin load failures reported"
            )

        if platform_count == 1:
            platform_name = Platform.getPlatform(
                0
            ).getName()

            if platform_name == "Reference":
                log(
                    "Only the OpenMM Reference platform is available. "
                    "Operations that use OpenMM may be extremely slow.",
                    level="WARNING",
                )


def count_topology(
    fixer: PDBFixer,
) -> tuple[int, int, int]:
    chain_count = sum(
        1
        for _ in fixer.topology.chains()
    )

    residue_count = sum(
        1
        for _ in fixer.topology.residues()
    )

    atom_count = sum(
        1
        for _ in fixer.topology.atoms()
    )

    return (
        chain_count,
        residue_count,
        atom_count,
    )


def log_topology(
    fixer: PDBFixer,
    label: str,
) -> None:
    (
        chain_count,
        residue_count,
        atom_count,
    ) = count_topology(fixer)

    log(
        f"{label}: "
        f"{chain_count} chains, "
        f"{residue_count} residues, "
        f"{atom_count} atoms"
    )


def atom_name(atom: Any) -> str:
    name = getattr(
        atom,
        "name",
        None,
    )

    if name is not None:
        return str(name)

    return str(atom)


def residue_description(residue: Any) -> str:
    chain = getattr(
        residue,
        "chain",
        None,
    )

    chain_id = getattr(
        chain,
        "id",
        "<unknown>",
    )

    residue_name = getattr(
        residue,
        "name",
        "<unknown>",
    )

    residue_id = getattr(
        residue,
        "id",
        "<unknown>",
    )

    return (
        f"chain={chain_id}, "
        f"residue={residue_name}, "
        f"id={residue_id}"
    )


def log_missing_residues(
    fixer: PDBFixer,
) -> None:
    missing_residues = fixer.missingResidues

    log(
        "Missing residue groups found: "
        f"{len(missing_residues)}"
    )

    for key, residue_names in missing_residues.items():
        try:
            chain_index, residue_index = key
        except Exception:
            log(
                f"Missing residues: location={key!r}, "
                f"residues={','.join(map(str, residue_names))}",
                level="WARNING",
            )

            continue

        log(
            "Missing residues: "
            f"chain_index={chain_index}, "
            f"insertion_index={residue_index}, "
            f"residues={','.join(map(str, residue_names))}"
        )


def log_nonstandard_residues(
    fixer: PDBFixer,
) -> None:
    nonstandard_residues = fixer.nonstandardResidues

    log(
        "Nonstandard residues found: "
        f"{len(nonstandard_residues)}"
    )

    for entry in nonstandard_residues:
        try:
            residue, replacement = entry
        except Exception:
            log(
                f"Unexpected nonstandard residue entry: {entry!r}",
                level="WARNING",
            )

            continue

        log(
            "Nonstandard residue: "
            f"{residue_description(residue)}, "
            f"replacement={replacement}"
        )


def log_missing_atoms(
    fixer: PDBFixer,
) -> None:
    missing_atoms = fixer.missingAtoms
    missing_terminals = fixer.missingTerminals

    total_missing_atoms = sum(
        len(atoms)
        for atoms in missing_atoms.values()
    )

    total_missing_terminal_atoms = sum(
        len(atoms)
        for atoms in missing_terminals.values()
    )

    log(
        "Missing atoms found: "
        f"{total_missing_atoms} atoms across "
        f"{len(missing_atoms)} residues"
    )

    log(
        "Missing terminal atoms found: "
        f"{total_missing_terminal_atoms} atoms across "
        f"{len(missing_terminals)} residues"
    )

    for residue, atoms in missing_atoms.items():
        names = [
            atom_name(atom)
            for atom in atoms
        ]

        log(
            "Missing atoms: "
            f"{residue_description(residue)}, "
            f"atoms={','.join(names)}"
        )

    for residue, atoms in missing_terminals.items():
        names = [
            atom_name(atom)
            for atom in atoms
        ]

        log(
            "Missing terminal atoms: "
            f"{residue_description(residue)}, "
            f"atoms={','.join(names)}"
        )


def fix_pdb(
    input_pdb: Path,
    output_pdb: Path,
    ph: float,
    add_missing_residues: bool,
    add_missing_atoms: bool,
    add_hydrogens: bool,
) -> None:
    log_environment()
    log_openmm_platforms()

    resolved_input = input_pdb.resolve()
    resolved_output = output_pdb.resolve()

    log(f"Input file: {resolved_input}")
    log(f"Output file: {resolved_output}")
    log(f"Input size: {input_pdb.stat().st_size} bytes")
    log(f"Add missing residues: {add_missing_residues}")
    log(f"Add missing atoms: {add_missing_atoms}")
    log(f"Add hydrogens: {add_hydrogens}")
    log(f"Hydrogen pH: {ph}")

    with logged_step("Load input PDB"):
        fixer = PDBFixer(
            filename=str(input_pdb)
        )

    log_topology(
        fixer,
        "Initial topology",
    )

    with logged_step("Find missing residues"):
        fixer.findMissingResidues()

    log_missing_residues(fixer)

    if add_missing_residues:
        log(
            "Missing residue addition enabled"
        )
    else:
        log(
            "Missing residue addition disabled; "
            "clearing missing residue list"
        )

        fixer.missingResidues = {}

    with logged_step("Find nonstandard residues"):
        fixer.findNonstandardResidues()

    log_nonstandard_residues(fixer)

    with logged_step("Replace nonstandard residues"):
        fixer.replaceNonstandardResidues()

    log_topology(
        fixer,
        "Topology after replacing nonstandard residues",
    )

    with logged_step("Find missing atoms"):
        fixer.findMissingAtoms()

    log_missing_atoms(fixer)

    if add_missing_atoms or add_missing_residues:
        reasons: list[str] = []

        if add_missing_atoms:
            reasons.append(
                "missing atoms requested"
            )

        if add_missing_residues:
            reasons.append(
                "missing residues requested"
            )

        log(
            "Calling addMissingAtoms(): "
            + ", ".join(reasons)
        )

        with logged_step(
            "Add missing atoms and residues"
        ):
            fixer.addMissingAtoms()

        log_topology(
            fixer,
            "Topology after adding missing atoms and residues",
        )
    else:
        log(
            "Skipping addMissingAtoms(): "
            "neither missing atoms nor missing residues requested"
        )

    if add_hydrogens:
        log(
            f"Calling addMissingHydrogens() at pH {ph}"
        )

        with logged_step("Add missing hydrogens"):
            fixer.addMissingHydrogens(
                pH=ph
            )

        log_topology(
            fixer,
            "Topology after adding hydrogens",
        )
    else:
        log(
            "Skipping hydrogen addition"
        )

    with logged_step("Create output directory"):
        output_pdb.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    with logged_step("Write output PDB"):
        with output_pdb.open(
            "w",
            encoding="utf-8",
        ) as output:
            PDBFile.writeFile(
                fixer.topology,
                fixer.positions,
                output,
                keepIds=True,
            )

    if not output_pdb.is_file():
        raise RuntimeError(
            f"Output file was not created: {output_pdb}"
        )

    log(
        f"Output size: {output_pdb.stat().st_size} bytes"
    )

    log(
        "PDB repair completed successfully"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair a PDB structure using "
            "PDBFixer and OpenMM."
        )
    )

    parser.add_argument(
        "input_pdb",
        type=Path,
        help="Input PDB file",
    )

    parser.add_argument(
        "output_pdb",
        type=Path,
        help="Output repaired PDB file",
    )

    parser.add_argument(
        "--missing-residues",
        action="store_true",
        help="Add missing residues",
    )

    parser.add_argument(
        "--missing-atoms",
        action="store_true",
        help="Add missing atoms",
    )

    parser.add_argument(
        "--hydrogens",
        action="store_true",
        help="Add missing hydrogens",
    )

    parser.add_argument(
        "--ph",
        type=float,
        default=7.0,
        help=(
            "pH used when adding hydrogens "
            "(default: 7.0)"
        ),
    )

    return parser.parse_args()


def main() -> int:
    start_time = time.perf_counter()

    try:
        arguments = parse_arguments()

        log(
            "Command line: "
            + " ".join(
                repr(argument)
                for argument in sys.argv
            )
        )

        if not arguments.input_pdb.is_file():
            log(
                "Input file does not exist or is not a regular file: "
                f"{arguments.input_pdb}",
                level="ERROR",
            )

            return 1

        if arguments.ph < 0.0 or arguments.ph > 14.0:
            log(
                f"pH must be between 0 and 14: {arguments.ph}",
                level="ERROR",
            )

            return 1

        if (
            arguments.output_pdb.exists()
            and arguments.output_pdb.is_dir()
        ):
            log(
                "Output path is a directory: "
                f"{arguments.output_pdb}",
                level="ERROR",
            )

            return 1

        fix_pdb(
            input_pdb=arguments.input_pdb,
            output_pdb=arguments.output_pdb,
            ph=arguments.ph,
            add_missing_residues=arguments.missing_residues,
            add_missing_atoms=arguments.missing_atoms,
            add_hydrogens=arguments.hydrogens,
        )

    except KeyboardInterrupt:
        elapsed = time.perf_counter() - start_time

        log(
            f"Interrupted after {elapsed:.3f} seconds",
            level="ERROR",
        )

        return 130

    except Exception as error:
        elapsed = time.perf_counter() - start_time

        log(
            f"Fatal error after {elapsed:.3f} seconds: "
            f"{type(error).__name__}: {error}",
            level="ERROR",
        )

        for line in traceback.format_exc().rstrip().splitlines():
            log(
                line,
                level="TRACEBACK",
            )

        return 1

    elapsed = time.perf_counter() - start_time

    log(
        f"Finished successfully in {elapsed:.3f} seconds"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
