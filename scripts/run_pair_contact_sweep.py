"""Fail closed until the physical two-Hydrogel FEM contact problem is supplied."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mechanistic_mv.mechanics.pair_interaction import PairForceScaling

try:
    from ._phase6_common import (
        PAIR_VALIDATION_DIRECTORY,
        PHASE6_SCHEMA_VERSION,
        provenance,
        write_json,
    )
except ImportError:  # direct ``python scripts/...`` execution
    from _phase6_common import (  # type: ignore[no-redef]
        PAIR_VALIDATION_DIRECTORY,
        PHASE6_SCHEMA_VERSION,
        provenance,
        write_json,
    )


MISSING_REQUIREMENTS = [
    {
        "key": "calibrated_hydrogel_parameters",
        "required": "Nnu, chi, phi0, delta_mu_over_kBT, kT_over_nu and source",
    },
    {
        "key": "particle_geometry",
        "required": "two undeformed radii and geometry calibration provenance",
    },
    {
        "key": "contact_fem_solver",
        "required": "implemented and versioned nonlinear FEM/contact solver",
    },
    {
        "key": "mesh_and_resolution",
        "required": "mesh method, resolution series and convergence criterion",
    },
    {
        "key": "mechanical_boundary_conditions",
        "required": "complete displacement/traction and rigid-mode constraints",
    },
    {
        "key": "solvent_bath_boundary_conditions",
        "required": "bath chemical/mass conditions from a physical source",
    },
    {
        "key": "contact_law",
        "required": "frictionless non-penetration discretization and tolerances",
    },
    {
        "key": "time_scale_assessment",
        "required": "tau_gel, tau_swarm and an explicit maximum ratio",
    },
    {
        "key": "distance_sweep",
        "required": "center distances, reference distance and force tolerance",
    },
    {
        "key": "reaction_force_convention",
        "required": "surface integration and positive-repulsion sign convention",
    },
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PAIR_VALIDATION_DIRECTORY,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_directory: Path = args.output_directory
    status_path = output_directory / "pair_contact_sweep_status.json"
    metadata_path = output_directory / "metadata.json"
    command = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    preexisting = [
        name
        for name in (
            "pair_force.csv",
            "force_curve.png",
            "potential_curve.png",
            "force_potential_consistency.png",
        )
        if (output_directory / name).exists()
    ]
    common = {
        "schema_version": PHASE6_SCHEMA_VERSION,
        "workflow_status": "BLOCKED",
        "physical_status": "PHYSICAL_CONTACT_DATA_NOT_AVAILABLE",
        "validation_status": "UNVERIFIED",
        "contact_solver_status": "NOT_AVAILABLE",
        "data_semantics": PairForceScaling.UNSCALED_SINGLE_PAIR.value,
        "missing_inputs": MISSING_REQUIREMENTS,
        "generated_data_files": [],
        "preexisting_untrusted_artifacts": preexisting,
        "prohibited_fabricated_outputs": [
            "pair_force.csv",
            "force_curve.png",
            "contact_solution_fields",
        ],
        "reason": (
            "two-Hydrogel contact FEM, geometry, mesh, boundary/bath conditions "
            "and calibrated parameters are incomplete; no force samples were run"
        ),
        **provenance(command),
    }
    write_json(
        metadata_path,
        {
            "schema_name": "mechanistic_mv.pair_contact_sweep_metadata",
            "artifact_type": "PAIR_CONTACT_SWEEP_METADATA",
            **common,
        },
    )
    write_json(
        status_path,
        {
            "schema_name": "mechanistic_mv.pair_contact_sweep_status",
            "artifact_type": "PAIR_CONTACT_SWEEP_STATUS",
            "metadata_file": metadata_path.name,
            **common,
        },
    )
    print(status_path)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
