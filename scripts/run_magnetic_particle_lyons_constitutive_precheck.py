"""Read-only Lyons-2020 lineage arithmetic; no PDE, fitting or parameter admission.

The published front slope is not silently treated as mean particle drift.
Source-specific nominal-force arithmetic is reported conditionally, with known
velocity SD separated from missing magnetic/thermal uncertainty. No phi is fit.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import subprocess
from typing import Sequence

import numpy as np

from mechanistic_mv.mechanics.magnetic_particle_potential import LinearMagneticParticle


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data/external/magnetic_particle"
DEFAULT_SOURCE = (
    SOURCE_DIR
    / "lyons2020_peg1000_8p9nm_agarose0p3__lineage_provenance__r1.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/validation/magnetic_particle"
ARTIFACT_STEM = "lyons2020_peg1000_8p9nm_agarose0p3__constitutive_precheck__r1"
FRONT_DIGITIZATION_FILENAME = (
    "lyons2020_peg1000_8p9nm_agarose0p3__front_digitization__r1.csv"
)
HISTORICAL_GENERIC_GATE_PATH = (
    "outputs/validation/magnetic_particle/generic__physical_gate__r1.json"
)
LINEAGE_ID = "LYONS_BROUGHAM_2020_PEG1000_CORE8P9_AGAROSE0P3"
INSUFFICIENT = "LYONS_CONSTITUTIVE_DATA_INSUFFICIENT"
PROJECT_STATUS = "CURRENT_2D_MV_PHYSICAL_REDUCTION_INVALID"
K_B = 1.380649e-23  # Exact SI Boltzmann constant, J/K.
UNITS = {
    "core_diameter": "nm", "hydrodynamic_diameter": "nm",
    "magnetic_volume": "m^3", "chi_manuscript": "1", "chi_thesis": "1",
    "susceptibility_reference_B": "T", "nominal_magnet_B": "T",
    "cited_gradient": "T/m", "water_viscosity": "Pa s",
    "characterization_temperature": "K", "transport_temperature": "K",
    "front_velocity": "mm/h", "agarose_concentration": "% w/v",
    "gel_depth": "mm", "injected_volume": "uL", "particle_concentration": "mg/mL",
    "published_vth_manuscript": "mm/h", "published_vth_thesis": "mm/h",
}


def finite_number(value: object, name: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite numeric value")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{name} is not finite and in its domain")
    return number


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _no_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON token: {value}")


def load_source(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"),
                         object_pairs_hook=_no_duplicates, parse_constant=_no_constant)
    if not isinstance(payload, dict) or payload.get("schema") != "mechanistic_mv.lyons_lineage_provenance" or payload.get("version") != 2:
        raise ValueError("expected Lyons lineage provenance v2")
    if payload.get("lineage_id") != LINEAGE_ID:
        raise ValueError("wrong lineage; 2026 PEG2000 is forbidden")
    lineage = payload["lineage"]
    for key, expected in {"coating": "PEG1000", "nominal_core_diameter_nm": 8.9,
                          "hydrodynamic_diameter_nm": 24.0, "agarose_percent_wv": 0.3,
                          "cross_lineage_parameter_transfer_allowed": False}.items():
        if lineage.get(key) != expected:
            raise ValueError(f"lineage {key} must remain {expected!r}")
    sources = payload["sources"]
    if not isinstance(sources, list) or len(sources) != 5:
        raise ValueError("expected five explicitly versioned sources")
    ids = {s["id"] for s in sources}
    if ids != {"RSC2020", "MANUSCRIPT2020", "ESI2020", "THESIS2020", "CORRECTION2021"}:
        raise ValueError("source identities mismatch")
    for source in sources:
        for key in ("title", "url", "status", "locator"):
            if not isinstance(source.get(key), str) or not source[key].strip():
                raise ValueError(f"source {source['id']} lacks {key}")
        doi = source.get("doi")
        expected_doi = None if source["id"] == "THESIS2020" else (
            "10.1039/D0NR90262D" if source["id"] == "CORRECTION2021" else "10.1039/D0NR01602K")
        if doi != expected_doi:
            raise ValueError("source DOI cannot be substituted across versions or lineages")
    quantities = payload["quantities"]
    if set(quantities) != set(UNITS):
        raise ValueError("incomplete or unexpected quantity set")
    for name, unit in UNITS.items():
        record = quantities[name]
        if record.get("unit") != unit or record.get("source") not in ids:
            raise ValueError(f"{name} unit/source mismatch")
        for key in ("locator", "role"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f"{name} lacks {key}")
        if name == "transport_temperature":
            if record["value"] is not None:
                raise ValueError("do not promote characterization temperature to transport")
        else:
            finite_number(record["value"], name)
        uncertainty = record["uncertainty"]
        if uncertainty is not None:
            finite_number(uncertainty["value"], name + " uncertainty")
            if uncertainty["unit"] != unit or not uncertainty.get("kind"):
                raise ValueError(f"{name} uncertainty unit/kind required")
    # Retain the actual source disagreements, not an averaged/fitted replacement.
    for name, expected in {"chi_manuscript": 0.289, "chi_thesis": 0.281,
                           "susceptibility_reference_B": 0.23, "nominal_magnet_B": 0.55,
                           "core_diameter": 8.9, "hydrodynamic_diameter": 24.0,
                           "agarose_concentration": 0.3}.items():
        if quantities[name]["value"] != expected:
            raise ValueError(f"source conflict/lineage {name} was overwritten")
    for name in ("chi_manuscript", "chi_thesis", "magnetic_volume", "nominal_magnet_B", "cited_gradient"):
        if quantities[name]["uncertainty"] is not None:
            raise ValueError("new force uncertainties need independent source review, not insertion here")
    correction = payload["correction"]
    if correction["observable"] != "v_exp * d_hyd" or correction["unit"] != "mm^2/h" or correction["numeric_values_changed"] is not False:
        raise ValueError("2021 correction must use product v*d, not quotient")
    front = payload["front_observation"]
    if front["kind"] != "OPTICAL_MIGRATION_FRONT" or front["field_to_drift_mapping_validated"] is not False:
        raise ValueError("optical front must not be relabelled as particle drift")
    admission = payload["admission"]
    required_flags = {"force_spatial_calibration_available", "complete_force_uncertainty_available",
                      "multiple_independent_force_levels_available", "transport_temperature_confirmed",
                      "front_to_mean_drift_confirmed", "passive_diffusivity_measured",
                      "phi_fitting_allowed", "physical_parameter_lock_allowed"}
    if set(admission) != required_flags or any(value is not False for value in admission.values()):
        raise ValueError("this audited v2 source set is incomplete; promotion needs new evidence review")
    if payload["data_availability_audit"]["raw_csv_recovered"] is not False or payload["data_availability_audit"]["author_request_sent"] is not False:
        raise ValueError("no raw CSV was recovered and no author request was sent")
    return payload


def digitized_front_rows(source: dict) -> list[dict]:
    """Reconstruct the plotted data from measured pixel centres, never replicates."""
    digit = source["figure_digitization"]
    if digit["status"] != "DIGITIZED_AUTHOR_MANUSCRIPT_NOT_RAW_DATA" or digit["raw_replicate_reconstruction_allowed"] is not False or digit["mean_drift_promotion_allowed"] is not False:
        raise ValueError("digitization cannot become raw data or mean drift")
    cal = digit["calibration"]
    xp, tp = np.asarray(cal["x_pixel"], float), np.asarray(cal["time_h"], float)
    yp, dp = np.asarray(cal["y_pixel"], float), np.asarray(cal["distance_mm"], float)
    xy = np.asarray(digit["marker_centres_xy_px"], dtype=float)
    if any(a.shape != (2,) or not np.isfinite(a).all() for a in (xp, tp, yp, dp)) or xy.shape != (16, 2) or not np.isfinite(xy).all():
        raise ValueError("digitization needs 16 finite points and finite 2-point axis calibration")
    if xp[1] <= xp[0] or tp[1] <= tp[0] or yp[0] == yp[1] or dp[0] == dp[1] or np.any(np.diff(xy[:, 0]) <= 0):
        raise ValueError("invalid calibration or unordered marker centres")
    sx, sy = (tp[1] - tp[0]) / (xp[1] - xp[0]), (dp[1] - dp[0]) / (yp[1] - yp[0])
    eps = finite_number(digit["pixel_localization_bound_px"], "pixel bound")
    rows = []
    for i, (x, y) in enumerate(xy):
        rows.append({"point_id": i + 1, "time_h": float(tp[0] + (x - xp[0])*sx),
                     "distance_to_base_mm": float(dp[0] + (y - yp[0])*sy),
                     "pixel_x": float(x), "pixel_y": float(y),
                     "time_localization_bound_h": abs(sx)*eps,
                     "distance_localization_bound_mm": abs(sy)*eps,
                     "provenance": "DIGITIZED_AUTHOR_MANUSCRIPT_NOT_RAW_DATA"})
    if rows[0]["time_h"] < 0.4 or rows[-1]["time_h"] > 8.1:
        raise ValueError("digitization falls outside the declared 0.5-8 h field-on series")
    return rows


def _conditional_arithmetic(source: dict) -> list[dict]:
    q = source["quantities"]
    value = lambda key: float(q[key]["value"])
    v = value("front_velocity") * 1e-3 / 3600
    v_sd = float(q["front_velocity"]["uncertainty"]["value"]) * 1e-3 / 3600
    M_water = 1 / (3 * math.pi * value("water_viscosity") * value("hydrodynamic_diameter") * 1e-9)
    cases = []
    for name in ("chi_manuscript", "chi_thesis"):
        law = LinearMagneticParticle(value(name), value("magnetic_volume"),
                                     q[name]["locator"], "SOURCE_RECIPE_CONDITIONAL_NOT_CALIBRATED")
        force = float(law.force_newton_from_flux_density_gradient(
            np.array([value("nominal_magnet_B")]), np.array([[value("cited_gradient"), 0.0]]))[0, 0])
        M_front = v/force
        ratio = M_front/M_water
        cases.append({
            "case": name, "status": "CONDITIONAL_REPORTED_RECIPE_NOT_ADMITTED_FIELD_OR_MOBILITY",
            "chi": value(name), "B_nominal_T": value("nominal_magnet_B"),
            "gradient_cited_T_per_m": value("cited_gradient"), "force_N": force,
            "force_complete_standard_uncertainty_N": None,
            "M_front_apparent_m_per_N_s": M_front,
            "M_front_velocity_SD_only_m_per_N_s": v_sd/force,
            "M_eff_admitted_m_per_N_s": None,
            "D_Einstein_apparent_over_T_m2_per_s_K": M_front*K_B,
            "D_Einstein_over_T_velocity_SD_only": v_sd/force*K_B,
            "D_Einstein_admitted_m2_per_s": None,
            "water_mobility_m_per_N_s": M_water, "D0_over_T_m2_per_s_K": M_water*K_B,
            "D0_at_transport_temperature_m2_per_s": None,
            "apparent_D_Einstein_over_D0_if_same_T": ratio,
            "ratio_velocity_SD_only": v_sd/force/M_water,
            "conditional_ratio_le_1p25": ratio <= 1.25,
            "interpretation": "Nominal recipe plus front=drift plus Einstein yields a super-water apparent ratio. At least one prerequisite needs checking; this is not measured passive D or an unconditional rejection of constant M.",
            "uncertainty_scope": "Only reported velocity SD propagated with all other values held fixed; not total uncertainty or a confidence interval.",
            "no_phi_water_velocity_mm_per_h": force*M_water*3.6e6,
            "source_reported_vth_mm_per_h": value("published_vth_manuscript" if name == "chi_manuscript" else "published_vth_thesis"),
        })
    return cases


def build_report(source: dict) -> dict:
    rows = digitized_front_rows(source)
    t = np.array([r["time_h"] for r in rows])
    d = np.array([r["distance_to_base_mm"] for r in rows])
    centered = t-t.mean()
    slope = float(centered @ (d-d.mean()) / (centered @ centered))
    intercept = float(d.mean()-slope*t.mean())
    residual = d - (intercept+slope*t)
    sse = float(residual @ residual)
    sst = float((d-d.mean()) @ (d-d.mean()))
    q = source["quantities"]
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    return {
        "schema": "mechanistic_mv.lyons_constitutive_precheck", "version": 2,
        "lineage_id": LINEAGE_ID, "status": INSUFFICIENT,
        "project_status": PROJECT_STATUS,
        "project_status_basis": {
            "role": "PRESERVED_PRIOR_CONCLUSION_NOT_A_NEW_PDE_TEST",
            "path": HISTORICAL_GENERIC_GATE_PATH,
            "reason": "Gel height is not a source-backed out-of-plane slab closure. Vertical distribution, contact-cell closure and admitted D/M remain absent; this lineage audit resolves none of them.",
        },
        "input_status": "VALID_SOURCE_REGISTER_WITH_INCOMPLETE_PHYSICAL_DATA",
        "lineage": source["lineage"], "quantities": q, "sources": source["sources"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "repo_head": head},
        "source_sha256": {"value": source.get("source_sha256"), "role": "OPTIONAL_NOT_A_GATE"},
        "source_disagreements": source["unresolved_source_conflicts"],
        "correction": source["correction"],
        "conditional_recipes": _conditional_arithmetic(source),
        "formulae": {
            "force": "F=chi_v*V_m*B*gradB/mu0 (Physics public API)",
            "apparent_mobility": "M_front=v_front/F; not admitted M_eff without front-to-drift closure",
            "einstein": "D_Einstein/T=k_B*M_front; transport T is not supplied",
            "free_water": "D0/T=k_B/(3*pi*eta_water*d_hyd)",
            "ratio": "D_Einstein/D0=M_front*3*pi*eta_water*d_hyd if same T and Einstein assumption",
            "uncertainty": "Full u(F)/F requires chi,V,B,gradB uncertainties and covariance. Only v SD is known; u_partial(M)=SD(v)/F. Core size spread is not silently used as standard uncertainty of mean magnetic volume.",
        },
        "digitized_front": {
            "source": "MANUSCRIPT2020 Fig.1 (not publisher/raw replicate CSV)",
            "rows": rows, "point_count": len(rows),
            "front_speed_mm_per_h": -slope, "distance_intercept_mm": intercept,
            "R2": 1-sse/sst, "residual_RMSE_mm": math.sqrt(sse/len(rows)),
            "quoted_front_speed_mm_per_h": q["front_velocity"]["value"],
            "absolute_slope_difference_mm_per_h": abs(-slope-float(q["front_velocity"]["value"])),
            "slope_y_localization_bound_at_fixed_x_mm_per_h": float(sum(abs(centered))/(centered@centered)*rows[0]["distance_localization_bound_mm"]),
            "bound_limit": "Fixed x only; per-point x bound and unknown axis systematics are separate. No full slope uncertainty/replicate confidence interval is inferred.",
            "raw_data_recovered": False, "constitutive_admission": False,
        },
        "constant_mobility_test": {
            "CV_M_across_force_levels": None,
            "status": "NOT_IDENTIFIABLE_NO_MATCHED_INDEPENDENT_FORCE_LEVELS",
            "why": "Repeated gels, coating differences and agarose concentrations are not a controlled same-system magnetic-force sweep. Alternative source recipes are not independent force experiments.",
            "published_velocity_SD_over_mean": float(q["front_velocity"]["uncertainty"]["value"])/float(q["front_velocity"]["value"]),
            "velocity_ratio_role": "repeatability_only_not_mobility_CV_gate",
        },
        "front_to_drift": {
            "decision": "UNVERIFIED_DO_NOT_LOCK_M_OR_D", "controls": source["front_observation"]["supporting_not_decisive_controls"],
            "reason": "An optical threshold/front may shift through diffusion, broadening, settling, aggregation or intensity changes. Linear front position and field-off arrest support magnetic response, not equality to population-mean drift.",
            "required": source["front_observation"]["mapping_required"],
        },
        "missing_inputs": [
            "Reconcile 0.281 vs 0.289 using same-batch magnetometry, units, temperature and uncertainty.",
            "Map B(x) and gradB(x), including covariance/position uncertainty; distinguish 0.23 T reference from 0.55 T nominal.",
            "Obtain raw per-gel front times/positions and front threshold or calibrated concentration/centroid profiles.",
            "Obtain multiple independent magnetic-force conditions for the same PEG1000 batch and 0.3% agarose.",
            "Confirm transport temperature, passive diffusion and Einstein-relation applicability.",
        ],
        "author_request": {"sent": False, "draft": author_request()},
        "generated_data_files": [FRONT_DIGITIZATION_FILENAME],
        "scope": {"PDE_run": False, "Gym_run": False, "RL_DQN_run": False,
                  "full_regression_run": False, "phi_fitted": False, "author_request_sent": False,
                  "2026_PEG2000_imported": False, "physical_parameters_locked": False},
    }


def author_request() -> str:
    return """## Author data request draft - NOT SENT

Subject: PEG1000 8.9 nm-core agarose magnetophoresis: source data and force/front calibration

Dear Dr Lyons, Prof Brougham and Prof Morrin,

We are auditing the constitutive interpretation of the PEG1000, 8.9 nm-core,
0.3% w/v low-EEO agarose/DI-water system in Nanoscale 2020,
DOI 10.1039/D0NR01602K (and correction 10.1039/D0NR90262D), together with
chapters 2-3 of the Lyons thesis. We are not pooling the 2026 PEG2000 batches.

Could you share, or point us to an existing public deposit containing:

1. Figure 1 and ESI S8-S10/S12-S13 per-gel time/front-position CSV files,
   replicate IDs, initial-point exclusions and slope-fit uncertainties; the
   original images/profiles and ImageJ front threshold/scale procedure.
2. The relevant synthesis/measurement batch map, TEM diameter distribution,
   DLS uncertainty and magnetometry M(H)/M(B) data with unit and normalization
   conventions. The thesis quotes chi=0.281, whereas manuscript/ESI gives
   0.289. ESI S2 refers to B=0.23 T at 6 mm, but the manuscript/thesis equation
   also quotes the magnet's nominal 0.55 T and gradB=45 T/m. Which values and
   magnet geometry apply to each measured trajectory, and with what errors?
3. B(x) and gradB(x) or an original field map/model and uncertainty over the
   front's full path; results at independently varied force levels for the
   same particle batch/gel, if available. Corner-repeatability data alone
   cannot establish the CV of mobility across force strengths.
4. Transport-run temperature, medium viscosity, no-field front-width or
   independently measured diffusivity, and centroid/mean-drift information
   that could test whether optical-front velocity equals mean particle drift.
5. Clarification of the reported no-phi theoretical velocity and its exact
   force/viscosity inputs; we will not fit a tortuosity phi to close a mismatch.

We can use anonymized replicate IDs and will preserve the distinction between
published summaries, digitized points and raw measurements, with your preferred
data citation and reuse terms. This request is a draft only and has not been sent.
"""


def render_markdown(source: dict, report: dict) -> str:
    d = report["digitized_front"]
    lines = [
        "# Magnetic-particle Lyons 2020 constitutive precheck — r1",
        "",
        f"Status: **{INSUFFICIENT}**",
        "",
             f"Project status preserved: **{PROJECT_STATUS}** (not recomputed by this source audit).", "",
             f"Lineage: `{LINEAGE_ID}`. No 2026 PEG2000 data, no PDE/Gym/RL, no phi fit.", "",
             "## Source-specific conditional arithmetic (not admitted physical parameters)", "",
             "| Recipe | F nominal [N] | M_front apparent [m/(N s)] | velocity-SD-only | apparent D/D0 |",
             "|---|---:|---:|---:|---:|"]
    for case in report["conditional_recipes"]:
        lines.append(f"| {case['case']} | {case['force_N']:.8e} | {case['M_front_apparent_m_per_N_s']:.8e} | {case['M_front_velocity_SD_only_m_per_N_s']:.8e} | {case['apparent_D_Einstein_over_D0_if_same_T']:.6f} |")
    lines += ["", "The force uncertainty is unknown. The displayed SD propagates only the reported",
              "front-velocity SD and is not full uncertainty or a confidence interval. D/T is",
              "reported in JSON, but D and D0 at transport temperature remain null. The apparent",
              "ratios exceed 1.25 only under unverified front=drift and Einstein assumptions.",
              "They expose a consistency problem to investigate, not a measured passive diffusivity.", "",
              "## Source and observation checks", "",
              "- chi=0.281 (thesis) versus 0.289 (manuscript/ESI) remain separate.",
              "- 0.23 T susceptibility reference is not the 0.55 T nominal magnet field.",
              "- The 2021 correction requires v*d, in mm^2/h, not v/d.",
              "- Single-force repeated gels cannot determine mobility CV across fields.",
              f"- Figure 1: {d['point_count']} digitized manuscript markers give front speed {d['front_speed_mm_per_h']:.8f} mm/h, R2={d['R2']:.8f}, residual RMSE={d['residual_RMSE_mm']:.8f} mm.",
              "- These are rendered-figure coordinates, not raw replicate CSV. Localization bounds",
              "  and axis calibration are stored; no replicate uncertainty is reconstructed.", "",
              "- The 6 mm gel height does not establish a depth-integrated 2D closure.", "",
              "## Missing inputs", ""]
    lines += ["- "+item for item in report["missing_inputs"]]
    lines += ["", "## Inspected sources", ""]
    lines += [f"- [{s['id']}]({s['url']}): {s['locator']}; {s['status']}." for s in source["sources"]]
    lines += ["", author_request()]
    return "\n".join(lines)+"\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        source = load_source(args.source)
        report = build_report(source)
        # Serialize before writing so nonfinite derived values cannot leave a
        # nominal success artifact. Hash absence does not enter this check.
        encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n"
    except (ValueError, KeyError, TypeError, OSError) as error:
        report = {"schema": "mechanistic_mv.lyons_constitutive_precheck", "version": 2,
                  "status": INSUFFICIENT, "input_status": "INVALID_SOURCE", "error": str(error),
                  "project_status": PROJECT_STATUS, "generated_data_files": [],
                  "scope": {"PDE_run": False, "Gym_run": False, "RL_DQN_run": False,
                            "full_regression_run": False, "phi_fitted": False, "author_request_sent": False}}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"{ARTIFACT_STEM}.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        (args.output_dir / f"{ARTIFACT_STEM}.md").write_text(f"# Magnetic-particle Lyons 2020 constitutive precheck — r1\n\n{INSUFFICIENT}\n\nProject status preserved: {PROJECT_STATUS}\n\nInvalid source: {error}\n\nNo data files generated by this run.\n", encoding="utf-8")
        print(f"{INSUFFICIENT}: {error}")
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{ARTIFACT_STEM}.json").write_text(encoded, encoding="utf-8")
    (args.output_dir / f"{ARTIFACT_STEM}.md").write_text(render_markdown(source, report), encoding="utf-8")
    with (args.output_dir / FRONT_DIGITIZATION_FILENAME).open("w", encoding="utf-8", newline="") as stream:
        rows = report["digitized_front"]["rows"]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{INSUFFICIENT}: 16 digitized points; no physical parameter admission; no simulation")
    return 2  # Scientific insufficiency is an explicit non-success exit.


if __name__ == "__main__":
    raise SystemExit(main())
