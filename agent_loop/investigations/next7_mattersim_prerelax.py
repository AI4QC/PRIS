"""Fixed-step, fixed-cell FIRE pre-relaxation with injectable prediction.

The generic predictor contract is deliberately small: a callable receives a
``list[ase.Atoms]`` and returns :class:`BatchPrediction` containing aligned total
energies in eV, forces in eV/A, and stresses in eV/A^3.  MatterSim itself is
optional and imported only when :func:`make_mattersim_predictor` is called.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from operator import index
from time import perf_counter
from types import MappingProxyType
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.geometry import find_mic
from ase.optimize import FIRE
from ase.units import GPa


SNAPSHOT_STEPS = (0, 2, 4, 8)
_CELL_ABSOLUTE_TOLERANCE_A = 1.0e-12
_ROUNDING_ERROR_MULTIPLIER = 8.0

FIRE_PARAMETERS = MappingProxyType(
    {
        "dt": 0.05,
        "dtmax": 0.20,
        "maxstep": 0.05,
        "Nmin": 5,
        "finc": 1.1,
        "fdec": 0.5,
        "astart": 0.1,
        "fa": 0.99,
    }
)


@dataclass(frozen=True, slots=True)
class BatchPrediction:
    """Aligned predictor output in eV, eV/A, and eV/A^3, respectively."""

    total_energies_ev: Sequence[Any]
    forces_ev_per_a: Sequence[Any]
    stresses_ev_per_a3: Sequence[Any]


BatchPredictor = Callable[[list[Atoms]], BatchPrediction]


@dataclass(frozen=True, slots=True)
class StructurePrerelaxResult:
    """One input's isolated trajectory outcome in original input order."""

    snapshots: dict[int, Atoms]
    error: str | None
    force_evaluations: int
    optimizer_updates: int
    retry_overhead_force_evaluations: int = 0
    retry_overhead_optimizer_updates: int = 0


@dataclass(frozen=True, slots=True)
class PrerelaxRunResult:
    """All per-structure outcomes plus actual predictor-call accounting."""

    results: tuple[StructurePrerelaxResult, ...]
    predictor_forward_calls: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class SnapshotSummary:
    """Validated scalar summary of one saved pre-relaxation snapshot."""

    step: int
    total_energy_ev: float
    energy_per_atom_ev: float
    energy_change_from_previous_snapshot_ev_per_atom: float
    fmax_ev_per_a: float
    frms_ev_per_a: float
    stress_frobenius_ev_per_a3: float
    stress_max_abs_eigenvalue_ev_per_a3: float
    rms_displacement_from_x0_a: float
    max_displacement_from_x0_a: float
    min_pair_distance_a: float


@dataclass(frozen=True, slots=True)
class SupportDecision:
    """Whether a snapshot trajectory is safe to use for screening.

    ``supported=False`` is deliberately fail-open: downstream code must keep,
    rather than reject, the structure when this protocol is unsupported.
    """

    supported: bool
    reason: str


class SnapshotValidationError(ValueError):
    """Raised when saved ASE snapshots cannot be summarized safely."""

    def __init__(self, message: str, *, reason: str = "invalid_snapshots") -> None:
        super().__init__(message)
        self.reason = reason


class _PredictionValidationError(ValueError):
    """Raised internally when a predictor violates the batch contract."""


class _TrajectoryFailure(RuntimeError):
    def __init__(
        self,
        *,
        force_evaluations: int,
        optimizer_updates: int,
    ) -> None:
        super().__init__("fixed-step trajectory failed")
        self.force_evaluations = force_evaluations
        self.optimizer_updates = optimizer_updates


def _validated_cutoff_step(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"cutoff_step must be one of {SNAPSHOT_STEPS}")
    try:
        cutoff_step = index(value)
    except TypeError as exc:
        raise ValueError(
            f"cutoff_step must be one of {SNAPSHOT_STEPS}"
        ) from exc
    if cutoff_step not in SNAPSHOT_STEPS:
        raise ValueError(f"cutoff_step must be one of {SNAPSHOT_STEPS}")
    return cutoff_step


def _required_snapshots(
    snapshots: Mapping[int, Any], *, steps: Sequence[int]
) -> list[Any]:
    if not isinstance(snapshots, Mapping):
        raise SnapshotValidationError("snapshots must be a mapping")
    missing = [step for step in steps if step not in snapshots]
    if missing:
        joined = ", ".join(str(step) for step in missing)
        raise SnapshotValidationError(
            f"missing required snapshots: {joined}", reason="missing_snapshots"
        )
    return [snapshots[step] for step in steps]


def _finite_scalar(value: Any, *, name: str, step: int) -> float:
    array = np.asarray(value)
    if array.shape != ():
        raise SnapshotValidationError(f"step {step} {name} must be a scalar")
    try:
        result = float(array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SnapshotValidationError(
            f"step {step} {name} must be numeric"
        ) from exc
    if not np.isfinite(result):
        raise SnapshotValidationError(f"step {step} {name} must be finite")
    return result


def _stress_matrix(value: Any, *, step: int) -> np.ndarray:
    try:
        stress = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SnapshotValidationError(
            f"step {step} stress must be numeric"
        ) from exc
    if stress.shape == (6,):
        xx, yy, zz, yz, xz, xy = stress
        matrix = np.array(
            [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float
        )
    elif stress.shape == (3, 3):
        matrix = 0.5 * (stress + stress.T)
    else:
        raise SnapshotValidationError(
            f"step {step} stress has shape {stress.shape}; expected (6,) or (3, 3)"
        )
    if not np.all(np.isfinite(matrix)):
        raise SnapshotValidationError(f"step {step} stress must be finite")
    return matrix


def _validated_snapshot(atoms: Any, *, step: int) -> tuple[float, np.ndarray, np.ndarray]:
    try:
        atom_count = len(atoms)
    except (TypeError, AttributeError) as exc:
        raise SnapshotValidationError(f"step {step} is not an ASE-like Atoms object") from exc
    if atom_count <= 0:
        raise SnapshotValidationError(f"step {step} contains no atoms")

    info = getattr(atoms, "info", None)
    arrays = getattr(atoms, "arrays", None)
    if not isinstance(info, Mapping) or not isinstance(arrays, Mapping):
        raise SnapshotValidationError(f"step {step} lacks ASE info/arrays mappings")
    if "total_energy" not in info:
        raise SnapshotValidationError(f"step {step} total_energy is missing")
    if "forces" not in arrays:
        raise SnapshotValidationError(f"step {step} forces are missing")
    if "stress" not in info:
        raise SnapshotValidationError(f"step {step} stress is missing")

    energy = _finite_scalar(info["total_energy"], name="total_energy", step=step)
    try:
        forces = np.asarray(arrays["forces"], dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SnapshotValidationError(f"step {step} forces must be numeric") from exc
    expected_shape = (atom_count, 3)
    if forces.shape != expected_shape:
        raise SnapshotValidationError(
            f"step {step} forces have shape {forces.shape}; expected {expected_shape}"
        )
    if not np.all(np.isfinite(forces)):
        raise SnapshotValidationError(f"step {step} forces must be finite")
    stress = _stress_matrix(info["stress"], step=step)

    try:
        positions = np.asarray(atoms.positions, dtype=float)
        cell = np.asarray(atoms.cell.array, dtype=float)
    except (TypeError, ValueError, AttributeError) as exc:
        raise SnapshotValidationError(
            f"step {step} positions/cell are invalid"
        ) from exc
    if positions.shape != expected_shape:
        raise SnapshotValidationError(
            f"step {step} positions have shape {positions.shape}; expected {expected_shape}"
        )
    if cell.shape != (3, 3):
        raise SnapshotValidationError(
            f"step {step} cell has shape {cell.shape}; expected (3, 3)"
        )
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(cell)):
        raise SnapshotValidationError(f"step {step} positions/cell must be finite")
    return energy, forces, stress


def _minimum_pair_distance(atoms: Any, *, step: int) -> float:
    if len(atoms) < 2:
        return float("inf")
    try:
        distances = np.asarray(atoms.get_all_distances(mic=True), dtype=float)
    except Exception as exc:
        raise SnapshotValidationError(
            f"step {step} periodic pair distances could not be computed"
        ) from exc
    pairs = distances[np.triu_indices(len(atoms), k=1)]
    if pairs.size == 0:
        return float("inf")
    return float(np.min(pairs))


def summarize_snapshots(
    snapshots: Mapping[int, Any], *, cutoff_step: int = 8
) -> dict[int, SnapshotSummary]:
    """Validate and summarize the saved snapshot prefix through ``cutoff_step``.

    Displacements are atom-wise minimum-image distances relative to step zero.
    Stress accepts either ASE Voigt order ``(xx, yy, zz, yz, xz, xy)`` or a
    3-by-3 tensor; the latter is symmetrized before norms are evaluated.
    """

    cutoff_step = _validated_cutoff_step(cutoff_step)
    cutoff_index = SNAPSHOT_STEPS.index(cutoff_step)
    snapshot_steps = SNAPSHOT_STEPS[: cutoff_index + 1]
    ordered = _required_snapshots(snapshots, steps=snapshot_steps)
    validated = [
        _validated_snapshot(atoms, step=step)
        for step, atoms in zip(snapshot_steps, ordered, strict=True)
    ]

    reference = ordered[0]
    reference_count = len(reference)
    reference_numbers = np.asarray(reference.get_atomic_numbers())
    reference_cell = np.asarray(reference.cell.array, dtype=float)
    reference_pbc = np.asarray(reference.pbc, dtype=bool)
    summaries: dict[int, SnapshotSummary] = {}

    for step, atoms, (energy, forces, stress) in zip(
        snapshot_steps, ordered, validated, strict=True
    ):
        if len(atoms) != reference_count or not np.array_equal(
            np.asarray(atoms.get_atomic_numbers()), reference_numbers
        ):
            raise SnapshotValidationError(
                f"step {step} atom count/order differs from step 0"
            )
        if not np.allclose(
            np.asarray(atoms.cell.array, dtype=float),
            reference_cell,
            rtol=0.0,
            atol=_CELL_ABSOLUTE_TOLERANCE_A,
        ):
            raise SnapshotValidationError(f"step {step} cell differs from step 0")
        if not np.array_equal(np.asarray(atoms.pbc, dtype=bool), reference_pbc):
            raise SnapshotValidationError(f"step {step} PBC differs from step 0")
        try:
            _, displacement_lengths = find_mic(
                np.asarray(atoms.positions) - np.asarray(reference.positions),
                reference.cell,
                pbc=reference.pbc,
            )
        except Exception as exc:
            raise SnapshotValidationError(
                f"step {step} minimum-image displacement could not be computed"
            ) from exc
        displacement_lengths = np.asarray(displacement_lengths, dtype=float)
        if displacement_lengths.shape != (reference_count,) or not np.all(
            np.isfinite(displacement_lengths)
        ):
            raise SnapshotValidationError(
                f"step {step} minimum-image displacement is invalid"
            )

        try:
            with np.errstate(over="ignore", invalid="ignore"):
                energy_per_atom = float(energy / reference_count)
                energy_change = (
                    0.0
                    if not summaries
                    else energy_per_atom
                    - next(reversed(summaries.values())).energy_per_atom_ev
                )
                force_norms = np.linalg.norm(forces, axis=1)
                fmax = float(np.max(force_norms))
                frms = float(np.sqrt(np.mean(force_norms**2)))
                stress_frobenius = float(np.linalg.norm(stress, ord="fro"))
                stress_max_abs_eigenvalue = float(
                    np.max(np.abs(np.linalg.eigvalsh(stress)))
                )
                rms_displacement = float(
                    np.sqrt(np.mean(displacement_lengths**2))
                )
                max_displacement = float(np.max(displacement_lengths))
        except (ArithmeticError, np.linalg.LinAlgError, ValueError) as exc:
            raise SnapshotValidationError(
                f"step {step} derived metrics could not be computed"
            ) from exc
        derived_metrics = (
            energy_per_atom,
            energy_change,
            fmax,
            frms,
            stress_frobenius,
            stress_max_abs_eigenvalue,
            rms_displacement,
            max_displacement,
        )
        if not np.all(np.isfinite(derived_metrics)):
            raise SnapshotValidationError(
                f"step {step} derived metrics must be finite"
            )
        summaries[step] = SnapshotSummary(
            step=step,
            total_energy_ev=energy,
            energy_per_atom_ev=energy_per_atom,
            energy_change_from_previous_snapshot_ev_per_atom=energy_change,
            fmax_ev_per_a=fmax,
            frms_ev_per_a=frms,
            stress_frobenius_ev_per_a3=stress_frobenius,
            stress_max_abs_eigenvalue_ev_per_a3=stress_max_abs_eigenvalue,
            rms_displacement_from_x0_a=rms_displacement,
            max_displacement_from_x0_a=max_displacement,
            min_pair_distance_a=_minimum_pair_distance(atoms, step=step),
        )
    return summaries


def _positive_int(value: Any, *, name: str) -> int:
    try:
        result = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def pack_structures(
    structures: Sequence[Any],
    *,
    atom_budget: int,
    structure_cap: int | None = None,
) -> list[list[Any]]:
    """Greedily pack structures in input order under deterministic limits."""

    budget = _positive_int(atom_budget, name="atom_budget")
    cap = (
        _positive_int(structure_cap, name="structure_cap")
        if structure_cap is not None
        else None
    )
    batches: list[list[Any]] = []
    current: list[Any] = []
    current_atoms = 0

    for structure in structures:
        try:
            atom_count = len(structure)
        except (TypeError, AttributeError) as exc:
            raise ValueError("each structure must define its atom count") from exc
        if atom_count > budget:
            raise ValueError(
                f"structure has {atom_count} atoms, exceeding atom budget {budget}"
            )
        if atom_count <= 0:
            raise ValueError("structures must contain at least one atom")

        over_atoms = current_atoms + atom_count > budget
        over_count = cap is not None and len(current) >= cap
        if current and (over_atoms or over_count):
            batches.append(current)
            current = []
            current_atoms = 0
        current.append(structure)
        current_atoms += atom_count

    if current:
        batches.append(current)
    return batches


def _sanitize_structure(structure: Atoms) -> Atoms:
    """Rebuild an Atoms object without calculators, labels, or extra arrays."""

    if not isinstance(structure, Atoms):
        raise TypeError("structures must contain ase.Atoms objects")
    return Atoms(
        numbers=np.asarray(structure.get_atomic_numbers(), dtype=int).copy(),
        positions=np.asarray(structure.get_positions(), dtype=float).copy(),
        cell=np.asarray(structure.cell.array, dtype=float).copy(),
        pbc=np.asarray(structure.pbc, dtype=bool).copy(),
    )


def _aligned_length(values: Any, *, name: str, expected: int) -> None:
    try:
        actual = len(values)
    except (TypeError, AttributeError) as exc:
        raise _PredictionValidationError(
            f"{name} must be an aligned sequence"
        ) from exc
    if actual != expected:
        raise _PredictionValidationError(
            f"{name} has length {actual}; expected {expected}"
        )


def _validated_prediction(
    prediction: Any,
    structures: Sequence[Atoms],
) -> tuple[list[float], list[np.ndarray], list[np.ndarray]]:
    if not isinstance(prediction, BatchPrediction):
        raise _PredictionValidationError(
            "predictor must return a BatchPrediction"
        )

    expected = len(structures)
    fields = (
        (prediction.total_energies_ev, "total_energies_ev"),
        (prediction.forces_ev_per_a, "forces_ev_per_a"),
        (prediction.stresses_ev_per_a3, "stresses_ev_per_a3"),
    )
    for values, name in fields:
        _aligned_length(values, name=name, expected=expected)

    energies: list[float] = []
    forces: list[np.ndarray] = []
    stresses: list[np.ndarray] = []
    for item, structure in enumerate(structures):
        try:
            energy_array = np.asarray(
                prediction.total_energies_ev[item], dtype=float
            )
            force_array = np.asarray(
                prediction.forces_ev_per_a[item], dtype=float
            )
            stress_array = np.asarray(
                prediction.stresses_ev_per_a3[item], dtype=float
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise _PredictionValidationError(
                f"prediction {item} must contain numeric values"
            ) from exc

        if energy_array.shape != () or not np.isfinite(energy_array):
            raise _PredictionValidationError(
                f"prediction {item} energy must be a finite scalar"
            )
        expected_force_shape = (len(structure), 3)
        if force_array.shape != expected_force_shape:
            raise _PredictionValidationError(
                f"prediction {item} forces have shape {force_array.shape}; "
                f"expected {expected_force_shape}"
            )
        if not np.all(np.isfinite(force_array)):
            raise _PredictionValidationError(
                f"prediction {item} forces must be finite"
            )
        if stress_array.shape not in ((6,), (3, 3)):
            raise _PredictionValidationError(
                f"prediction {item} stress has shape {stress_array.shape}; "
                "expected (6,) or (3, 3)"
            )
        if not np.all(np.isfinite(stress_array)):
            raise _PredictionValidationError(
                f"prediction {item} stress must be finite"
            )

        energies.append(float(energy_array))
        forces.append(force_array.copy())
        stresses.append(stress_array.copy())
    return energies, forces, stresses


def _predicted_snapshot(
    atoms: Atoms,
    *,
    energy: float,
    forces: np.ndarray,
    stress: np.ndarray,
) -> Atoms:
    snapshot = _sanitize_structure(atoms)
    snapshot.info["total_energy"] = energy
    snapshot.info["stress"] = stress.copy()
    snapshot.new_array("forces", forces.copy())
    return snapshot


def _run_fire_trajectory(
    structures: Sequence[Atoms],
    predictor: BatchPredictor,
) -> list[StructurePrerelaxResult]:
    working = [_sanitize_structure(structure) for structure in structures]
    optimizers = [
        FIRE(atoms, logfile=None, **dict(FIRE_PARAMETERS)) for atoms in working
    ]
    saved: list[dict[int, Atoms]] = [{} for _ in working]
    force_evaluations = 0
    optimizer_updates = 0

    try:
        for step in range(9):
            force_evaluations += 1
            energies, forces, stresses = _validated_prediction(
                predictor(working), working
            )

            if step in SNAPSHOT_STEPS:
                for item, atoms in enumerate(working):
                    saved[item][step] = _predicted_snapshot(
                        atoms,
                        energy=energies[item],
                        forces=forces[item],
                        stress=stresses[item],
                    )

            if step < 8:
                for atoms, optimizer, item_forces in zip(
                    working, optimizers, forces, strict=True
                ):
                    atoms.calc = SinglePointCalculator(atoms, forces=item_forces)
                    try:
                        optimizer.step()
                    finally:
                        atoms.calc = None
                optimizer_updates += 1
    except Exception as exc:
        raise _TrajectoryFailure(
            force_evaluations=force_evaluations,
            optimizer_updates=optimizer_updates,
        ) from exc

    return [
        StructurePrerelaxResult(
            snapshots=item_snapshots,
            error=None,
            force_evaluations=force_evaluations,
            optimizer_updates=optimizer_updates,
        )
        for item_snapshots in saved
    ]


def run_fixed_cell_fire(
    structures: Sequence[Atoms],
    predictor: BatchPredictor,
    *,
    atom_budget: int,
    structure_cap: int | None = None,
) -> PrerelaxRunResult:
    """Run exactly nine predictions and eight independent FIRE updates.

    Inputs are sanitized before packing.  Any failed or invalid batch attempt is
    discarded, then every member is retried separately from its own x0.  A
    single-structure retry failure is returned as ``predictor_failed`` so that
    downstream screening can fail open without disturbing neighboring inputs.
    """

    started = perf_counter()
    clean_inputs = [_sanitize_structure(structure) for structure in structures]
    batches = pack_structures(
        clean_inputs,
        atom_budget=atom_budget,
        structure_cap=structure_cap,
    )
    forward_calls = 0
    results: list[StructurePrerelaxResult] = []

    def counted_predictor(batch: list[Atoms]) -> BatchPrediction:
        nonlocal forward_calls
        forward_calls += 1
        return predictor(batch)

    for batch in batches:
        try:
            results.extend(_run_fire_trajectory(batch, counted_predictor))
            continue
        except _TrajectoryFailure as exc:
            batch_failure = exc

        for structure in batch:
            try:
                retry_result = _run_fire_trajectory(
                    [structure], counted_predictor
                )[0]
                results.append(
                    replace(
                        retry_result,
                        force_evaluations=(
                            batch_failure.force_evaluations
                            + retry_result.force_evaluations
                        ),
                        optimizer_updates=(
                            batch_failure.optimizer_updates
                            + retry_result.optimizer_updates
                        ),
                        retry_overhead_force_evaluations=(
                            batch_failure.force_evaluations
                        ),
                        retry_overhead_optimizer_updates=(
                            batch_failure.optimizer_updates
                        ),
                    )
                )
            except _TrajectoryFailure as exc:
                results.append(
                    StructurePrerelaxResult(
                        snapshots={},
                        error="predictor_failed",
                        force_evaluations=(
                            batch_failure.force_evaluations
                            + exc.force_evaluations
                        ),
                        optimizer_updates=(
                            batch_failure.optimizer_updates
                            + exc.optimizer_updates
                        ),
                        retry_overhead_force_evaluations=(
                            batch_failure.force_evaluations
                        ),
                        retry_overhead_optimizer_updates=(
                            batch_failure.optimizer_updates
                        ),
                    )
                )

    return PrerelaxRunResult(
        results=tuple(results),
        predictor_forward_calls=forward_calls,
        elapsed_seconds=perf_counter() - started,
    )


def make_mattersim_predictor(
    checkpoint: Any,
    *,
    device: str,
    batch_size: int,
) -> BatchPredictor:
    """Create a lazy-imported MatterSim adapter and load its checkpoint once."""

    from mattersim.datasets.utils.build import build_dataloader
    from mattersim.forcefield import Potential

    dataloader_batch_size = _positive_int(batch_size, name="batch_size")
    potential = Potential.from_checkpoint(
        str(checkpoint), device=device, load_training_state=False
    )
    model_args = potential.model.model_args
    cutoff = float(model_args["cutoff"])
    threebody_cutoff = float(model_args["threebody_cutoff"])

    def predict(structures: list[Atoms]) -> BatchPrediction:
        loader = build_dataloader(
            structures,
            cutoff=cutoff,
            threebody_cutoff=threebody_cutoff,
            batch_size=dataloader_batch_size,
            only_inference=True,
        )
        energies, forces, stresses_gpa = potential.predict_properties(
            loader,
            include_forces=True,
            include_stresses=True,
        )
        stresses_ev_per_a3 = [
            np.asarray(stress, dtype=float) * GPa for stress in stresses_gpa
        ]
        return BatchPrediction(
            total_energies_ev=energies,
            forces_ev_per_a=forces,
            stresses_ev_per_a3=stresses_ev_per_a3,
        )

    return predict


def assess_prerelax_support(
    snapshots: Mapping[int, Any],
    *,
    cutoff_step: int = 8,
) -> SupportDecision:
    """Apply frozen safety limits and return a stable fail-open decision."""

    cutoff_step = _validated_cutoff_step(cutoff_step)

    try:
        summaries = summarize_snapshots(snapshots, cutoff_step=cutoff_step)
    except SnapshotValidationError as exc:
        return SupportDecision(False, exc.reason)
    except Exception:
        return SupportDecision(False, "invalid_snapshots")

    cutoff_index = SNAPSHOT_STEPS.index(cutoff_step)
    prefix_steps = SNAPSHOT_STEPS[: cutoff_index + 1]

    if any(
        _exceeds_limit(
            summary.fmax_ev_per_a,
            20.0,
            upstream_scale=summary.fmax_ev_per_a,
        )
        for step, summary in summaries.items()
        if step in prefix_steps
    ):
        return SupportDecision(False, "force_limit_exceeded")
    if _exceeds_limit(
        summaries[cutoff_step].max_displacement_from_x0_a,
        0.40,
        upstream_scale=_displacement_input_scale(
            snapshots, target_step=cutoff_step
        ),
    ):
        return SupportDecision(
            False, f"x{cutoff_step}_displacement_limit_exceeded"
        )

    for previous_step, current_step in zip(
        prefix_steps[:-1], prefix_steps[1:], strict=True
    ):
        previous_energy = summaries[previous_step].energy_per_atom_ev
        current_energy = summaries[current_step].energy_per_atom_ev
        increase = current_energy - previous_energy
        if _exceeds_limit(
            increase,
            0.02,
            upstream_scale=max(abs(previous_energy), abs(current_energy)),
        ):
            return SupportDecision(False, "adjacent_energy_increase")

    if any(
        not np.isfinite(summary.min_pair_distance_a)
        or summary.min_pair_distance_a <= 0.0
        for step, summary in summaries.items()
        if step in prefix_steps
    ):
        return SupportDecision(False, "invalid_min_pair_distance")
    return SupportDecision(True, "supported")


def _displacement_input_scale(
    snapshots: Mapping[int, Any], *, target_step: int
) -> float:
    reference = snapshots[0]
    target = snapshots[target_step]
    return max(
        1.0,
        float(np.max(np.abs(np.asarray(reference.cell.array, dtype=float)))),
        float(np.max(np.abs(np.asarray(target.cell.array, dtype=float)))),
        float(np.max(np.abs(np.asarray(reference.positions, dtype=float)))),
        float(np.max(np.abs(np.asarray(target.positions, dtype=float)))),
    )


def _exceeds_limit(value: float, limit: float, *, upstream_scale: float) -> bool:
    tolerance = (
        _ROUNDING_ERROR_MULTIPLIER
        * float(np.finfo(np.float64).eps)
        * max(1.0, abs(upstream_scale))
    )
    return value > limit + tolerance
