from dataclasses import dataclass
from functools import partial

import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array, jit, lax
from scipy.constants import Avogadro, Boltzmann, calorie, zero_Celsius


@dataclass
class EnergyConstants:
    PAIR_IDX = jnp.array(
        [
            # A  C  G  U
            [-1, -1, -1, 4],  # A
            [-1, -1, 0, -1],  # C
            [-1, 1, -1, 2],  # G
            [5, -1, 3, -1],  # U
        ],
        dtype=jnp.int32,
    )
    """\
    Corresponds to the index of the pair in stacking energies.
    """
    INF = jnp.finfo(jnp.float32).max
    STACKING_PAIRS = jnp.array(
        [
            [
                -2.23740,  # CG
                -1.89434,  # GC
                -1.22942,  # GU
                -1.44085,  # UG
                -1.11752,  # AU
                -1.10548,  # UA
                INF,
            ],  # CG
            [
                -1.89434,  # CG
                -2.23740,  # GC
                -1.44085,  # GU
                -1.22942,  # UG
                -1.10548,  # AU
                -1.11752,  # UA
                INF,
            ],  # GC
            [
                -1.26209,  # CG
                -1.58478,  # GC
                -0.72185,  # GU
                -0.68876,  # UG
                -0.55066,  # AU
                -0.49625,  # UA
                INF,
            ],  # GU
            [
                -1.58478,  # CG
                -1.26209,  # GC
                -0.68876,  # GU
                -0.72185,  # UG
                -0.49625,  # AU
                -0.55066,  # UA
                INF,
            ],  # UG
            [
                -1.13291,  # CG
                -1.09787,  # GC
                -1.22942,  # GU
                -1.44085,  # UG
                -0.18826,  # AU
                -0.26510,  # UA
                INF,
            ],  # AU
            [
                -1.09787,  # CG
                -1.13291,  # GC
                -1.44085,  # GU
                -1.22942,  # UG
                -0.26510,  # AU
                -0.18826,  # UA
                INF,
            ],  # UA
            [
                INF,
                INF,
                INF,
                INF,
                INF,
                INF,
                INF,
            ],
        ],
        dtype=jnp.float32,
    )
    THERMAL_ENERGY = Boltzmann * (zero_Celsius + 37) * Avogadro / calorie * 0.001


class _Partition:
    # thermal energy in kcal/mol
    def __init__(self) -> None:
        self.translater = bytes.maketrans(b"ACGU", b"\x00\x01\x02\x03")

    @partial(jit, static_argnums=(0,))
    def _partition_arr(
        self,
        seqarr: Array,
        energy_stack: Array,
    ) -> tuple[Array, Array, Array]:
        (n,) = seqarr.shape
        all_ensemble = jnp.full(
            (n,),
            jnp.finfo(jnp.float32).min,
            dtype=jnp.float32,
        )  # log of the partition function for the prefix subsequence
        # all_ensemble[i] = log(Z(i)) where Z(i) is the partition of seqarr[:i]
        pair_prefix = self._initialize_zuker(seqarr)
        pair_suffix = self._initialize_zuker(seqarr)
        all_ensemble = all_ensemble.at[0].set(0)  # initialize the first element
        pair_prefix = self._zuker_prefix(
            seqarr,
            pair_prefix,
            energy_stack,
        )
        pair_suffix = self._zuker_suffix(
            seqarr,
            pair_suffix,
            energy_stack,
        )
        all_ensemble = self._pair_stack(all_ensemble, pair_prefix)
        return pair_prefix, pair_suffix, all_ensemble
        # first element is the base pair probability
        # second element is the ensemble free energy

    @staticmethod
    @jit
    def _initialize_zuker(
        seqarr: Array,
        pair_ref: Array = EnergyConstants.PAIR_IDX,
        min_sharp_turn: int = 3,
    ) -> Array:
        (n,) = seqarr.shape
        i, j = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")
        pair_idx = pair_ref[seqarr[i], seqarr[j]]
        pairs: Array = jnp.where(
            (pair_idx != -1) & (i + min_sharp_turn < j),
            0.0,
            jnp.finfo(jnp.float32).min,
        )
        return pairs

    @staticmethod
    @partial(jit, static_argnums=(5))
    def _zuker_prefix(
        seqarr: Array,
        pair_prefix: Array,
        energy_stack: Array,
        thermal: float = EnergyConstants.THERMAL_ENERGY,
        pair_ref: Array = EnergyConstants.PAIR_IDX,
        min_sharp_turn: int = 3,
    ) -> Array:

        (n,) = seqarr.shape

        def update_pair_subseq(pair_arr: Array, offset: int) -> tuple[Array, None]:
            i = jnp.arange(n, dtype=jnp.int32)
            j, k, l = i + offset, i + 1, i + offset - 1
            pair_kl = pair_arr.at[k, l].get(
                mode="fill", fill_value=jnp.finfo(jnp.float32).min
            )
            helix_energy = (
                -energy_stack[
                    pair_ref[seqarr[i], seqarr[j]], pair_ref[seqarr[l], seqarr[k]]
                ]
                / thermal
            )  # shape (n,)
            pair_at_offset = helix_energy + pair_kl
            pair_arr = pair_arr.at[i, j].set(pair_at_offset, mode="drop")
            return pair_arr, None

        diag_offsets = jnp.arange(min_sharp_turn + 3, n, dtype=jnp.int32)
        pair_prefix, _ = lax.scan(
            update_pair_subseq,  # type: ignore
            pair_prefix,
            diag_offsets,
        )

        return pair_prefix

    @staticmethod
    @partial(jit, static_argnums=(5))
    def _zuker_suffix(
        seqarr: Array,
        pair_suffix: Array,
        energy_stack: Array,
        thermal: float = EnergyConstants.THERMAL_ENERGY,
        pair_ref: Array = EnergyConstants.PAIR_IDX,
        min_sharp_turn: int = 3,
    ) -> Array:

        (n,) = seqarr.shape

        def update_pair_suffix(pair_suff: Array, offset: int) -> tuple[Array, None]:
            i = jnp.arange(n, dtype=jnp.int32)
            j, k, l = i + offset, i - 1, i + offset + 1
            pair_kl = pair_suff.at[k, l].get(
                mode="fill", fill_value=jnp.finfo(jnp.float32).min
            )
            helix_energy = (
                -energy_stack[
                    pair_ref[seqarr[i], seqarr[j]], pair_ref[seqarr[l], seqarr[k]]
                ]
                / thermal
            )

            pair_at_offset = helix_energy + pair_kl
            pair_suff = pair_suff.at[i, j].set(pair_at_offset, mode="drop")
            return pair_suff, None

        diag_offsets = jnp.arange(n - 3, min_sharp_turn, -1, dtype=jnp.int32)
        pair_suffix, _ = lax.scan(
            update_pair_suffix,  # type: ignore
            pair_suffix,
            diag_offsets,
        )

        return pair_suffix

    @staticmethod
    @jit
    def _pair_stack(all_ensemble: Array, pair_prefix: Array) -> Array:
        n, _ = pair_prefix.shape

        def update_all_ensemble(prefix_arr: Array, j: int) -> tuple[Array, None]:
            pair_at_j = jsp.special.logsumexp(pair_prefix[:, j] + prefix_arr[:])
            prefix_arr = prefix_arr.at[j].set(
                jnp.logaddexp(
                    prefix_arr[j - 1],  # no pair at j
                    pair_at_j,  # pair at j
                )
            )
            return prefix_arr, None

        all_ensemble, _ = lax.scan(
            update_all_ensemble,  # type: ignore
            all_ensemble,
            jnp.arange(1, n),
        )

        return all_ensemble
