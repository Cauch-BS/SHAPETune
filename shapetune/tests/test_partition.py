from dataclasses import dataclass

import jax.numpy as jnp
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
    # All energies are in kcal/mol
    INF = jnp.finfo(jnp.float32).max
    INTERNAL_PAIRS = jnp.array(
        [
            [INF, INF, INF, -0.52309],  # A
            [INF, INF, -2.10208, INF],  # C
            [INF, -2.10208, INF, -0.88474],  # G
            [-0.52309, INF, -0.88474, INF],  # U
        ],
        dtype=jnp.float32,
    )
    TERMINAL_PAIRS = jnp.array(
        [
            [INF, INF, INF, 1.26630],  # A
            [INF, INF, -0.09070, INF],  # C
            [INF, -0.09070, INF, 0.78566],  # G
            [1.26630, INF, 0.78566, INF],  # U
        ],
        dtype=jnp.float32,
    )
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
    def __init__(
        self,
        seq: str,
        energy_pair: Array,
        energy_terminal: Array,
        energy_stack: Array,
        pair_ref: Array,
        thermal: float,
    ):
        self.seq = seq.upper().replace("T", "U")
        self.translater = bytes.maketrans(b"ACGU", b"\x00\x01\x02\x03")
        self.seqarr = jnp.frombuffer(
            seq.encode().translate(self.translater), dtype=jnp.uint8
        )
        self.bpmtx, self.efe = self.partition_array(
            self.seqarr, energy_pair, energy_terminal, energy_stack, pair_ref, thermal
        )

    def partition_array(
        self,
        seqarr: Array,
        energy_pair: Array,
        energy_terminal: Array,
        energy_stack: Array,
        pair_ref: Array,
        thermal: float,
        max_internal_loop: int = 30,
        min_sharp_turn: int = 3,
    ) -> tuple[Array, Array]:
        (n,) = seqarr.shape
        prefix_subseq = jnp.full(
            (n,),
            jnp.finfo(jnp.float32).min,
            dtype=jnp.float32,
        )  # log of the partition function for the prefix subsequence
        # prefix_subseq[i] = log(Z(i)) where Z(i) is the partition of seqarr[:i]
        pair_subseq = jnp.full(
            (n, n),
            jnp.finfo(jnp.float32).min,
            dtype=jnp.float32,
        )  # log of the partition function for the pair subsequence
        # pair_subseq[i, j] = log(Z(i, j)) where Z(i, j) is the partition of seqarr[i:j]
        prefix_subseq = prefix_subseq.at[0].set(0)  # initialize the first element
        pair_subseq = self._initialize(  # type: ignore
            seqarr, pair_subseq, energy_pair, thermal, min_sharp_turn
        )
        pair_subseq = self._zuker_update(  # type: ignore
            seqarr,
            pair_subseq,
            energy_pair,
            energy_terminal,
            energy_stack,
            thermal,
            pair_ref,
            max_internal_loop,
            min_sharp_turn,
        )
        prefix_subseq = self._pair_stack(prefix_subseq, pair_subseq)
        return jnp.exp(pair_subseq - prefix_subseq[-1]), -thermal * prefix_subseq[-1]
        # first element is the base pair probability
        # second element is the ensemble free energy

    @staticmethod
    def _initialize(seqarr, pair_subseq, energy_pair, thermal, min_sharp_turn=3):  # type: ignore
        (n,) = seqarr.shape
        i_indices, j_indices = jnp.meshgrid(
            jnp.arange(n, dtype=jnp.int32),
            jnp.arange(n, dtype=jnp.int32),
            indexing="ij",
        )
        pair_subseq = jnp.where(
            i_indices + min_sharp_turn < j_indices,
            -energy_pair[seqarr[i_indices], seqarr[j_indices]] / thermal,
            pair_subseq,
        )
        return pair_subseq

    @staticmethod
    def _zuker_update(  # type: ignore
        seqarr,
        pair_subseq,
        energy_pair,
        energy_terminal,
        energy_stack,
        thermal,
        pair_ref,
        max_internal_loop,
        min_sharp_turn,
    ):
        def stack_fn(i, j, k, l):  # type: ignore
            return (
                -(
                    jnp.where(
                        ((k - i) == 1) & ((j - l) == 1),
                        energy_pair[seqarr[i], seqarr[j]]
                        + energy_stack[
                            pair_ref[seqarr[i], seqarr[j]],
                            pair_ref[seqarr[l], seqarr[k]],
                        ],
                        energy_pair[seqarr[i], seqarr[j]]
                        + energy_terminal[seqarr[k], seqarr[l]]
                        - energy_pair[seqarr[k], seqarr[l]],
                    )
                )
                / thermal
            )  # type: ignore

        (n,) = seqarr.shape

        dk_grid, dl_grid = jnp.meshgrid(
            jnp.arange(max_internal_loop + 1, dtype=jnp.int32),  # values of k - i - 1
            jnp.arange(max_internal_loop + 1, dtype=jnp.int32),  # values of j - l - 1
            indexing="ij",
        )
        dk, dl = dk_grid.flatten(), dl_grid.flatten()
        # num_delta = dk.shape[0]

        def update_pair_subseq(pair_arr: Array, offset: int):  # type: ignore
            i_flat = jnp.arange(n, dtype=jnp.int32)
            j_flat = i_flat + offset

            valid_idx = j_flat < n

            # prepare for broadcasting
            i, j, dk_expand, dl_expand = jnp.broadcast_arrays(
                i_flat[None, :],  # shape: (1, n)
                j_flat[None, :],  # shape: (1, n)
                dk[:, None],  # shape: (num_delta, 1)
                dl[:, None],  # shape: (num_delta, 1)
            )
            k = i + dk_expand + 1  # shape: (num_delta, n)
            l = j - dl_expand - 1  # shape: (num_delta, n)
            valid_deltas = dk_expand + dl_expand < max_internal_loop
            valid_k = (k >= 0) & (k < j - 1)
            valid_l = (l > i + 1) & (l < n)
            valid_prev = (
                valid_k & valid_l & (k + min_sharp_turn < l) & (valid_deltas) & (j < n)
            )
            pair_kl = jnp.where(
                valid_prev, pair_arr[k, l], jnp.finfo(jnp.float32).min
            )  # shape: (num_delta, n)
            stacks_offset = stack_fn(i, j, k, l)  # s
            P_contrib_exp = jnp.exp(pair_kl + stacks_offset)
            P_contrib = jnp.where(
                valid_idx,
                jnp.log(jnp.sum(P_contrib_exp, axis=0)),
                jnp.finfo(jnp.float32).min,
            )

            pair_arr_new = jnp.logaddexp(
                pair_arr.at[i_flat, j_flat].get(
                    mode="fill", fill_value=jnp.finfo(jnp.float32).min
                ),
                P_contrib,
            )
            pair_arr = pair_arr.at[i_flat, j_flat].set(pair_arr_new, mode="drop")
            return pair_arr, None

        diag_offsets = jnp.arange(min_sharp_turn + 3, n, dtype=jnp.int32)
        pair_subseq, _ = lax.scan(
            update_pair_subseq,  # type: ignore
            pair_subseq,
            diag_offsets,
        )

        return pair_subseq

    @staticmethod
    @jit
    def _pair_stack(prefix_subseq, pair_subseq):  # type: ignore
        n, _ = pair_subseq.shape

        def update_prefix_subseq(prefix_arr, j):  # type: ignore
            pair_at_j = jnp.log(
                jnp.dot(jnp.exp(pair_subseq[:, j]), jnp.exp(prefix_arr))
            )
            prefix_arr = prefix_arr.at[j].set(
                jnp.logaddexp(
                    prefix_arr[j - 1],  # no pair at j
                    pair_at_j,  # pair at j
                )
            )
            return prefix_arr, None

        prefix_subseq, _ = lax.scan(
            update_prefix_subseq,  # type: ignore
            prefix_subseq,
            jnp.arange(1, n),
        )

        return prefix_subseq


if __name__ == "__main__":
    # check proper jit compilation
    partition = _Partition(
        "GAGCAAGGCUC",
        EnergyConstants.INTERNAL_PAIRS,
        EnergyConstants.TERMINAL_PAIRS,
        EnergyConstants.STACKING_PAIRS,
        EnergyConstants.PAIR_IDX,
        EnergyConstants.THERMAL_ENERGY,
    )
    jnp.set_printoptions(precision=3, suppress=True)
    print(partition.bpmtx)  # type: ignore
    print(partition.efe)  # type: ignore
    import ViennaRNA as RNA

    fc = RNA.fold_compound(partition.seq)
    print(fc.pf()[1])
    print(jnp.array(fc.bpp())[1:, 1:])
