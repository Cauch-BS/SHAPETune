from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import jax.numpy as jnp
import jax.scipy.special as jsp
from jax import Array, jit, lax, vmap
from scipy.constants import Avogadro, Boltzmann, calorie, zero_Celsius


# NOTE: Magic Parameters from the RNA RedPrint paper.
# NOTE: This should be replaced with Turner Parameters in the Future.
@dataclass
class CONSTANTS:
    TRANSLATER = bytes.maketrans(b"ACGU", b"\x00\x01\x02\x03")
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
    INF = jnp.float32("inf")
    STACKING_PAIRS = jnp.array(
        [
            [
                -1.89434,  # CG
                -2.23740,  # GC
                -1.44085,  # GU
                -1.22942,  # UG
                -1.10548,  # AU
                -1.11752,  # UA
                INF,
            ],  # CG
            [
                -2.23740,  # CG
                -2.23740,  # CG
                -1.26209,  # GU
                -1.26209,  # UG
                -1.13291,  # AU
                -1.13291,  # UA
                INF,
            ],  # GC
            [
                -1.58478,  # CG
                -1.26209,  # GC
                -0.68876,  # GU
                -0.72185,  # UG
                -0.49625,  # AU
                -0.55066,  # UA
                INF,
            ],  # GU
            [
                -1.22942,  # CG
                -1.22942,  # GC
                -0.72185,  # GU
                -0.72185,  # UG
                -0.38606,  # AU
                -0.38606,  # UA
                INF,
            ],  # UG
            [
                -1.09787,  # CG
                -1.13291,  # GC
                -0.62086,  # GU
                -0.38606,  # UG
                -0.26510,  # AU
                -0.18826,  # UA
                INF,
            ],  # AU
            [
                -1.11752,  # CG
                -1.11752,  # GC
                -0.55066,  # GU
                -0.55066,  # UG
                -0.18826,  # AU
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


@partial(jit, static_argnums=(6))
def _zuker_inside(
    seqarr: Array,
    pair_prefix: Array,
    all_prefix: Array,
    energy_stack: Array,
    thermal: float = CONSTANTS.THERMAL_ENERGY,
    pair_ref: Array = CONSTANTS.PAIR_IDX,
    min_sharp_turn: int = 3,
) -> tuple[Array, Array] | Any:

    (n,) = seqarr.shape

    def update_pair_prefix(
        carry: tuple[Array, Array], offset: int | Array
    ) -> tuple[tuple[Array, Array], None]:
        pair_pref, all_pref = carry
        i = jnp.arange(n, dtype=jnp.int32)
        j = i + offset
        # valid_ij is the $\delta_{ij}$ in the above equations
        valid_ij = (pair_ref[seqarr[i], seqarr[j]] != -1) & (j < n)

        # SECTION: When $j$ pairs with $i$
        # SECTION: When $j-1$ pairs with $i+1$
        prev_pair = pair_pref[i + 1, j - 1]

        helix_energy = (
            -energy_stack[
                pair_ref[seqarr[i], seqarr[j]], pair_ref[seqarr[j - 1], seqarr[i + 1]]
            ]
            / thermal
        )  # shape (n,)
        stack = helix_energy + prev_pair
        #!SECTION: END $j-1$ pairs with $i+1$

        # SECTION: otherwise $j-1$ pairs with $k$ between $i+1$ and $j-1$ or does not pair
        prev_all = all_pref[i + 1, j]
        no_stack = jnp.where(
            valid_ij,
            prev_all + jnp.log1p(-jnp.exp(prev_pair - prev_all)),
            jnp.float32("-inf"),
        )
        # !SECTION: Otherwise
        update_pair_ij = jnp.logaddexp(stack, no_stack)
        pair_pref = pair_pref.at[i, j].set(update_pair_ij, mode="drop")
        # !SECTION: END $j$ pairs with $i$

        # SECTION: All subseq between $i$ and $j$
        # SECTION: When $j$ pairs with $i$
        pair_at_j = jsp.logsumexp(
            all_pref[i, :-1] + pair_pref[:, j].T,
            axis=1,
        )
        #!SECTION: END $j$ pairs with $k$ between $i$ and $j-1$
        # SECTION: When $j$ does not pair
        no_pair_j = all_pref[i, j]
        # !SECTION: END $j$ does not pair
        update_all_ij = jnp.logaddexp(pair_at_j, no_pair_j)
        all_pref = all_pref.at[i, j + 1].set(update_all_ij, mode="drop")
        # !SECTION: END All subseq between $i$ and $j$

        return (pair_pref, all_pref), None

    diag_offsets = jnp.arange(min_sharp_turn + 1, n, dtype=jnp.int32)
    prefixes, _ = lax.scan(
        update_pair_prefix,
        (pair_prefix, all_prefix),
        diag_offsets,
    )

    return prefixes  # type: ignore


@partial(jit, static_argnums=(8))
def _zuker_outside(
    seqarr: Array,
    pair_suffix: Array,
    all_suffix: Array,
    pair_prefix: Array,
    all_prefix: Array,
    energy_stack: Array,
    thermal: float = CONSTANTS.THERMAL_ENERGY,
    pair_ref: Array = CONSTANTS.PAIR_IDX,
    min_sharp_turn: int = 3,
) -> tuple[Array, Array] | Any:

    (n,) = seqarr.shape

    initial = jnp.log1p(pair_ref[seqarr[0], seqarr[n - 1]] != -1)
    all_suffix = all_suffix.at[1, n - 1].set(initial, mode="drop")

    def update_pair_suffix(
        carry: tuple[Array, Array], offset: int | Array
    ) -> tuple[tuple[Array, Array], None]:
        pair_suff, all_suff = carry
        i = jnp.arange(1, n - 1, dtype=jnp.int32)
        j = i + offset
        # valid_ij is the $\delta_{ij}$ in the above equations
        valid_ij = pair_ref[seqarr[i], seqarr[j]] != -1

        # SECTION: When $j$ pairs with $i$
        # SECTION: When $j-1$ pairs with $i+1$
        prev_pair = pair_suff[i - 1, j + 1]
        helix_energy = (
            -energy_stack[
                pair_ref[seqarr[i - 1], seqarr[j + 1]], pair_ref[seqarr[j], seqarr[i]]
            ]
            / thermal
        )  # shape (n,)
        stack = helix_energy + prev_pair
        #!SECTION: END $j-1$ pairs with $i+1$

        # SECTION: Otherwise
        prev_all = all_suff[i, j + 1]
        no_stack = jnp.where(
            valid_ij,
            prev_all + jnp.log1p(-jnp.exp(prev_pair - prev_all)),
            jnp.float32("-inf"),
        )
        # !SECTION: Otherwise
        update_pair_ij = jnp.logaddexp(stack, no_stack)
        update = jnp.where(
            j < n - 1,
            update_pair_ij,
            pair_suff[i, j],
        )
        pair_suff = pair_suff.at[i, j].set(update, mode="drop")
        # !SECTION: END $j$ pairs with $i$

        # SECTION: All subseq between $i$ and $j$
        # SECTION: When $j$ pairs after $j$
        push_at_j = jsp.logsumexp(
            all_suff[i, 1:] + pair_prefix[j, :],
            axis=1,
        )
        #!SECTION: END $j$ pairs after j
        # SECTION: When $j$ pairs before $j$
        pull_at_j = jsp.logsumexp(
            all_prefix[1:, i] + pair_suff[:-1, j],
            axis=0,
        )
        #!SECTION: END $j$ pairs before $j$
        # SECTION: When $j$ does not pair
        no_pair_j = all_suff[i, j + 1]
        # !SECTION: END $j$ does not pair
        update_all_ij = jnp.logaddexp(jnp.logaddexp(push_at_j, pull_at_j), no_pair_j)
        update = jnp.where(
            j < n,
            update_all_ij,
            all_suff[i, j],
        )
        all_suff = all_suff.at[i, j].set(update, mode="drop")
        # !SECTION: END All subseq between $i$ and $j$
        return (pair_suff, all_suff), None

    diag_offsets = jnp.arange(n - 3, min_sharp_turn, -1, dtype=jnp.int32)
    suffixes, _ = lax.scan(
        update_pair_suffix,
        (pair_suffix, all_suffix),
        diag_offsets,
    )

    return suffixes


@partial(jit, static_argnums=(3, 4))
def _partition_arr(
    seqarr: Array,
    energy_stack: Array,
    zuker_inside: Callable = _zuker_inside,
    zuker_outside: Callable = _zuker_outside,
    pair_ref: Array = CONSTANTS.PAIR_IDX,
    min_sharp_turn: int = 3,
) -> tuple[Array, Array, Array, Array]:
    (n,) = seqarr.shape
    pair_prefix: Array = jnp.full((n, n), jnp.float32("-inf"))

    i, j = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")
    p, q = jnp.meshgrid(jnp.arange(n), jnp.arange(n + 1), indexing="ij")

    all_prefix: Array = jnp.where(
        (p <= q) & (q <= p + min_sharp_turn + 1),
        0.0,
        jnp.float32("-inf"),
    )

    pair_prefix, all_prefix = zuker_inside(
        seqarr,
        pair_prefix,
        all_prefix,
        energy_stack,
        min_sharp_turn=min_sharp_turn,
    )

    pair_suffix: Array = jnp.where(
        ((i == 0) | (j == n - 1)) & (pair_ref[seqarr[i], seqarr[j]] != -1),
        jnp.where(
            j == n - 1,
            all_prefix[0, i],
            all_prefix[j + 1, n],
        ),
        jnp.float32("-inf"),
    )

    all_suffix: Array = jnp.where(
        ((p == 0) | (q == n)) & (q != 0),
        jnp.where(
            q == n,
            all_prefix[0, p],
            all_prefix[q, n],
        ),
        jnp.float32("-inf"),
    )

    pair_suffix, all_suffix = zuker_outside(
        seqarr,
        pair_suffix,
        all_suffix,
        pair_prefix,
        all_prefix,
        energy_stack,
        min_sharp_turn=min_sharp_turn,
    )

    return pair_prefix, pair_suffix, all_prefix, all_suffix
    # first element is the base pair probability
    # second element is the ensemble free energy


def partition(
    seqs: list[str] | str,
    energy_stack: Array,
    constants: CONSTANTS = CONSTANTS,  # type: ignore
) -> tuple[Array, Array]:
    if isinstance(seqs, str):
        seqs = [seqs]
    seqarrs = jnp.vstack(
        [
            jnp.frombuffer(
                seq.upper().replace("T", "U").encode().translate(constants.TRANSLATER),
                dtype=jnp.uint8,
            )
            for seq in seqs
        ],
        dtype=jnp.uint8,
    )

    print(seqarrs)

    pair_prefixes, pair_suffixes, all_prefixes, _ = vmap(
        partial(_partition_arr, energy_stack=energy_stack),
        in_axes=0,
        out_axes=0,
    )(seqarrs)

    @jit
    def bp_prob(prefix: Array, suffix: Array, norm: Array) -> Array:
        norm = norm[:, None, None]
        return jnp.exp(prefix + suffix - norm)

    bpp = bp_prob(pair_prefixes, pair_suffixes, all_prefixes[:, 0, -1])
    efe = -constants.THERMAL_ENERGY * all_prefixes[:, 0, -1]

    return bpp, efe


# SECTION: SAMPLE USAGE
if __name__ == "__main__":

    seq = "AAGGGAAACCCAAAGGGAAA"
    energy_stack = CONSTANTS.STACKING_PAIRS
    bpp, efe = partition(seq, energy_stack)
    jnp.set_printoptions(precision=3, suppress=True)
    print(f"Base Pair Probability:\n{bpp}")
    print(f"Ensemble Free Energy: {efe} kcal/mol")

    # import numpy as np
    # import ViennaRNA

    # fc = ViennaRNA.fold_compound(seq)
    # _, efe = fc.pf()
    # bpp = np.array(fc.bpp())[1:, 1:]
    # np.set_printoptions(precision=3, suppress=True)
    # print(bpp)
    # print(efe)
# !SECTION: END SAMPLE USAGE
