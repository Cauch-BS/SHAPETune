import json
from functools import partial

import jax.numpy as jnp
import jax.scipy.special as jsp
import jax.scipy.stats as jst
from jax import Array, jit

from shapetune import PACKAGEDIR


class EternaLikelihood:
    """\
    Calculate the likelihood of a sequence given reactivities and the probability that
    the nucleotides are paired. Parameters are taken from the Eterna paper.
    (Wayment-Steele et al. 2021)
    """

    nucs = ["A", "C", "G", "U"]
    params = json.load(open(PACKAGEDIR / "potentials" / "steele.json"))

    def __init__(
        self,
        params_A: Array = jnp.empty((2, 2), dtype=jnp.float32),
        params_C: Array = jnp.empty((2, 2), dtype=jnp.float32),
        params_G: Array = jnp.empty((2, 2), dtype=jnp.float32),
        params_U: Array = jnp.empty((2, 2), dtype=jnp.float32),
    ) -> None:

        for param, nuc in zip([params_A, params_C, params_G, params_U], self.nucs):
            if jnp.all(param == 0):
                continue
            self.params["k"]["paired"][nuc] = param[0, 0]
            self.params["k"]["unpaired"][nuc] = param[0, 1]
            self.params["t"]["paired"][nuc] = param[1, 0]
            self.params["t"]["unpaired"][nuc] = param[1, 1]

        self.k_p = jnp.exp(
            jnp.array([self.params["k"]["paired"][nuc] for nuc in self.nucs])
        )
        self.k_u = jnp.exp(
            jnp.array([self.params["k"]["unpaired"][nuc] for nuc in self.nucs])
        )
        self.t_p = jnp.exp(
            jnp.array([self.params["t"]["paired"][nuc] for nuc in self.nucs])
        )
        self.t_u = jnp.exp(
            jnp.array([self.params["t"]["unpaired"][nuc] for nuc in self.nucs])
        )

    def __call__(self, seqarr: Array, reactivities: Array, pair_prob: Array) -> Array:
        return self.find_liklihood(seqarr, reactivities, pair_prob)  # type: ignore

    @partial(jit, static_argnums=(0,))
    def find_liklihood(
        self, seqarr: Array, reactivities: Array, pair_prob: Array
    ) -> Array:
        """\
        Find the likelihood of a sequence given reactivities and the probability that
        the nucleotides are paired.

        Args:
            seqarr: Array of integers representing the sequence.
            reactivities: Array of floats representing the reactivities.
            pair_prob: Array of floats representing the probability that the nucleotides are paired.

        Returns:
            Array of floats representing the likelihood of the sequence.
        """
        seqarr_kp = self.k_p[seqarr]
        seqarr_ku = self.k_u[seqarr]
        seqarr_tp = self.t_p[seqarr] * pair_prob
        seqarr_tu = self.t_u[seqarr] * (1 - pair_prob)
        # index for larger t (tp or tu)
        seqarr_tm = jnp.minimum(seqarr_tp, seqarr_tu)
        seqarr_tM = jnp.maximum(seqarr_tp, seqarr_tu)
        seqarr_tM_idx = jnp.where(seqarr_tp > seqarr_tu, 1, 0)
        seqarr_kM = jnp.where(seqarr_tM_idx, seqarr_kp, seqarr_ku)
        seqarr_K = seqarr_kp + seqarr_ku

        # make extensive use of the double where trick to avoid *nan* in gradients
        # see https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
        # in most cases, I think the outside where is not necessary, but I'm keeping it for now

        normalizer: Array = seqarr_kM * jnp.where(
            seqarr_tm != 0,
            jnp.log(jnp.where(seqarr_tm != 0, seqarr_tm / seqarr_tM, 1)),
            0,
        )

        gamma_term: Array = jnp.where(
            seqarr_tm != 0,
            jst.gamma.logpdf(
                reactivities,
                seqarr_K,
                scale=jnp.where(seqarr_tm != 0, seqarr_tm, seqarr_tM),
            ),
            0,
        )

        kummer_term: Array = jnp.where(
            seqarr_tm != 0,
            jnp.log(
                jsp.hyp1f1(
                    seqarr_kM,
                    seqarr_K,
                    (reactivities)
                    * (seqarr_tM - seqarr_tm)
                    / (
                        jnp.where(
                            seqarr_tm != 0,
                            seqarr_tm * seqarr_tM,
                            1.0,
                        )
                    ),
                )
            ),
            0,
        )

        return normalizer + gamma_term + kummer_term


if __name__ == "__main__":
    likelihood = EternaLikelihood()
    seqarr = jnp.array([0, 1, 2, 3, 0, 1, 2, 3])
    reactivites = jnp.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4])
    pair_prob = jnp.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4])
    print(likelihood(seqarr, reactivites, pair_prob))
