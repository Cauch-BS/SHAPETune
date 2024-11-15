import json
from functools import partial

import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array, jit

from shapetune import PACKAGEDIR


class Likelihood:

    nucs = ["A", "C", "G", "U"]
    params = json.load(open(PACKAGEDIR / "potentials" / "steele.json"))

    def __init__(self, params: dict = dict()) -> None:
        if params:
            self.params = params
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
        """
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
        seqarr_tp = self.t_p[seqarr]
        seqarr_tu = self.t_u[seqarr]
        seqarr_ksum = seqarr_kp + seqarr_ku

        gamma_term: Array = jnp.power(
            reactivities, seqarr_ksum - 1
        ) / jsp.special.gamma(seqarr_ksum)

        exp_term: Array = (
            jnp.exp(-reactivities / (pair_prob * seqarr_tp))
            * jnp.power(pair_prob * seqarr_tp, -seqarr_kp)
            * jnp.power((1 - pair_prob) * seqarr_tu, -seqarr_ku)
        )

        kummer_term: Array = jsp.special.hyp1f1(
            seqarr_ku,
            seqarr_ksum,
            (reactivities / pair_prob) * (1 / seqarr_tp - 1 / seqarr_tu),
        )

        return jnp.log(gamma_term * exp_term * kummer_term)


if __name__ == "__main__":
    likelihood = Likelihood()
    seqarr = jnp.array([0, 1, 2, 3, 0, 1, 2, 3])
    reactivites = jnp.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4])
    pair_prob = jnp.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4])
    print(likelihood(seqarr, reactivites, pair_prob))
