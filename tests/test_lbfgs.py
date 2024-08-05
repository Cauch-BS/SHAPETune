import unittest
import logging
import numpy as np
from src.lbfgs_opt import LBFGS
from src.prettier import prettier_energy


class TestLBFGS(unittest.TestCase):
    def setUp(self):
        # Set up logging
        logging.basicConfig(
            filename="lbfgs_test.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(console_handler)
        self.logger = logging.getLogger(__name__)

    def test_lbfgs_optimizer(self):
        # Define a simple quadratic function and its gradient for 18 dimensions
        def quadratic_function(params):
            return sum((x - y) ** 2 for x, y in zip(params, range(1, 19)))

        def quadratic_gradient(params):
            return np.array([2 * (x - y) for x, y in zip(params, range(1, 19))])

        # Initial parameters (18 dimensions)
        initial_params = tuple(0.0 for _ in range(18))

        # Create an instance of LBFGS_Optimizer
        optimizer = LBFGS(
            params=initial_params,
            func=quadratic_function,
            grad=quadratic_gradient,
            logger=self.logger,
            lr=1.0,
            max_iter=100,
            tolerance_grad=1e-6,
            tolerance_change=1e-9,
            history_size=10,
            line_search_fn="strong_wolfe",
        )

        # Run optimization
        self.logger.info("Starting optimization")
        final_loss = optimizer.optimize()
        self.logger.info(f"Optimization complete. Final loss: {final_loss}")

        # Check if the optimized parameters are close to the expected minimum
        expected_minimum = tuple(range(1, 19))
        optimized_params = optimizer.params

        np.testing.assert_allclose(optimized_params, expected_minimum, atol=1e-5)
        self.assertLess(
            final_loss, 1e-10, f"Final loss {final_loss} is not close enough to zero"
        )

        self.logger.info("Test passed successfully!")
        self.logger.info(
            f"Optimized parameters: {prettier_energy(np.array(optimized_params))}"
        )
        self.logger.info(f"Final loss: {final_loss}")


if __name__ == "__main__":
    unittest.main()
