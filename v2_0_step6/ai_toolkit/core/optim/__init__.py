"""Optimization utilities (linear regression GD/SGD)."""

from .linear import OptimResult, batch_gradient_descent, mse_loss_and_grad, stochastic_gradient_descent

__all__ = ["OptimResult", "mse_loss_and_grad", "batch_gradient_descent", "stochastic_gradient_descent"]
