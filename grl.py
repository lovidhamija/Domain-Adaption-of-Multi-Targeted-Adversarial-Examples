import math
import random

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReverseFunction(Function):
    """
    Gradient Reversal Function

    Forward  : Identity
    Backward : Multiply gradient by -lambda
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class RandomizedGRL(nn.Module):
    """
    Randomized Gradient Reversal Layer (RGRL)

    Features
    --------
    1. Dynamic DANN scheduling.
    2. Randomized gradient reversal strength.
    3. Randomized activation time.

    Parameters
    ----------
    lambda_min : minimum reversal strength
    lambda_max : maximum reversal strength
    p_start    : minimum activation probability
    """

    def __init__(
        self,
        lambda_min=0.10,
        lambda_max=1.00,
        p_start=0.20,
    ):
        super(RandomizedGRL, self).__init__()

        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.p_start = p_start

        # Variables to monitor training
        self.current_lambda = 0.0
        self.is_active = False

    def dann_schedule(self, epoch, total_epochs):
        """
        Standard DANN schedule.
        """
        progress = epoch / total_epochs
        return 2.0 / (1.0 + math.exp(-10 * progress)) - 1.0

    def forward(self, x, epoch, total_epochs):

        # --------------------------------------------------
        # Step 1 : Compute activation probability
        # --------------------------------------------------

        progress = epoch / total_epochs

        activation_probability = (
            self.p_start +
            (1.0 - self.p_start) * progress
        )

        # --------------------------------------------------
        # Step 2 : Decide whether to activate GRL
        # --------------------------------------------------

        if random.random() < activation_probability:

            self.is_active = True

            # DANN upper bound
            lambda_upper = self.dann_schedule(
                epoch,
                total_epochs
            )

            lambda_upper = max(
                lambda_upper,
                self.lambda_min
            )

            lambda_upper = min(
                lambda_upper,
                self.lambda_max
            )

            # Random reversal strength
            self.current_lambda = random.uniform(
                self.lambda_min,
                lambda_upper
            )

            return GradientReverseFunction.apply(
                x,
                self.current_lambda
            )

        else:

            self.is_active = False
            self.current_lambda = 0.0

            # Identity mapping
            return x

    def get_lambda(self):
        """
        Returns current lambda value.
        """
        return self.current_lambda

    def active(self):
        """
        Returns whether GRL is active.
        """
        return self.is_active
