from typing import Tuple, Union

import diffrax as dfx
import jax
from jaxtyping import Array, Float

from diabayes.forward_models import Forward
from diabayes.typedefs import Variables, _BlockConstants, _Constants, _Params

jax.config.update("jax_enable_x64", True)


class ODESolver:

    forward_model: Forward
    rtol: float
    atol: float
    checkpoints: int

    def __init__(
        self,
        forward_model: Forward,
        rtol: float = 1e-10,
        atol: float = 1e-12,
        checkpoints: int = 100,
    ) -> None:
        self.forward_model = forward_model
        self.rtol = rtol
        self.atol = atol
        self.checkpoints = checkpoints
        pass

    def _forward_wrapper(
        self,
        t: Float,
        variables: Variables,
        args: Tuple[_Params, _Constants, _BlockConstants],
    ) -> Variables:
        params, friction_constants, block_constants = args
        return self.forward_model(
            variables=variables,
            params=params,
            friction_constants=friction_constants,
            block_constants=block_constants,
        )

    def solve_forward(
        self,
        t: Float[Array, "Nt"],
        y0: Variables,
        params: _Params,
        friction_constants: _Constants,
        block_constants: _BlockConstants,
        adjoint: Union[None, dfx.AbstractAdjoint] = None,
    ) -> dfx.Solution:

        term = dfx.ODETerm(self._forward_wrapper)
        t0 = t.min()
        t1 = t.max()
        dt0 = t[1] - t[0]
        saveat = dfx.SaveAt(ts=t)
        args = (params, friction_constants, block_constants)

        controller = dfx.PIDController(rtol=self.rtol, atol=self.atol)
        if adjoint is None:
            adjoint = dfx.RecursiveCheckpointAdjoint(checkpoints=self.checkpoints)
        assert isinstance(adjoint, dfx.AbstractAdjoint)

        sol = dfx.diffeqsolve(
            terms=term,
            solver=dfx.Tsit5(),
            t0=t0,
            t1=t1,
            dt0=dt0,
            saveat=saveat,
            y0=y0,
            args=args,
            stepsize_controller=controller,
            adjoint=adjoint,
        )

        assert sol is not None

        return sol
