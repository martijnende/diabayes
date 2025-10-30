import diabayes as db
from diabayes.forward_models import Forward, ageing_law, rsf, slip_rate, springblock
from diabayes.solver import ODESolver


def data_are_valid(start, stop, fields):

    # At this point, fields["t"] must exist
    assert fields.get("t") is not None
    assert fields.get("mu") is not None

    # Check start/stop criteria
    valid_start = (start is not None) and (start >= 0)
    valid_stop = (stop is not None) and (stop > start) and (stop < len(fields["t"]))
    valid_start_stop = valid_start and valid_stop
    if not valid_start_stop:
        return False

    # Check all fields are not None
    if any(v is None for v in fields.values()):
        return False

    # Check all fields are positive
    if any(v < 0 for k, v in fields.items() if k not in ("t", "x", "mu", "v")):
        return False

    return True


def run_forward(start, stop, fields):

    state_dict = {"theta": ageing_law, "x": slip_rate}
    forward = Forward(
        friction_model=rsf, state_evolution=state_dict, stress_transfer=springblock
    )
    solver = ODESolver(forward_model=forward)

    params = db.RSFParams(a=fields["a"], b=fields["b"], Dc=fields["Dc"])
    constants = db.RSFConstants(v0=fields["v0"], mu0=fields["mu0"])
    block_constants = db.SpringBlockConstants(k=fields["k"], v_lp=fields["v1"])

    forward.set_initial_values(mu=fields["mu0"], theta=fields["theta0"], x=0.0)
    y0 = forward.variables
    result = solver.solve_forward(
        t=fields["t"][start:stop],
        y0=y0,
        params=params,
        friction_constants=constants,
        block_constants=block_constants,
    )
    v = rsf(result, params, constants)
    return result.mu, v, result.x


def run_inversion(start, stop, fields):

    state_dict = {"theta": ageing_law, "x": slip_rate}
    forward = Forward(
        friction_model=rsf, state_evolution=state_dict, stress_transfer=springblock
    )
    solver = ODESolver(forward_model=forward)

    params = db.RSFParams(a=fields["a"], b=fields["b"], Dc=fields["Dc"])
    constants = db.RSFConstants(v0=fields["v0"], mu0=fields["mu0"])
    block_constants = db.SpringBlockConstants(k=fields["k"], v_lp=fields["v1"])

    forward.set_initial_values(mu=fields["mu0"], theta=fields["theta0"], x=0.0)
    y0 = forward.variables
    inv_result = solver.max_likelihood_inversion(
        t=fields["t"][start:stop],
        mu=fields["mu"][start:stop],
        y0=y0,
        params=params,
        friction_constants=constants,
        block_constants=block_constants,
    )
    params_inv = inv_result.value
    result = solver.solve_forward(
        t=fields["t"][start:stop],
        y0=y0,
        params=params_inv,
        friction_constants=constants,
        block_constants=block_constants,
    )
    v = rsf(result, params_inv, constants)

    return result.mu, v, result.x, params_inv
