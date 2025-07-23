import diabayes as db


def init_params():

    a = 0.01
    b = 0.9 * a
    Dc = 1e-5

    k = 1e3
    v0 = 1e-6
    v1 = 1e-5

    mu0 = 0.6
    theta0 = Dc / v0

    variables = db.Variables(mu=mu0, state=theta0)
    params = db.RSFParams(a=a, b=b, Dc=Dc)
    constants = db.RSFConstants(v0=v0, mu0=mu0)
    block_constants = db.SpringBlockConstants(k=k, v_lp=v1)

    return variables, params, constants, block_constants
