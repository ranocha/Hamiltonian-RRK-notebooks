import numpy as np
from scipy import interpolate
import scipy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import root, fsolve, newton, brentq, bisect

# line cycler adapted to colourblind people
from cycler import cycler
line_cycler   = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["-", "--", "-.", ":", "-", "--", "-."]))
marker_cycler = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["none", "none", "none", "none", "none", "none", "none"]) +
                 cycler(marker=["4", "2", "3", "1", "+", "x", "."]))
plt.rc("axes", prop_cycle=line_cycler)

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{newpxtext}\usepackage{newpxmath}\usepackage{commath}\usepackage{mathtools}")
plt.rc("font", family="serif", size=16.)
plt.rc("savefig", dpi=200)
plt.rc("legend", fontsize="medium", fancybox=True, framealpha=0.5)
plt.rc("lines", linewidth=2.5, markersize=10, markeredgewidth=2.5)


from nodepy import *

ssp22 = rk.loadRKM('SSP22').__num__()
heun = rk.loadRKM('Heun33').__num__()
ssp33 = rk.loadRKM('SSP33').__num__()
rk4 = rk.loadRKM('RK44').__num__()
fehlberg45 = rk.loadRKM("Fehlberg45").__num__()
dp5 = rk.loadRKM('DP5').__num__()
bs5 = rk.loadRKM('BS5').__num__()

rk_methods = [ssp22, heun, ssp33, dp5, bs5]

trbdf2 = rk.loadRKM('TR-BDF2')
lobattoA2 = rk.loadRKM('LobattoIIIA2')

rk_methods = [ssp22, heun, ssp33, dp5, bs5]


class SolveForGammaException(Exception):
    def __init__(self, message, data):
        self.message = message
        self.data = data

def projection_ERK(rkm, dt, f, eta, deta, w0, t_final):
    """Explicit Projection Runge-Kutta method."""
    rkm = rkm.__num__()

    w = np.array(w0) # current value of the unknown function
    t = 0 # current time
    ww = np.zeros([np.size(w0), 1]) # values at each time step
    ww[:,0] = w.copy()
    tt = np.zeros(1) # time points for ww
    tt[0] = t
    b = rkm.b
    s = len(rkm)
    y = np.zeros((s, np.size(w0))) # stage values
    F = np.zeros((s, np.size(w0))) # stage derivatives
    eta0 = eta(w0)

    while t < t_final and not np.isclose(t, t_final):
        if t + dt > t_final:
            dt = t_final - t

        for i in range(s):
            y[i,:] = w.copy()
            for j in range(i):
                y[i,:] += rkm.A[i,j]*dt*F[j,:]
            F[i,:] = f(y[i,:])

        w = w + dt*sum([b[i]*F[i] for i in range(s)])
        t += dt

        lamda = 0
        dlam = 10
        while dlam >1.e-14:
            dg = deta(w)
            dlam = -(eta(w+dg*lamda)-eta0)/(np.dot(dg,dg)+1.e-16)
            lamda += dlam

        w = w + dg*lamda

        tt = np.append(tt, t)
        ww = np.append(ww, np.reshape(w.copy(), (len(w), 1)), axis=1)

    return tt, ww

def projection_DIRK(rkm, dt, f, eta, deta, w0, t_final):
    """Explicit Projection Runge-Kutta method."""
    rkm = rkm.__num__()

    w = np.array(w0) # current value of the unknown function
    t = 0 # current time
    ww = np.zeros([np.size(w0), 1]) # values at each time step
    ww[:,0] = w.copy()
    tt = np.zeros(1) # time points for ww
    tt[0] = t
    b = rkm.b
    s = len(rkm)
    y = np.zeros((s, np.size(w0))) # stage values
    F = np.zeros((s, np.size(w0))) # stage derivatives
    eta0 = eta(w0)

    while t < t_final and not np.isclose(t, t_final):
        if t + dt > t_final:
            dt = t_final - t

        for i in range(s):
            stageeq = lambda Y: (Y - w - dt*sum([rkm.A[i,j]*F[j,:] for j in range(i)]) \
                                 - dt*rkm.A[i,i]*f(Y)).squeeze()
            nexty, info, ier, mesg = fsolve(stageeq,w,full_output=1)
            if ier != 1:
                print(mesg)
                # print(info)
                # raise Exception("System couldn't be solved.")
            y[i,:] = nexty.copy()
            F[i,:] = f(y[i,:])

        w = w + dt*sum([b[i]*F[i] for i in range(s)])
        t += dt

        lamda = 0
        dlam = 10
        while dlam >1.e-14:
            dg = deta(w)
            dlam = -(eta(w+dg*lamda)-eta0)/(np.dot(dg,dg)+1.e-16)
            lamda += dlam

        w = w + dg*lamda

        tt = np.append(tt, t)
        ww = np.append(ww, np.reshape(w.copy(), (len(w), 1)), axis=1)

    return tt, ww


def convex_relaxed_ERK(rkm, dt, f, eta, deta, w0, t_final,
                       relaxed=True, method="brentq", tol=1.e-14, maxiter=10000, jac=False, newdt=True,
                       debug=False, correct_last_step=True, print_gamma=False):
    """Relaxed explicit Runge-Kutta method for convex functionals. It is also possible to apply
    these schemes to general functionals. In that case, some root finding procedure have to be adapted
    slightly."""

    rkm = rkm.__num__()

    w = np.array(w0) # current value of the unknown function
    t = 0 # current time
    ww = np.zeros([np.size(w0), 1]) # values at each time step
    ww[:,0] = w.copy()
    tt = np.zeros(1) # time points for ww
    gg = np.ones(1)  # values of gamma
    tt[0] = t
    b = rkm.b
    s = len(rkm)
    y = np.zeros((s, np.size(w0))) # stage values
    F = np.zeros((s, np.size(w0))) # stage derivatives
    max_gammam1 = 0.  # max(gamma-1) over all timesteps
    old_gamma = 1.0


    # Because of the scaling by gam, the time step which should hit t_final might be a bit too short.
    # In that case, accept this step as the last one in order to terminate the integration.
    while t < t_final and not np.isclose(t, t_final):
        if t + dt > t_final:
            dt = t_final - t

        for i in range(s):
            y[i,:] = w.copy()
            for j in range(i):
                y[i,:] += rkm.A[i,j]*dt*F[j,:]
            F[i,:] = f(y[i,:])

        if relaxed and ((not np.isclose(dt, t_final - t)) or correct_last_step):
            direction = dt * sum([b[i]*F[i,:] for i in range(s)])
            estimate = dt * sum([b[i]*np.dot(deta(y[i,:]),F[i,:]) for i in range(s)])

            r = lambda gamma: eta(w+gamma*direction) - eta(w) - gamma*estimate
            if debug:
                print('r(1): ', r(1))
            rjac= lambda gamma: np.array([np.dot(deta(w+gamma*direction), direction) - estimate])

            if rjac == False:
                use_jac = False
            else:
                use_jac = rjac

            if method == "newton":
                gam = newton(r, old_gamma, fprime=rjac, tol=tol, maxiter=maxiter)
                success = True
                msg = "Newton method did not converge"
            elif method == "brentq" or method == "bisect":
                # For convex functionals, additional insights are provided: There is exactly one root
                # and r is negative for smaller gamma and positive for bigger gamma. Thus, we can use
#                 left = 0.9 * old_gamma
#                 right = 1.1 * old_gamma
#                 while r(left) > 0:
#                     right = left
#                     left *= 0.5
#                 while r(right) < 0:
#                     left = right
#                     right *= 2.0
                # For general functionals, we might need to use omething like:
#                 left = old_gamma - 0.1
#                 right = old_gamma + 0.1
#                 while r(left) * r(right) > 0:
#                     left -= 0.1
#                     right += 0.1
                left = 0.9 * old_gamma
                right = 1.1 * old_gamma
                left_right_iter = 0
                while r(left) * r(right) > 0:
                    left *= 0.9
                    right *= 1.1
                    left_right_iter += 1
                    if left_right_iter > 100:
                        raise SolveForGammaException(
                            "No suitable bounds found after %d iterations.\nLeft = %e; r(left) = %e\nRight = %e; r(right) = %e\n"%(
                                left_right_iter, left, r(left), right, r(right)),
                            w)

                if method == "brentq":
                    gam = brentq(r, left, right, xtol=tol, maxiter=maxiter)
                else:
                    gam = bisect(r, left, right, xtol=tol, maxiter=maxiter)
                success = True
                msg = "%s method did not converge"%method
            else:
                sol = root(r, old_gamma, jac=use_jac, method=method, tol=tol,
                           options={'xtol': tol, 'maxiter': maxiter})
                gam = sol.x; success = sol.success; msg = sol.message

            if success == False:
                print('Warning: fsolve did not converge.')
                print(gam)
                print(msg)

            if gam <= 0:
                print('Warning: gamma is negative.')

        else:
            gam = 1.

        old_gamma = gam

        if debug:
            gm1 = np.abs(1.-gam)
            max_gammam1 = max(max_gammam1,gm1)
            if gm1 > 0.5:
                print(gam)
                raise Exception("The time step is probably too large.")

        w = w + dt*gam*sum([b[i]*F[i] for i in range(s)])
        if newdt == True:
            t += gam*dt
        else:
            t += dt

        tt = np.append(tt, t)
        ww = np.append(ww, np.reshape(w.copy(), (len(w), 1)), axis=1)
        gg = np.append(gg, gam)

    if debug:
        if print_gamma:
            print(max_gammam1)
        return tt, ww, gg
    else:
        return tt, ww



def convex_relaxed_DIRK(rkm, dt, f, eta, deta, w0, t_final,
                        relaxed=True, method="brentq", tol=1.e-14, maxiter=10000, jac=False, newdt=True,
                        debug=False, correct_last_step=True, print_gamma=False):
    """Relaxed diagonally implicit Runge-Kutta method for convex functionals. It is also possible to apply
    these schemes to general functionals. In that case, some root finding procedure have to be adapted
    slightly."""

    rkm = rkm.__num__()

    w = np.array(w0) # current value of the unknown function
    t = 0 # current time
    ww = np.zeros([np.size(w0), 1]) # values at each time step
    ww[:,0] = w.copy()
    tt = np.zeros(1) # time points for ww
    gg = np.ones(1)  # values of gamma
    tt[0] = t
    b = rkm.b
    s = len(rkm)
    y = np.zeros((s, np.size(w0))) # stage values
    F = np.zeros((s, np.size(w0))) # stage derivatives
    max_gammam1 = 0.  # max(gamma-1) over all timesteps
    old_gamma = 1.0


    # Because of the scaling by gam, the time step which should hit t_final might be a bit too short.
    # In that case, accept this step as the last one in order to terminate the integration.
    while t < t_final and not np.isclose(t, t_final):
        if t + dt > t_final:
            dt = t_final - t

        for i in range(s):
            stageeq = lambda Y: (Y - w - dt*sum([rkm.A[i,j]*F[j,:] for j in range(i)]) \
                                 - dt*rkm.A[i,i]*f(Y)).squeeze()
            nexty, info, ier, mesg = fsolve(stageeq,w,full_output=1)
            if ier != 1:
                print(mesg)
                # print(info)
                # raise Exception("System couldn't be solved.")
            y[i,:] = nexty.copy()
            F[i,:] = f(y[i,:])

        if relaxed and ((not np.isclose(dt, t_final - t)) or correct_last_step):
            direction = dt * sum([b[i]*F[i,:] for i in range(s)])
            estimate = dt * sum([b[i]*np.dot(deta(y[i,:]),F[i,:]) for i in range(s)])

            r = lambda gamma: eta(w+gamma*direction) - eta(w) - gamma*estimate
            if debug:
                print('r(1): ', r(1))
            rjac= lambda gamma: np.array([np.dot(deta(w+gamma*direction), direction) - estimate])

            if rjac == False:
                use_jac = False
            else:
                use_jac = rjac

            if method == "newton":
                gam = newton(r, old_gamma, fprime=rjac, tol=tol, maxiter=maxiter)
                success = True
                msg = "Newton method did not converge"
            elif method == "brentq" or method == "bisect":
                # For convex functionals, additional insights are provided: There is exactly one root
                # and r is negative for smaller gamma and positive for bigger gamma. Thus, we can use
#                 left = 0.9 * old_gamma
#                 right = 1.1 * old_gamma
#                 while r(left) > 0:
#                     right = left
#                     left *= 0.5
#                 while r(right) < 0:
#                     left = right
#                     right *= 2.0
                # For general functionals, we might need to use omething like:
#                 left = old_gamma - 0.1
#                 right = old_gamma + 0.1
#                 while r(left) * r(right) > 0:
#                     left -= 0.1
#                     right += 0.1
                left = 0.9 * old_gamma
                right = 1.1 * old_gamma
                left_right_iter = 0
                while r(left) * r(right) > 0:
                    left *= 0.9
                    right *= 1.1
                    left_right_iter += 1
                    if left_right_iter > 100:
                        raise SolveForGammaException(
                            "No suitable bounds found after %d iterations.\nLeft = %e; r(left) = %e\nRight = %e; r(right) = %e\n"%(
                                left_right_iter, left, r(left), right, r(right)),
                            w)

                if method == "brentq":
                    gam = brentq(r, left, right, xtol=tol, maxiter=maxiter)
                else:
                    gam = bisect(r, left, right, xtol=tol, maxiter=maxiter)
                success = True
                msg = "%s method did not converge"%method
            else:
                sol = root(r, old_gamma, jac=use_jac, method=method, tol=tol,
                           options={'xtol': tol, 'maxiter': maxiter})
                gam = sol.x; success = sol.success; msg = sol.message

            if success == False:
                print('Warning: fsolve did not converge.')
                print(gam)
                print(msg)

            if gam <= 0:
                print('Warning: gamma is negative.')

        else:
            gam = 1.

        old_gamma = gam

        if debug:
            gm1 = np.abs(1.-gam)
            max_gammam1 = max(max_gammam1,gm1)
            if gm1 > 0.5:
                print(gam)
                raise Exception("The time step is probably too large.")

        w = w + dt*gam*sum([b[i]*F[i] for i in range(s)])
        if newdt == True:
            t += gam*dt
        else:
            t += dt

        tt = np.append(tt, t)
        ww = np.append(ww, np.reshape(w.copy(), (len(w), 1)), axis=1)
        gg = np.append(gg, gam)

    if debug:
        if print_gamma:
            print(max_gammam1)
        return tt, ww, gg
    else:
        return tt, ww


def compute_rest(rkm, dt, f, eta, deta, w0):
    """Compute the term which is set to zero by relaxed explicit Runge-Kutta methods for general convex quantities."""
    s = len(rkm)
    y = np.zeros((s, len(w0))) # stage values
    F = np.zeros((s, len(w0))) # right hand sides
    for i in range(s):
        y[i,:] = w0.copy()
        for j in range(i):
            y[i,:] += rkm.A[i,j]*dt*F[j,:]
        F[i,:] = f(y[i,:])

    direction = dt * sum([rkm.b[i]*F[i] for i in range(s)])
    estimate = dt * sum([rkm.b[i]*np.dot(deta(y[i,:]),F[i]) for i in range(s)])
    r = lambda gamma: eta(w0+gamma*direction) - eta(w0) - gamma*estimate
    return r


def relaxed_ERK(rkm, dt, f, w0=[1.,0], t_final=1., relaxed=True,
                newdt=True, debug=False, correct_last_step=True, gammatol=0.25, print_gamma=False):
    """
    Relaxation Runge-Kutta method implementation.

    Options:

        rkm: Base Runge-Kutta method, in Nodepy format
        dt: time step size
        f: RHS of ODE system
        w0: Initial data
        t_final: final solution time
        relaxed: if True, use relaxation method.  Otherwise, use vanilla RK method.
        newdt: if True, new time step is t_n + \gamma dt
        debug: output some additional diagnostics
        gammatol: Fail if abs(1-gamma) exceeds this value

    """

    rkm = rkm.__num__()

    w = np.array(w0)
    t = 0
    # We pre-allocate extra space because if newdt==True then
    # we don't know exactly how many steps we will take.
    ww = np.zeros([np.size(w0), 1]) # values at each time step
    ww[:,0] = w.copy()
    tt = np.zeros(1) # time points for ww
    gg = np.ones(1)  # values of gamma
    ii = 0
    s = len(rkm)
    b = rkm.b
    y = np.zeros((s,len(w0)))
    max_gammam1 = 0.

    while t < t_final:
        if t + dt >= t_final:
            dt = t_final - t # Hit final time exactly

        for i in range(s):
            y[i,:] = w.copy()
            for j in range(i):
                y[i,:] += rkm.A[i,j]*dt*f(y[j,:])

        F = np.array([f(y[i,:]) for i in range(s)])

        if relaxed and ((not np.isclose(dt, t_final - t)) or correct_last_step):
            numer = 2*sum(b[i]*rkm.A[i,j]*np.dot(F[i],F[j]) \
                                for i in range(s) for j in range(s))
            denom = sum(b[i]*b[j]*np.dot(F[i],F[j]) for i in range(s) for j in range(s))
            if denom != 0:
                gam = numer/denom
            else:
                gam = 1.
        else:  # Use standard RK method
            gam = 1.

        if print_gamma:
            print(gam)

        if np.abs(gam-1.) > gammatol:
            print(gam)
            raise Exception("The time step is probably too large.")

        w = w + gam*dt*sum([b[j]*F[j] for j in range(s)])
        if (t+dt < t_final) and newdt:
            t += gam*dt
        else:
            t += dt
        ii += 1
        tt = np.append(tt, t)
        ww = np.append(ww, np.reshape(w.copy(), (len(w), 1)), axis=1)
        if debug:
            gm1 = np.abs(1.-gam)
            max_gammam1 = max(max_gammam1,gm1)
            gg = np.append(gg, gam)

    if debug:
        if print_gamma:
            print(max_gammam1)
        return tt, ww, gg
    else:
        return tt, ww


def compute_eoc(dts, errors):
    eocs = np.zeros(errors.size - 1)
    for i in np.arange(eocs.size):
        eocs[i] = np.log(errors[i+1] / errors[i]) / np.log(dts[i+1] / dts[i])
    return eocs


def convex_relaxed_ERK_one_step(rkm, dt, f, t, eta, deta, w, old_gamma=1., 
                       relaxed=True, method="brentq", tol=1.e-14, maxiter=10000, jac=False, newdt=True,
                       debug=False, print_gamma=False):
    """Take a single step with a relaxed explicit Runge-Kutta method for convex
    functionals. It is also possible to apply these schemes to general
    functionals. In that case, some root finding procedures have to be adapted slightly."""

    rkm = rkm.__num__()

    b = rkm.b
    s = len(rkm)
    y = np.zeros((s, np.size(w))) # stage values
    F = np.zeros((s, np.size(w))) # stage derivatives

    for i in range(s):
        y[i,:] = w.copy()
        for j in range(i):
            y[i,:] += rkm.A[i,j]*dt*F[j,:]
        F[i,:] = f(y[i,:])

    if relaxed:
        direction = dt * sum([b[i]*F[i,:] for i in range(s)])
        estimate = dt * sum([b[i]*np.dot(deta(y[i,:]),F[i,:]) for i in range(s)])

        r = lambda gamma: eta(w+gamma*direction) - eta(w) - gamma*estimate
        if debug:
            print('r(1): ', r(1))
        rjac= lambda gamma: np.array([np.dot(deta(w+gamma*direction), direction) - estimate])

        if rjac == False:
            use_jac = False
        else:
            use_jac = rjac

        if method == "newton":
            gam = newton(r, old_gamma, fprime=rjac, tol=tol, maxiter=maxiter)
            success = True
            msg = "Newton method did not converge"
        elif method == "brentq" or method == "bisect":
            # For convex functionals, additional insights are provided: There is exactly one root
            # and r is negative for smaller gamma and positive for bigger gamma. Thus, we can use
#                 left = 0.9 * old_gamma
#                 right = 1.1 * old_gamma
#                 while r(left) > 0:
#                     right = left
#                     left *= 0.5
#                 while r(right) < 0:
#                     left = right
#                     right *= 2.0
            # For general functionals, we might need to use something like:
#                 left = old_gamma - 0.1
#                 right = old_gamma + 0.1
#                 while r(left) * r(right) > 0:
#                     left -= 0.1
#                     right += 0.1
            left = 0.9 * old_gamma
            right = 1.1 * old_gamma
            left_right_iter = 0
            while r(left) * r(right) > 0:
                left *= 0.9
                right *= 1.1
                left_right_iter += 1
                if left_right_iter > 100:
                    raise SolveForGammaException(
                        "No suitable bounds found after %d iterations.\nLeft = %e; r(left) = %e\nRight = %e; r(right) = %e\n"%(
                            left_right_iter, left, r(left), right, r(right)),
                        w)

            if method == "brentq":
                gam = brentq(r, left, right, xtol=tol, maxiter=maxiter)
            else:
                gam = bisect(r, left, right, xtol=tol, maxiter=maxiter)
            success = True
            msg = "%s method did not converge"%method
        else:
            sol = root(r, old_gamma, jac=use_jac, method=method, tol=tol,
                       options={'xtol': tol, 'maxiter': maxiter})
            gam = sol.x; success = sol.success; msg = sol.message

        if not success:
            print('Warning: fsolve did not converge.')
            print(gam)
            print(msg)

        if gam <= 0:
            print('Warning: gamma is negative.')

    else:
        gam = 1.

    w = w + dt*gam*sum([b[i]*F[i] for i in range(s)])

    return w, gam


