"""
Module rotors provides classes and functions to compute and analyse
orientational dynamics and statistics of interacting Brownian rotors.

(see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#N-interacting%20Brownian%20rotors)
"""

import numpy as np
import scipy.special as special
import scipy.optimize as optimize
import scipy.misc as misc
import scipy.integrate as integrate

from coll_dyn_activem.read import DatR
from coll_dyn_activem.maths import Distribution

############
### DATA ###
############

class Rotors(DatR):
    """
    Compute and analyse orientational dynamics and statistics from simulation
    data.

    (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#N-interacting%20Brownian%20rotors)
    """

    def __init__(self, filename, skip=1):
        """
        Loads file.

        Parameters
        ----------
        filename : string
            Name of input data file.
        skip : int
            Skip the `skip' first computed frames in the following calculations.
            (default: 1)
            NOTE: This can be changed at any time by setting self.skip.
        """

        super().__init__(filename)  # initialise with super class

        self.skip = skip    # skip the `skip' first frames in the analysis

    def nOrder(self, int_max=None, norm=False):
        """
        Returns array of order parameters.

        Parameters
        ----------
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
        norm : bool
            Return norm of order parameter rather than 2D order parameter.
            (default: False)

        Returns
        -------
        nu : [not(norm)] (*, self.N, 2) float numpy array
             [norm] (*, self.N) float numpy array
            Array of order parameters.
        """

        nu = []
        for time0 in self._time0(int_max=int_max):
            nu += [self.getOrderParameter(time0, norm=norm)]
        nu = np.array(nu)

        return nu

    def orderHist(self, nBins, int_max=None, vmin=0, vmax=1, log=False,
        rescaled_to_max=False):
        """
        Returns histogram with `nBins' bins of order parameter norm.

        Parameters
        ----------
        nBins : int
            Number of bins of the histogram.
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
        vmin : float
            Minimum value of the bins. (default: 0)
        vmax : float
            Maximum value of the bins. (default: 1)
        log : bool
            Consider the log of the occupancy of the bins. (default: False)
        rescaled_to_max : bool
            Rescale occupancy of the bins by its maximum over bins.
            (default: False)

        Returns
        -------
        bins : float numpy array
            Values of the bins.
        hist : float numpy array
            Occupancy of the bins.
        """

        return Distribution(self.nOrder(int_max=int_max, norm=True)).hist(
                nBins, vmin=vmin, vmax=vmax, log=log,
                rescaled_to_max=rescaled_to_max)

    def nu_pdf_th(self, *nu):
        """
        Returns value of theoretical probability density function of the order
        parameter norm.

        (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#N-interacting%20Brownian%20rotors)

        Parameters
        ----------
        nu : float
            Values of the order parameter norm at which to evaluate the
            probability density function.

        Returns
        -------
        pdf : (*,) float numpy array
            Probability density function.
        """

        return nu_pdf_th(self.N, self.g, self.Dr, *nu)

    def _time0(self, int_max=None):
        """
        Returns array of frames at which to compute orientations.

        Parameters
        ----------
        int_max : int or None
            Maximum number of frames to consider. (default: None)
            NOTE: If int_max == None, then take the maximum number of frames.
                  WARNING: This can be very big.

        Returns
        -------
        time0 : (*,) int numpy array
            Array of frames.
        """

        if int_max == None: return np.array(range(self.skip, self.frames - 1))
        return np.linspace(
            self.skip, self.frames - 1, int(int_max), endpoint=False, dtype=int)

def nu_pdf_th(N, g, Dr, *nu):
    """
    Returns value of theoretical probability density function of the order
    parameter norm.

    (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#N-interacting%20Brownian%20rotors)

    Parameters
    ----------
    N : int
        Number of rotors.
    g : float
        Aligning torque parameter.
    Dr : float
        Rotational diffusivity.
    nu : float
        Values of the order parameter norm at which to evaluate the
        probability density function.

    Returns
    -------
    pdf : (*,) float numpy array
        Probability density function.
    """

    Z = (1 - np.exp(-N*(1 + g/Dr)))/(2*N*(1 + g/Dr))    # partition function

    return np.array(list(map(
        lambda _nu: _nu*np.exp(-N*(1 + g/Dr)*(_nu**2))/Z,
        nu)))

###############################
### THEORETICAL PREDICTIONS ###
###############################

class Mathieu:
    """
    Provides estimates of the SCGF and the rate function of a single rotor from
    Mathieu functions, as well as optimal control potential for the angle.

    (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#Brownian%20rotors%20LDP)
    (see https://en.wikipedia.org/wiki/Mathieu_function)
    """

    def __init__(self, Dr):
        """
        Defines parameters.

        Parameters
        ----------
        Dr : float
            Rotational diffusivity.
        """

        self.Dr = Dr

        # physical parameters
        self._mathieu_order = 0 # order of Mathieu function

        # numerical parameters
        self.width_inv_search = 5   # width (in units of Dr) of the interval to search when inverting functions
        self.dx = 1e-6              # accuracy for derivative

    def SCGFX(self, *s):
        """
        Returns SCGF of the polarisation along x-axis.

        Parameters
        ----------
        s : float
            Biasing parameter.

        Returns
        -------
        psi : float Numpy array
            Scaled cumulant generating function.
        """

        return np.array(list(map(
            lambda _s: -(self.Dr/4.)*(
                self._mathieu_characteristic_a(_s)),
            s)))

    def pX(self, *s):
        """
        Returns biased average of polarisation along x-axis from derivative
        of SCGF.

        Parameters
        ----------
        s : float
            Biasing parameter.

        Returns
        -------
        px : float Numpy array
            Biased average of polarisation along x-axis.
        """

        return np.array(list(map(
            lambda _s:
                -misc.derivative(lambda _: self.SCGFX(_)[0], _s, dx=self.dx),
            s)))

    def sPX(self, *px):
        """
        Returns biasing parameter which gives biased average of polarisation
        along x-axis.

        Parameters
        ----------
        px : float Numpy array
            Biased average of polarisation along x-axis.

        Returns
        -------
        s : float Numpy array
            Biasing parameter.
        """

        return np.array(list(map(
            lambda _px:
                optimize.root_scalar(
                    lambda _: self.pX(_)[0] - _px,
                    x0=-self.width_inv_search*self.Dr,
                    x1=self.width_inv_search*self.Dr
                ).root,
            px)))

    def SCGF(self, *s):
        """
        Returns SCGF of the vectorial polarisation.

        Parameters
        ----------
        s : float 2-uple
            Biasing parameter.

        Returns
        -------
        psi : float Numpy array
            Scaled cumulant generating function.
        """

        return self.SCGFX(*np.sqrt((np.array(s)**2).sum(axis=-1)))

    def rateX(self, *p):
        """
        Returns rate function of the polarisation along x-axis.

        Parameters
        ----------
        p : float
            Polarisation along x-axis.

        Returns
        -------
        I : float Numpy array
            Rate function.
        """

        return np.array(list(map(
            lambda _p: -np.array(optimize.minimize(
                lambda s: s*_p + self.SCGFX(s)[0],
                0).fun, ndmin=1)[0],
            p)))

    def rate(self, *p):
        """
        Returns rate function of the vectorial polarisation.

        Parameters
        ----------
        p : float 2-uple
            Polarisation.

        Returns
        -------
        I : float Numpy array
            Rate function.
        """

        return np.array(list(map(
            lambda _p: -np.array(optimize.minimize(
                lambda s: np.dot(s, _p) + self.SCGF(s)[0],
                (0, 0))['fun'], ndmin=1)[0],
            p)))

    def optimal_potential(self, s, *theta):
        """
        Returns optimal control potential for biasing parameter `s' (on x-axis).

        Parameters
        ----------
        s : float
            Biasing parameter.
        theta : float
            Angles (in radians) at which to evaluate the potential.

        Returns
        -------
        phi : float Numpy array
            Optimal control potential.
        """

        return (np.array(list(map(
            lambda _theta: -2*np.log(self._mathieu_function(s, _theta)),
            theta)))
            + 2*np.log(self._mathieu_function(s, 0)))   # normalisation

    def optimal_potential_curvature(self, s):
        """
        Returns curvature of optimal control potential at theta = 0 for biasing
        parameter `s' (on x-axis).

        Parameters
        ----------
        s : float
            Biasing parameter.

        Returns
        -------
        k : float
            Curvature at theta = 0.
        """

        return (1./2.)*(
            self._mathieu_characteristic_a(s)
            - 2*self._mathieu_characteristic_q(s))

    def _mathieu_characteristic_q(self, s):
        """
        Returns characteristic value 'q' of the Mathieu function for biasing
        parameter `s'.

        Notation from https://en.wikipedia.org/wiki/Mathieu_function.

        Parameters
        ----------
        s : float
            Biasing parameter.

        Returns
        -------
        q : float
            Characteristic value 'q' of the Mathieu function.
        """

        return (2.*s)/self.Dr

    def _mathieu_characteristic_a(self, s):
        """
        Returns characteristic value 'a' of the Mathieu function for biasing
        parameter `s'.

        Notation from https://en.wikipedia.org/wiki/Mathieu_function.

        Parameters
        ----------
        s : float
            Biasing parameter.

        Returns
        -------
        a : float
            Characteristic value 'a' of the Mathieu function.
        """

        return special.mathieu_a(
            self._mathieu_order,
            self._mathieu_characteristic_q(s))

    def _mathieu_function(self, s, theta):
        """
        Returns Mathieu function evaluated at angle `theta' for biasing
        parameter `s'.

        Notation from https://en.wikipedia.org/wiki/Mathieu_function.

        Parameters
        ----------
        s : float
            Biasing parameter.
        theta : float
            Angle (in radians) at which to evaluate.

        Returns
        -------
        ce : float
            Mathieu function evaluated at `theta'.
        """

        return special.mathieu_cem(
            self._mathieu_order, self._mathieu_characteristic_q(s),
            (180./np.pi)*theta/2.)[0]

class VariationalPolarisationSquared:
    """
    Provides estimates of the SCGF and the rate function for independent
    Brownian rotors biased with respect to their squared polarisation, following
    a variational approach based on the following ansatz for the distribution of
    orientations

        P[\\{\\theta_i\\}] \\propto \\exp(h(s) \\sum_i \\cos\\theta_i)

    which is optimised with respect to the distribution parameter h(s).

    (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#Brownian%20rotors%20LDP)
    """

    def __init__(self, Dr):
        """
        Defines parameters.

        Parameters
        ----------
        Dr : float
            Rotational diffusivity.
        """

        self.Dr = Dr

        # numerical parameters
        self.width_inv_search = 5   # width (in units of Dr) of the interval to search when inverting functions
        self.dx = 1e-6              # accuracy for derivative

    def SCGF(self, *s):
        """
        Returns maximised lower bound to the scaled cumulant generating
        function.

        Parameters
        ----------
        s : float
            Biasing parameter.

        Returns
        -------
        psi : float Numpy array
            Scaled cumulant generating function.
        """

        return np.array(list(map(
            lambda _s: np.array(self._maximise_SCGF_bound(_s).fun, ndmin=1)[0],
            s)))

    def h(self, *s):
        """
        Returns optimised distribution parameter.

        Parameters
        ----------
        s : float
            Biasing parameter.

        Returns
        -------
        h : float Numpy array
            Distribution parameter.
        """

        return np.array(list(map(
            lambda _s: np.array(self._maximise_SCGF_bound(_s).x, ndmin=1)[0],
            s)))

    def p(self, *s):
        """
        Returns estimate of the biased average of the (squared) polarisation
        from the derivative of the bound to the SCGF.

        Parameters
        ----------
        s : float
            Biasing parameter.

        Returns
        -------
        p : float Numpy array
            Biased average of the (squared) polarisation.
        """

        return np.array(list(map(
            lambda _s:
                -misc.derivative(lambda _: self.SCGF(_)[0], _s, dx=self.dx),
            s)))

    def s(self, *p):
        """
        Returns biasing parameter associated to estimate of the biased average
        of the (squared) polarisation from the derivative of the bound to the
        SCGF.

        Parameters
        ----------
        p : float
            (Squared) polarisation.

        Returns
        -------
        s : float Numpu array
            Biasing parameter.
        """

        return np.array(list(map(
            lambda _p:
                optimize.root_scalar(
                    lambda _: self.p(_)[0] - _p,
                    x0=-self.width_inv_search*self.Dr,
                    x1=self.width_inv_search*self.Dr
                ).root,
            p)))

    def rate(self, *p):
        """
        Returns maximised upper bound to the rate function.

        Parameters
        ----------
        p : float
            (Squared) polarisation.

        Returns
        -------
        I : float Numpy array
            Rate function.
        """

        return np.array(list(map(
            lambda _p: -np.array(optimize.minimize_scalar(
                lambda _: self._rate_bound(_p, _)).fun,
                ndmin=1)[0],
            p)))

    def _rate_bound(self, p, x):
        """
        Function which minimum opposite on `x' corresponds to the upper bound
        to the rate function for (squared) polarisation `p' following our
        variational approach.

        Parameters
        ----------
        p : float
            (Squared) polarisation.
        x : float
            Biasing parameter.

        Returns
        -------
        B : float
            Evaluated bound.
        """

        return x*p + self.SCGF(x)[0]

    def _maximise_SCGF_bound(self, s):
        """
        Maximises bound to the SCGF for biasing parameter `s' following our
        variational approach.

        Parameters
        ----------
        s : float
            Biasing parameter.

        Returns
        -------
        opt : scipy.optimize.OptimizeResult
            Optimisation result.
        """

        opt = optimize.minimize_scalar(lambda _: -self._SCGF_bound(s, _))
        opt.fun = -opt.fun      # the opposite of the functon is minimised
        opt.x = np.abs(opt.x)/2 # self._SCGF_bound is a function of 2 h(s) and is furthermore even when biasing wrt the squared polarisation and we are interested in the positive solution

        return opt

    def _SCGF_bound(self, s, x):
        """
        Function which maximum on `x' corresponds to the lower bound to the SCGF
        for biasing parameter `s' following our variational approach.

        Parameters
        ----------
        s : float
            Biasing parameter.
        x : float
            Double of the distribution parameter.

        Returns
        -------
        B : float
            Evaluated bound.
        """

        return (-self.Dr*x/4*special.iv(1.0, x)/special.iv(0.0, x)
            - s*(special.iv(1.0, x)**2)/(special.iv(0.0, x)**2))

class VariationalPolarisation(VariationalPolarisationSquared):
    """
    Provides estimates of the SCGF and the rate function for independent
    Brownian rotors biased with respect to their polarisation, following a
    variational approach based on the following ansatz for the distribution of
    orientations

        P[\\{\\theta_i\\}] \\propto \\exp(h(s) \\sum_i \\cos\\theta_i)

    which is optimised with respect to the distribution parameter h(s).

    (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#Brownian%20rotors%20LDP)
    """

    def g(self, *p):
        """
        Returns torque parameter associated to biased average of polarisation
        `p'.

        Parameters
        ----------
        p : float
            Polarisation.

        Returns
        -------
        g : float
            Torque parameter.
        """

        return np.array(list(map(
            lambda _: -self.Dr*self.h(self.s(_)[0])[0]/(2*_),
            p)))

    def _SCGF_bound(self, s, x):
        """
        Function which maximum on `x' corresponds to the lower bound to the SCGF
        for biasing parameter `s' following our variational approach.

        Parameters
        ----------
        s : float
            Biasing parameter.
        x : float
            Double of the distribution parameter.

        Returns
        -------
        B : float
            Evaluated bound.
        """

        return (-self.Dr*x/4*special.iv(1.0, x)/special.iv(0.0, x)
            - s*(special.iv(1.0, x))/(special.iv(0.0, x)))

class MeanFieldRotors:
    """
    Provides estimates of the average polarisation for a mean-field model of
    rotors with aligning torque.

    (see https://yketa.github.io/DAMTP_MSC_2019_Wiki/#N-interacting%20Brownian%20rotors)
    """

    def __init__(self, Dr):
        """
        Defines parameters.

        Parameters
        ----------
        Dr : float
            Rotational diffusivity.
        """

        self.Dr = Dr

    def nu(self, *g):
        """
        Returns average polarisation for a given torque parameter `g'.

        Parameters
        ----------
        g : float
            Torque parameter.

        Returns
        -------
        p : float Numpy array
            Average polarisation.
        """

        return np.array(list(map(
            lambda _g: optimize.fsolve(
                lambda _: _ - self._polarisation(_, _g), x0=0.5)[0],
            g)))

    def _boltzmann(self, nu, g, theta):
        """
        Returns Boltzmann coefficient for angle `theta' given the polarisation
        `nu' and the torque parameter `g'.

        Parameters
        ----------
        nu : float
            Polarisation.
        g : float
            Torque parameter.
        theta : float
            Angle.

        Returns
        -------
        b : float
            Boltzmann coefficient.
        """

        return np.exp(-2*g*nu*np.cos(theta)/self.Dr)

    def _polarisation(self, nu, g):
        """
        Returns average polarisation given the polarisation `nu' and the
        torque parameter `g'.

        NOTE: This function is meant to be used in order to determine the
              polarisation self-consistently.

        Parameters
        ----------
        nu : float
            Polarisation.
        g : float
            Torque parameter.

        Returns
        -------
        p : float
            Average polarisation.
        """

        Z = integrate.quad(lambda _: self._boltzmann(nu, g, _), 0, 2*np.pi)[0]  # partition function

        return integrate.quad(
            lambda _: np.cos(_)*self._boltzmann(nu, g, _), 0, 2*np.pi)[0]/Z
