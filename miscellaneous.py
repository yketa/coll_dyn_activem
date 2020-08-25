"""
Module miscellaneous contains miscellaneous functions and classes related to
this project.
"""

import numpy as np
import scipy.optimize as optimize

######################
### RTPs ON A RING ###
######################

class RTPring:
    """
    Theoretical results for 2 RTPs on a ring.

    (see https://github.com/yketa/DAMTP_MSC_2019_Wiki/tree/master/Summaries/RTPring)
    """

    def __init__(self):
        """
        Initialises computation parameters.
        """

        self._zero = 1e-16
        self.none = np.nan

        self.lambdaLmin = -20

    def s(self, L, psi):
        """
        Biasing parameter rescaled by the product of persistence length and
        propulsion velocity as a function of the cumulant generating function
        (CGF) scaled by the persistence time.

        Parameters
        ----------
        L : float
            Ring length rescaled by the persistence length.
        psi : float
            CGF rescaled by the persistence time.

        Returns
        -------
        s : float
            Associated biasing parameter rescaled by the product of persistence
            length and propulsion velocity
        """

        if psi == 0: return 0

        k = self._k(psi)

        try:

            s = (psi*(psi + 4))/(2*psi + 4)

            if psi > 0: tan = np.tanh
            if psi < 0: tan = np.tan

            den = psi*(psi + 2)*(psi + 4) + k*((psi + 2)**2)/tan(k*L/2.)
            if den < 0: return self.none

            s += (psi*(psi + 4))/den

            if s < self.lambdaLmin: return self.none
            if s*psi < 0 : return self.none
            return s

        except ZeroDivisionError:

            return self.none

    def nu(self, psi):
        """
        Polarisation of the RTPs.

        Parameters
        ----------
        psi : float
            CGF rescaled by the persistence time.

        Returns
        -------
        nu : float
            Polarisation.
        """

        if psi == 0: return 1./2.

        return 2./(psi + 4)

    def nuAve(self, L, psi):
        """
        Polarisation of the RTPs computed with the full-time distribution.

        Parameters
        ----------
        L : float
            Ring length rescaled by the persistence length.
        psi : float
            CGF rescaled by the persistence time.

        Returns
        -------
        nu : float
            Polarisation.
        """

        if psi == 0: return 1./2.

        k = self._k(psi)
        omega = 2*self._k(psi)/(psi + 2)

        if psi < 0:

            omegasq = -omega**2
            sin = np.sin
            cos = np.cos

        if psi > 0:

            omegasq = omega**2
            sin = np.sinh
            cos = np.cosh

        nu = (
            (2*(omega/k)*(1 + cos(k*L)) + 2*(omegasq/k)*sin(k*L)
                + (4/((psi + 2)**2))*(sin(k*L)/k + L))
            /(2*(omega/k)*(1 + cos(k*L)) + (2 + (psi + 2)**2)*(omegasq/k)*sin(k*L)
                + (psi + 2)*(1 - omegasq) + (psi + 2)*(1 + omegasq)*cos(k*L)
                + (4/((psi + 2)**2) + (1 - omegasq))*sin(k*L)/k
                + (4/((psi + 2)**2) + (1 + omegasq))*L)
        )

        return nu

    def Psi(self, Lambda):
        """
        Rescaled cumulant generating function (CGF) in the scaling regime.

        Parameters
        ----------
        Lambda : float
            Rescaled biasing parameter.

        Returns
        -------
        Psi : float
            Rescaled SCGF.
        """

        if Lambda == 0: return 0

        if Lambda > 0:
            return optimize.brentq(
                lambda _: self._scaling_function(_, Lambda),
                np.abs(self._zero),
                1e4)

        if Lambda < 0:
            return optimize.brentq(
                lambda _: self._scaling_function(_, Lambda),
                -((np.pi**2)/4 - np.abs(self._zero)),
                -np.abs(self._zero))

    def Gamma(self, Lambda):
        """
        Rescaled sticking probability of parallel particles in the scaling
        regime.

        Parameters
        ----------
        Lambda : float
            Rescaled biasing parameter.

        Returns
        -------
        Gamma : float
            Rescaled sticking probability.
        """

        if Lambda == 0: return 1    # wild guess

        return self.Psi(Lambda)/Lambda

    def LEpsilon(self, Lambda, *rL):
        """
        Product of ring length and regular part of the probability density
        function in the scaling regime.

        Parameters
        ----------
        Lambda : float
            Resclaed biasing parameter.
        rL : float
            Ratio of position over ring length.

        Returns
        -------
        Epsilon : float Numpy array
            Probability density function.
        """

        rL = np.array(rL)

        prefactor = (1./4.)*self.Gamma(Lambda)

        if Lambda == 0: return np.full(rL.shape, fill_value=prefactor)

        psi = self.Psi(Lambda)

        if Lambda > 0:
            return (prefactor*np.cosh(np.sqrt(psi)*(1 - 2*rL))
                /np.cosh(np.sqrt(psi)))
        if Lambda < 0:
            return (prefactor*np.cos(np.sqrt(-psi)*(1 - 2*rL))
                /np.cos(np.sqrt(-psi)))

    def _scaling_function(self, Psi, Lambda):
        """
        Function which root for a given `Lambda` gives the corresponding `Psi`.

        Parameters
        ----------
        Psi : float
            Rescaled cumulant generating function (CGF).
        Lambda : float
            Rescaled biasing parameter.

        Returns
        -------
        scale : float
            Scaling function.
        """

        if Lambda == 0: return 0

        if Lambda > 0:
            return (
                1./(np.tanh(np.sqrt(np.abs(Psi)))*np.sqrt(np.abs(Psi)))
                - 1./Lambda)
        if Lambda < 0:
            return (
                1./(np.tan(np.sqrt(np.abs(Psi)))*np.sqrt(np.abs(Psi)))
                + 1./Lambda)

    def _k(self, psi):
        """
        Modulus of the scaled exponential length scale.

        Parameters
        ----------
        psi : float
            CGF rescaled by the persistence time.

        Returns
        -------
        k : float
            Scaled exponential length scale.
        """

        return np.sqrt(np.abs(psi*(psi/4. + 1)))
