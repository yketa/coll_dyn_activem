"""
Module add provides classes to read from data files produced by ADD simulations.
"""

from coll_dyn_activem.read import Dat, _Read

import numpy as np

class ADD(Dat):
    """
    Read data files from simulation.
    """

    def __init__(self, filename, corruption=False):
        """
        Get data from report file.

        Parameters
        ----------
        filename : str
            Path to data file.
        corruption : bool
            Pass corruption test. (default: False)
        """

        super().__init__(
            filename, loadWork=False, corruption='datN' if corruption else None)

        self.report = (lambda r: np.array(
            [[r._read('i'), r._read('i'), r._read('d'), r._read('d'),
                r._read('d'), r._read('d')]
            for i in range(int(r.fileSize/(2*r._bpe('i') + 4*r._bpe('d'))))]))(
            _Read('%s.add' % self.filename))

        self.ep = np.zeros((len(self.report) + 1,))
        for i, dep in enumerate(self.report[:, -2]):
            self.ep[i + 1] = self.ep[i] + dep
