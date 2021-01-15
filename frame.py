"""
Module frame renders images of the 2D system.

(modified from
https://github.com/yketa/active_particles/tree/master/analysis/frame.py)
"""

from coll_dyn_activem.init import get_env, mkdir
from coll_dyn_activem.read import Dat
from coll_dyn_activem.maths import normalise1D, amplogwidth, cooperativity,\
    relative_positions
from coll_dyn_activem.flow import Displacements
from coll_dyn_activem.structure import Positions
from coll_dyn_activem.force import Force

from os import getcwd
from os import environ as envvar
from os.path import join as joinpath

import sys

from math import ceil

import numpy as np
np.seterr(divide='ignore')

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as ColorsNormalise
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

from datetime import datetime

from collections import OrderedDict

import subprocess

# DEFAULT VARIABLES

_frame_per = 1      # default frame rendering period
_frame_max = 1000   # default maximum number of frames

_frame_ver = 12 # default vertical size of the frames (in inches)
_frame_hor = 16 # default horizontal size of the frames (in inches)
_frame_def = 80 # default definition of images (in dots per inches (dpi))

_arrow_width = 1e-3                         # default width of the arrows
_arrow_head_width = _arrow_width*3e2        # default width of the arrows' head
_arrow_head_length = _arrow_head_width*1.5  # default length of the arrows' head

_font_size = 15 # font size

_colormap_label_pad = 30    # default separation between label and colormap

# FUNCTIONS AND CLASSES

class _Frame:
    """
    This class is designed as the superclass of all other plotting classes
    specific to each mode. It initialises the figure and provides methods to
    plot circles representing particles and arrows at the particles' positions,
    and to add a colorbar.
    """

    def __init__(self, dat, frame, box_size, centre,
        arrow_width=_arrow_width,
        arrow_head_width=_arrow_head_width,
        arrow_head_length=_arrow_head_length,
        WCA_diameter=False, **kwargs):
        """
        Initialises figure.

        Parameters
        ----------
        dat : coll_dyn_activem.read.Dat
            Data object.
        frame : int
            Frame to render.
        box_size : float
            Length of the square box to render.
        centre : 2-uple like
            Centre of the box to render.
        arrow_width : float
            Width of the arrows.
        arrow_head_width : float
            Width of the arrows' head.
        arrow_head_length : float
            Length of the arrows' head.
        WCA_diameter : bool
            Multiply diameters by 2^(1/6). (default: False)

        Optional keyword parameters
        ---------------------------
        remove_cm : int or None
            Remove centre of mass displacement from frame.
            NOTE: if remove_cm == None, nothing is done.
        """

        self.fig, self.ax = plt.subplots()
        self.box_size = box_size
        self.ax.set_xlim([-self.box_size/2, self.box_size/2])
        self.ax.set_xlabel(r'$x$')
        self.ax.set_ylim([-self.box_size/2, self.box_size/2])
        self.ax.set_ylabel(r'$y$')
        self.ax.set_aspect('equal')
        self.ax.tick_params(axis='both', which='both', direction='in',
            bottom=True, top=True, left=True, right=True)
        self.fig.subplots_adjust(top=0.80)

        self.dat = dat
        self.frame = frame
        self.positions = self.dat.getPositions(frame)
        if 'remove_cm' in kwargs:
            if kwargs['remove_cm'] != None:
                self.positions -= self.dat.getDisplacements(
                    kwargs['remove_cm'], frame).mean(axis=0, keepdims=True)
                self.positions -= (self.positions//self.dat.L)*self.dat.L
        self.positions = relative_positions(self.positions, centre, self.dat.L) # particles' positions at frame frame with centre as centre of frame
        self.diameters = self.dat.diameters*(2**(1./6.) if WCA_diameter else 1) # particles' diameters

        self.particles = [particle for particle in range(len(self.positions))
            if (np.abs(self.positions[particle]) <= self.box_size/2
                + self.diameters.reshape((self.dat.N, 1))/2).all()] # particles inside box of centre centre and length box_size

        self.arrow_width = arrow_width
        self.arrow_head_width = arrow_head_width
        self.arrow_head_length = arrow_head_length

    def __del__(self):
        """
        Closes figure.
        """

        plt.close(self.fig)

    def draw_circle(self, particle, color='black', fill=False, border=None,
        label=False):
        """
        Draws circle at particle's position with particle's diameter.

        Parameters
        ----------
        particle : int
            Particle index.
        color : any matplotlib color
            Circle color. (default: 'black')
        fill : bool
            Filling the circle with same color. (default: False)
        border : string or None
            Circle bordel color. (default: None)
            NOTE: if border == None, do not add border.
        label : bool
            Write indexes of particles in circles. (default: False)
        """

        def _draw_circle(position):
            """
            Draw circle at position.

            Parameters
            ----------
            position : (2,) float array-like
                Position of circle to draw.
            """

            circle = plt.Circle(position,
                self.diameters[particle]/2, color=color, fill=fill,
                zorder=0)       # circle artist representing particle
            self.ax.add_artist(circle)
            if border != None:  # add black border
                circleBorder = plt.Circle(position,
                    self.diameters[particle]/2, color=border, fill=False,
                    zorder=1)
                self.ax.add_artist(circleBorder)

        _draw_circle(self.positions[particle])

        for dim in range(2):
            if (np.abs(self.positions[particle][dim]) >
                self.dat.L/2 - self.diameters[particle]/2):
                newPosition = self.positions[particle].copy()
                newPosition[dim] -= (np.sign(self.positions[particle][dim])
                    *self.dat.L)
                _draw_circle(newPosition)
        if (np.abs(self.positions[particle]) >
            self.dat.L/2 - self.diameters[particle]/2).all():
            newPosition = self.positions[particle].copy()
            newPosition -= (np.sign(self.positions[particle])
                *self.dat.L)
            _draw_circle(newPosition)

        if label:
            self.ax.annotate(
                "%i" % particle, xy=self.positions[particle], ha="center")

    def draw_arrow(self, particle, dx, dy, color='black'):
        """
        Draws arrow starting from particle's position.

        Parameters
        ----------
        particle : int
            Particle index.
        dx : float
            Arrow length in x-direction.
        dy : float
            Arrow length in y-direction.
        color : any matplotlib color
            Arrow color. (default: 'black')
        """

        length = np.sqrt(dx**2 + dy**2) # length of arrow
        if length == 0: return
        self.ax.arrow(*self.positions[particle], dx, dy, color=color,
            width=length*self.arrow_width,
            head_width=length*self.arrow_head_width,
            head_length=length*self.arrow_head_length, zorder=1)

    def colorbar(self, vmin, vmax, cmap=plt.cm.jet):
        """
        Adds colorbar to plot.

        Parameters
        ----------
        vmin : float
            Minimum value of the colorbar.
        vmax : float
            Maximum value of the colorbar.
        cmap : matplotlib colorbar
            Matplotlib colorbar to be used. (default: matplotlib.pyplot.cm.jet)
        """

        vNorm = ColorsNormalise(vmin=vmin, vmax=vmax)
        self.scalarMap = ScalarMappable(norm=vNorm, cmap=cmap)

        self.colormap_ax = make_axes_locatable(self.ax).append_axes('right',
            size='5%', pad=0.05)
        self.colormap = mpl.colorbar.ColorbarBase(self.colormap_ax, cmap=cmap,
            norm=vNorm, orientation='vertical')

    def colorbar_discrete(self, nColors, vmin, vmax, cmap=plt.cm.jet):
        """
        Adds discrete colorbar to plot.

        Parameters
        ----------
        nColors : int
            Number of values between minimum and maximum (included).
        vmin : float
            Minimum value of the colorbar.
        vmax : float
            Maximum value of the colorbar.
        cmap : matplotlib colorbar
            Matplotlib colorbar to be used. (default: matplotlib.pyplot.cm.jet)
        """

        vNorm = BoundaryNorm(np.linspace(vmin, vmax, nColors + 1), nColors)
        _cmap = ListedColormap(
            [ScalarMappable(
                norm=ColorsNormalise(vmin=vmin, vmax=vmax), cmap=cmap
                ).to_rgba(x)
                for x in np.linspace(vmin, vmax, nColors)])
        self.scalarMap = ScalarMappable(norm=vNorm, cmap=_cmap)

        self.colormap_ax = make_axes_locatable(self.ax).append_axes('right',
            size='5%', pad=0.05)
        self.colormap = mpl.colorbar.ColorbarBase(self.colormap_ax, cmap=_cmap,
            norm=vNorm, orientation='vertical')
        self.colormap.set_ticks(
            [(0.5 + i)*(vmax - vmin)/nColors for i in range(nColors)])
        self.colormap.set_ticklabels(
            [r'$%s$' % i for i in np.linspace(vmin, vmax, nColors)])

class Orientation(_Frame):
    """
    Plotting class specific to 'orientation' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        arrow_width=_arrow_width,
        arrow_head_width=_arrow_head_width,
        arrow_head_length=_arrow_head_length,
        pad=_colormap_label_pad,
        label=False, **kwargs):
        """
        Initialises and plots figure.

        Parameters
        ----------
        dat : coll_dyn_activem.read.Dat
            Data object.
        frame : int
            Frame to render.
        box_size : float
            Length of the square box to render.
        centre : 2-uple like
            Centre of the box to render.
        arrow_width : float
            Width of the arrows.
        arrow_head_width : float
            Width of the arrows' head.
        arrow_head_length : float
            Length of the arrows' head.
        pad : float
            Separation between label and colormap.
            (default: coll_dyn_activem.frame._colormap_label_pad)
        label : bool
            Write indexes of particles in circles. (default: False)

        Optional keyword parameters
        ---------------------------
        (see coll_dyn_activem.frame._Frame)
        """

        super().__init__(dat, frame, box_size, centre,
            arrow_width=arrow_width,
            arrow_head_width=arrow_head_width,
            arrow_head_length=arrow_head_length,
            **kwargs)   # initialise superclass

        self.orientations = (
            dat.getOrientations(frame, *self.particles)%(2*np.pi))  # particles' orientations at frame

        self.colorbar(0, 2, cmap=plt.cm.hsv)                                    # add colorbar to figure
        self.colormap.set_label(r'$\theta_i/\pi$', labelpad=pad, rotation=270)  # colorbar legend

        self.label = label  # write labels

        self.draw()

    def draw_arrow(self, particle, orientation, color='black'):
        """
        Draws arrow in particle.

        Parameters
        ----------
        particle : int
            Particle index.
        orientation : float
            Orientation of particle in radians.
        color : any matplotlib color
            Arrow color. (default: 'black')
        """

        direction = np.array([np.cos(orientation), np.sin(orientation)])
        length = self.diameters[particle]*0.75
        self.ax.arrow(
            *(self.positions[particle] - direction*length/(2*np.sqrt(2))),
            *direction*length/np.sqrt(2),
            color=color,
            width=length*self.arrow_width,
            head_width=length*self.arrow_head_width,
            head_length=length*self.arrow_head_length, zorder=1,
            length_includes_head=True)

    def draw(self):
        """
        Plots figure.
        """

        for particle, orientation in zip(self.particles,
            self.orientations):                                     # for particle and particle's displacement in rendered box
            self.draw_circle(particle,
                color=self.scalarMap.to_rgba(orientation/np.pi), fill=False,
                label=self.label)                                   # draw particle circle with color corresponding to displacement amplitude
            self.draw_arrow(particle, orientation,
                color=self.scalarMap.to_rgba(orientation/np.pi))    # draw displacement direction arrow

class Displacement(_Frame):
    """
    Plotting class specific to 'displacement' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        arrow_width=_arrow_width,
        arrow_head_width=_arrow_head_width,
        arrow_head_length=_arrow_head_length,
        pad=_colormap_label_pad, dt=1, jump=1,
        rescale=False,
        label=False, **kwargs):
        """
        Initialises and plots figure.

        Parameters
        ----------
        dat : coll_dyn_activem.read.Dat
            Data object.
        frame : int
            Frame to render.
        box_size : float
            Length of the square box to render.
        centre : 2-uple like
            Centre of the box to render.
        arrow_width : float
            Width of the arrows.
        arrow_head_width : float
            Width of the arrows' head.
        arrow_head_length : float
            Length of the arrows' head.
        pad : float
            Separation between label and colormap.
            (default: coll_dyn_activem.frame._colormap_label_pad)
        dt : int
            Lag time for displacement. (default: 1)
        jump : int
            Period in number of frames at which to check if particles have
            crossed any boundary. (default: 1)
            NOTE: `jump' must be chosen so that particles do not move a distance
                  greater than half the box size during this time.
        rescale : bool
            Rescale displacements by mean squared displacement. (default: False)
        label : bool
            Write indexes of particles in circles. (default: False)

        Optional keyword parameters
        ---------------------------
        (see coll_dyn_activem.frame._Frame)
        vmin : float
            Minimum value of the colorbar.
        vmax : float
            Maximum value of the colorbar.
        """

        super().__init__(dat, frame, box_size, centre,
            arrow_width=arrow_width,
            arrow_head_width=arrow_head_width,
            arrow_head_length=arrow_head_length,
            **kwargs)   # initialise superclass

        self.zetad = cooperativity(
            dat.getDisplacements(frame, frame + dt, jump=jump, remove_cm=False))
        self.displacements = dat.getDisplacements(
            frame, frame + dt, *self.particles, jump=jump, remove_cm=True)  # particles' displacements at frame
        self.rescale = rescale
        if self.rescale:
            self.displacements /= (
                np.sqrt((self.displacements**2).sum(axis=-1).mean()))

        self.vmin, self.vmax = amplogwidth(self.displacements)
        try:
            self.vmin = np.log10(kwargs['vmin'])
        except (KeyError, AttributeError, TypeError): pass  # 'vmin' not in keyword arguments or None
        try:
            self.vmax = np.log10(kwargs['vmax'])
        except (KeyError, AttributeError, TypeError): pass  # 'vmax' not in keyword arguments or None

        self.colorbar(self.vmin, self.vmax) # add colorbar to figure
        self.colormap.set_label(            # colorbar legend
            # r'$\log_{10} ||\delta \Delta \vec{r}_i(t, t + \Delta t)||$'
            #     + (r'$/\sqrt{\left<|\ldots|^2\right>_i}$'
            #         if self.rescale else '')
            #     + ' ' + r'$(\zeta_{\Delta \vec{r}} = %.4f)$' % self.zetad,
            r'$\log_{10} ||\delta \Delta \boldsymbol{r}_i(t, t + \Delta t)||$'
                + (r'$/\sqrt{\left<|\ldots|^2\right>_i}$'
                    if self.rescale else '')
                + ' '
                + r'$(\zeta_{\Delta \boldsymbol{r}} = %.4f)$' % self.zetad,
            labelpad=pad, rotation=270)

        self.label = label  # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, displacement in zip(
            self.particles, self.displacements):                            # for particle and particle's displacement in rendered box
            self.draw_circle(particle,
                color=self.scalarMap.to_rgba(
                    np.log10(np.linalg.norm(displacement))),
                fill=True,
                label=self.label)                                           # draw particle circle with color corresponding to displacement amplitude
            self.draw_arrow(particle,
                *normalise1D(displacement)*0.75*self.diameters[particle])   # draw displacement direction arrow

class Movement(_Frame):
    """
    Plotting class specific to 'movement' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        arrow_width=_arrow_width,
        arrow_head_width=_arrow_head_width,
        arrow_head_length=_arrow_head_length,
        dt=1, jump=1,
        arrow_factor=1,
        label=False, **kwargs):
        """
        Initialises and plots figure.

        Parameters
        ----------
        dat : coll_dyn_activem.read.Dat
            Data object.
        frame : int
            Frame to render.
        box_size : float
            Length of the square box to render.
        centre : 2-uple like
            Centre of the box to render.
        arrow_width : float
            Width of the arrows.
        arrow_head_width : float
            Width of the arrows' head.
        arrow_head_length : float
            Length of the arrows' head.
        dt : int
            Lag time for displacement. (default: 1)
        jump : int
            Period in number of frames at which to check if particles have
            crossed any boundary. (default: 1)
            NOTE: `jump' must be chosen so that particles do not move a distance
                  greater than half the box size during this time.
        arrow_factor : float
            Displacement arrow dilatation factor. (default: 1)
        label : bool
            Write indexes of particles in circles. (default: False)

        Optional keyword parameters
        ---------------------------
        (see coll_dyn_activem.frame._Frame)
        vmin : float
            Minimum value of the colorbar.
        vmax : float
            Maximum value of the colorbar.
        """

        super().__init__(dat, frame, box_size, centre,
            arrow_width=arrow_width,
            arrow_head_width=arrow_head_width,
            arrow_head_length=arrow_head_length,
            **kwargs)   # initialise superclass

        self.arrow_factor = arrow_factor    # displacement arrow dilatation factor
        self.fig.text(1.03, 0,
            r'$\mathrm{factor} = %.2e$' % self.arrow_factor,
            rotation=270, transform=self.ax.transAxes)

        self.displacements = dat.getDisplacements(
            frame, frame + dt, *self.particles, jump=jump, remove_cm=True)  # particles' displacements at frame

        self.label = label  # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, displacement in zip(
            self.particles, self.displacements):                            # for particle and particle's displacement in rendered box
            self.draw_arrow(particle, *(self.arrow_factor*displacement))    # draw displacement arrow

class Overlap(_Frame):
    """
    Plotting class specific to 'overlap' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        pad=_colormap_label_pad, dt=1, jump=1, a=0.3,
        label=False, **kwargs):
        """
        Initialises and plots figure.

        Parameters
        ----------
        dat : coll_dyn_activem.read.Dat
            Data object.
        frame : int
            Frame to render.
        box_size : float
            Length of the square box to render.
        centre : 2-uple like
            Centre of the box to render.
        pad : float
            Separation between label and colormap.
            (default: coll_dyn_activem.frame._colormap_label_pad)
        dt : int
            Lag time for displacement. (default: 1)
        jump : int
            Period in number of frames at which to check if particles have
            crossed any boundary. (default: 1)
            NOTE: `jump' must be chosen so that particles do not move a distance
                  greater than half the box size during this time.
        a : float
            Length scale to compute overlaps. (default: 1)
        label : bool
            Write indexes of particles in circles. (default: False)

        Optional keyword parameters
        ---------------------------
        (see coll_dyn_activem.frame._Frame)
        """

        super().__init__(dat, frame, box_size, centre,
            **kwargs)   # initialise superclass

        self.a = a
        self.overlaps = np.exp(
            -(dat.getDisplacements(
                frame, frame + dt, *self.particles, jump=jump, remove_cm=True,
                norm=True)/self.a)**2)  # particles' overlaps between frame and frame + dt

        self.colorbar(0, 1, cmap=plt.cm.jet.reversed())# add colorbar to figure
        self.colormap.set_label(                            # colorbar legend
            # (r'$\exp(-[$'
            #     + r'$||\delta \Delta \vec{r}_i(t, t + \Delta t)||/a]^2)$'
            #     + r'$,$' + ' ' + r'$a=%.2f$' % self.a),
            (r'$\exp(-[$'
                + r'$||\delta \Delta \boldsymbol{r}_i(t, t + \Delta t)||/a]^2)$'
                + r'$,$' + ' ' + r'$a=%.2f$' % self.a),
            labelpad=pad, rotation=270)

        self.label = label  # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, overlap in zip(
            self.particles, self.overlaps):                                 # for particle and particle's overlap in rendered box
            self.draw_circle(particle,
                color=self.scalarMap.to_rgba(overlap),
                fill=True,
                label=self.label)                                           # draw particle circle with color corresponding to overlap

class Bond(_Frame):
    """
    Plotting class specific to 'bond' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        pad=_colormap_label_pad, dt=1, a1=1.15, a2=1.5,
        label=False, **kwargs):
        """
        Initialises and plots figure.

        Parameters
        ----------
        dat : coll_dyn_activem.read.Dat
            Data object.
        frame : int
            Frame to render.
        box_size : float
            Length of the square box to render.
        centre : 2-uple like
            Centre of the box to render.
        pad : float
            Separation between label and colormap.
            (default: coll_dyn_activem.frame._colormap_label_pad)
        dt : int
            Lag time for displacement. (default: 1)
        a1 : float
            Distance relative to their diameters below which particles are
            considered bonded. (default: 1.15)
        a2 : float
            Distance relative to their diameters above which particles are
            considered unbonded. (default: 1.5)
        label : bool
            Write indexes of particles in circles. (default: False)

        Optional keyword parameters
        ---------------------------
        (see coll_dyn_activem.frame._Frame)
        """

        super().__init__(dat, frame, box_size, centre,
            **kwargs)   # initialise superclass

        self.a1, self.a2 = a1, a2
        self.brokenBonds = Displacements(
            dat.filename, corruption=dat.corruption).brokenBonds(
                frame, frame + dt, A1=self.a1, A2=self.a2)[0]   # particles' number of broken bonds between frame and frame + dt

        self.colorbar_discrete(7, 0, 6) # add colorbar to figure
        self.colormap.set_label(        # colorbar legend
            # (r'$B_i(t, t + \Delta t, A_1=%.2f, A_2=%.2f)$'
            #     % (self.a1, self.a2))
            (r'$\mathcal{B}_i(t, t + \Delta t, A_1=%.2f, A_2=%.2f)$'
                % (self.a1, self.a2)),
            labelpad=pad, rotation=270)

        self.label = label  # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, bond in zip(
            self.particles, self.brokenBonds):  # for particle and particle's number of broken bonds in rendered box
            self.draw_circle(particle,
                color=self.scalarMap.to_rgba(bond),
                fill=True,
                label=self.label)               # draw particle circle with color corresponding to number of broken bond

class D2min(_Frame):
    """
    Plotting class specific to 'd2min' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        pad=_colormap_label_pad, dt=1, a1=1.15,
        label=False, **kwargs):
        """
        Initialises and plots figure.

        Parameters
        ----------
        dat : coll_dyn_activem.read.Dat
            Data object.
        frame : int
            Frame to render.
        box_size : float
            Length of the square box to render.
        centre : 2-uple like
            Centre of the box to render.
        pad : float
            Separation between label and colormap.
            (default: coll_dyn_activem.frame._colormap_label_pad)
        dt : int
            Lag time for displacement. (default: 1)
        a1 : float
            Distance relative to their diameters below which particles are
            considered bonded. (default: 1.15)
        label : bool
            Write indexes of particles in circles. (default: False)

        Optional keyword parameters
        ---------------------------
        (see coll_dyn_activem.frame._Frame)
        vmin : float
            Minimum value of the colorbar.
        vmax : float
            Maximum value of the colorbar.
        """

        super().__init__(dat, frame, box_size, centre,
            **kwargs)   # initialise superclass

        self.a1 = a1
        self.d2min = Displacements(
            dat.filename, corruption=dat.corruption).d2min(
                frame, frame + dt, A1=self.a1)  # particles' nonaffine squared displacements between frame and frame + dt

        self.vmin, self.vmax = amplogwidth(np.reshape(self.d2min, (dat.N, 1)))
        try:
            self.vmin = np.log10(kwargs['vmin'])
        except (KeyError, AttributeError, TypeError): pass  # 'vmin' not in keyword arguments or None
        try:
            self.vmax = np.log10(kwargs['vmax'])
        except (KeyError, AttributeError, TypeError): pass  # 'vmax' not in keyword arguments or None

        self.colorbar(self.vmin, self.vmax, cmap=plt.cm.get_cmap('Greys'))  # add colorbar to figure
        self.colormap.set_label(                                            # colorbar legend
            r'$\log_{10} D^2_{i,\mathrm{min}}(t, t + \Delta t, A_1=%.2f)$'
                % self.a1,
            labelpad=pad, rotation=270)

        self.label = label  # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, d2min in zip(
            self.particles, self.d2min):    # for particle and particle's nonaffine squared displacement in rendered box
            self.draw_circle(particle,
                color=self.scalarMap.to_rgba(np.log10(d2min)),
                border='black',
                fill=True,
                label=self.label)           # draw particle circle with color corresponding to D2min

class Velocity(_Frame):
    """
    Plotting class specific to 'velocity' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        arrow_width=_arrow_width,
        arrow_head_width=_arrow_head_width,
        arrow_head_length=_arrow_head_length,
        pad=_colormap_label_pad,
        label=False, **kwargs):
        """
        Initialises and plots figure.

        Parameters
        ----------
        dat : coll_dyn_activem.read.Dat
            Data object.
        frame : int
            Frame to render.
        box_size : float
            Length of the square box to render.
        centre : 2-uple like
            Centre of the box to render.
        arrow_width : float
            Width of the arrows.
        arrow_head_width : float
            Width of the arrows' head.
        arrow_head_length : float
            Length of the arrows' head.
        pad : float
            Separation between label and colormap.
            (default: coll_dyn_activem.frame._colormap_label_pad)
        label : bool
            Write indexes of particles in circles. (default: False)

        Optional keyword parameters
        ---------------------------
        (see coll_dyn_activem.frame._Frame)
        vmin : float
            Minimum value of the colorbar.
        vmax : float
            Maximum value of the colorbar.
        """

        super().__init__(dat, frame, box_size, centre,
            arrow_width=arrow_width,
            arrow_head_width=arrow_head_width,
            arrow_head_length=arrow_head_length,
            **kwargs)   # initialise superclass

        self.zetav = cooperativity(dat.getVelocities(
            frame, remove_cm=False))
        self.velocities = dat.getVelocities(
            frame, *self.particles, remove_cm=True) # particles' velocities at frame

        self.vmin, self.vmax = amplogwidth(self.velocities)
        try:
            self.vmin = np.log10(kwargs['vmin'])
        except (KeyError, AttributeError, TypeError): pass  # 'vmin' not in keyword arguments or None
        try:
            self.vmax = np.log10(kwargs['vmax'])
        except (KeyError, AttributeError, TypeError): pass  # 'vmax' not in keyword arguments or None

        self.colorbar(self.vmin, self.vmax) # add colorbar to figure
        self.colormap.set_label(            # colorbar
            # r'$\log_{10}||\delta \vec{v}_i(t)||$' + ' '
            #     + r'$(\zeta_{\vec{v}} = %.4f)$' % self.zetav,
            r'$\log_{10}||\delta \boldsymbol{v}_i(t)||$' + ' '
                + r'$(\zeta_{\boldsymbol{v}} = %.4f)$' % self.zetav,
            labelpad=pad, rotation=270)

        self.label = label  # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, velocity in zip(
            self.particles, self.velocities):                           # for particle and particle's velocity in rendered box
            self.draw_circle(particle,
                color=self.scalarMap.to_rgba(
                    np.log10(np.linalg.norm(velocity))),
                fill=True,
                label=self.label)                                       # draw particle circle with color corresponding to velocity amplitude
            self.draw_arrow(particle,
                *normalise1D(velocity)*0.75*self.diameters[particle])   # draw velocity direction arrow

class Kinetic(_Frame):
    """
    Plotting class specific to 'kinetic' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        pad=_colormap_label_pad,
        label=False, **kwargs):
        """
        Initialises and plots figure.

        Parameters
        ----------
        dat : coll_dyn_activem.read.Dat
            Data object.
        frame : int
            Frame to render.
        box_size : float
            Length of the square box to render.
        centre : 2-uple like
            Centre of the box to render.
        pad : float
            Separation between label and colormap.
            (default: coll_dyn_activem.frame._colormap_label_pad)
        label : bool
            Write indexes of particles in circles. (default: False)

        Optional keyword parameters
        ---------------------------
        (see coll_dyn_activem.frame._Frame)
        vmin : float
            Minimum value of the colorbar.
        vmax : float
            Maximum value of the colorbar.
        """

        super().__init__(dat, frame, box_size, centre,
            **kwargs)   # initialise superclass

        self.mke = (dat.getVelocities(
            frame, norm=True, remove_cm=True)**2).mean()
        self.kineticE = dat.getVelocities(
            frame, *self.particles, norm=True, remove_cm=True)**2   # particles' kinetic energies at frame

        self.vmin, self.vmax = self.kineticE.min(), self.kineticE.max()
        try:
            if kwargs['vmin'] != None: self.vmin = kwargs['vmin']
        except (KeyError, AttributeError, TypeError): pass  # 'vmin' not in keyword arguments
        try:
            if kwargs['vmax'] != None: self.vmax = kwargs['vmax']
        except (KeyError, AttributeError, TypeError): pass  # 'vmax' not in keyword arguments

        self.colorbar(self.vmin, self.vmax) # add colorbar to figure
        self.colormap.set_label(            # colorbar
            # r'$||\delta \vec{v}_i(t)||^2$' + ' '
            #     + (r'$(\left<||\delta \vec{v}_i(t)||^2\right>_i = %.4e)$'
            #         % self.mke),
            r'$||\delta \boldsymbol{v}_i(t)||^2$' + ' '
                + (r'$(\left<||\delta \boldsymbol{v}_i(t)||^2\right>_i = %.4e)$'
                    % self.mke),
            labelpad=pad, rotation=270)

        self.label = label  # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, kineticE in zip(
            self.particles, self.kineticE):                             # for particle and particle's kinetic energy in rendered box
            self.draw_circle(particle,
                color=self.scalarMap.to_rgba(kineticE),
                fill=True,
                label=self.label)                                       # draw particle circle with color corresponding to kinetic energy

class Interaction(_Frame):
    """
    Plotting class specific to 'interaction' mode.

    Uses the velocity as a proxy to the force (exact if there is no
    translational noise.)
    """

    def __init__(self, dat, frame, box_size, centre,
        arrow_width=_arrow_width,
        arrow_head_width=_arrow_head_width,
        arrow_head_length=_arrow_head_length,
        pad=_colormap_label_pad,
        label=False, **kwargs):
        """
        Initialises and plots figure.

        Parameters
        ----------
        dat : coll_dyn_activem.read.Dat
            Data object.
        frame : int
            Frame to render.
        box_size : float
            Length of the square box to render.
        centre : 2-uple like
            Centre of the box to render.
        arrow_width : float
            Width of the arrows.
        arrow_head_width : float
            Width of the arrows' head.
        arrow_head_length : float
            Length of the arrows' head.
        pad : float
            Separation between label and colormap.
            (default: coll_dyn_activem.frame._colormap_label_pad)
        label : bool
            Write indexes of particles in circles. (default: False)

        Optional keyword parameters
        ---------------------------
        (see coll_dyn_activem.frame._Frame)
        vmin : float
            Minimum value of the colorbar.
        vmax : float
            Maximum value of the colorbar.
        """

        super().__init__(dat, frame, box_size, centre,
            arrow_width=arrow_width,
            arrow_head_width=arrow_head_width,
            arrow_head_length=arrow_head_length,
            **kwargs)   # initialise superclass

        self.forces = Force(dat.filename,
            from_velocity=True, corruption=dat.corruption).getForce(
                frame, *self.particles) # particles' forces at frame

        self.vmin, self.vmax = amplogwidth(self.forces)
        try:
            self.vmin = np.log10(kwargs['vmin'])
        except (KeyError, AttributeError, TypeError): pass  # 'vmin' not in keyword arguments or None
        try:
            self.vmax = np.log10(kwargs['vmax'])
        except (KeyError, AttributeError, TypeError): pass  # 'vmax' not in keyword arguments or None

        self.colorbar(self.vmin, self.vmax) # add colorbar to figure
        self.colormap.set_label(            # colorbar
            # r'$\log_{10}||\vec{F}_i(t)||$',
            r'$\log_{10}||\boldsymbol{F}_i(t)||$',
            labelpad=pad, rotation=270)

        self.label = label  # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, force in zip(
            self.particles, self.forces):                           # for particle and particle's force in rendered box
            self.draw_circle(particle,
                color=self.scalarMap.to_rgba(
                    np.log10(np.linalg.norm(force))),
                fill=True,
                label=self.label)                                   # draw particle circle with color corresponding to velocity amplitude
            self.draw_arrow(particle,
                *normalise1D(force)*0.75*self.diameters[particle])  # draw velocity direction arrow

class Order(_Frame):
    """
    Plotting class specific to 'order' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        arrow_width=_arrow_width,
        arrow_head_width=_arrow_head_width,
        arrow_head_length=_arrow_head_length,
        pad=_colormap_label_pad,
        label=False, **kwargs):
        """
        Initialises and plots figure.

        Parameters
        ----------
        dat : coll_dyn_activem.read.Dat
            Data object.
        frame : int
            Frame to render.
        box_size : float
            Length of the square box to render.
        centre : 2-uple like
            Centre of the box to render.
        pad : float
            Separation between label and colormap.
            (default: coll_dyn_activem.frame._colormap_label_pad)
        label : bool
            Write indexes of particles in circles. (default: False)

        Optional keyword parameters
        ---------------------------
        (see coll_dyn_activem.frame._Frame)
        """

        super().__init__(dat, frame, box_size, centre,
            **kwargs)   # initialise superclass

        self.bondOrder = np.abs(
            Positions(
                dat.filename, corruption=dat.corruption).getBondOrderParameter(
                    frame, *self.particles))

        self.colorbar(0, 1, cmap=plt.cm.inferno)    # add colorbar to figure
        self.colormap.set_label(                    # colorbar legend
            r'$|\psi_{6,i}|$' + ' '
                + r'$(\left<|\psi_{6,i}|\right>=%.3f)$' % self.bondOrder.mean(),
            labelpad=pad, rotation=270)

        self.label = label  # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, order in zip(self.particles, self.bondOrder): # for particle and particle's bond orientation order parameter norm in rendered box
            self.draw_circle(particle,
                color=self.scalarMap.to_rgba(order),
                fill=True, border='black',
                label=self.label)                                   # draw particle circle with color corresponding to bond orientation order parameter norm

class Density(_Frame):
    """
    Plotting class specific to 'density' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        arrow_width=_arrow_width,
        arrow_head_width=_arrow_head_width,
        arrow_head_length=_arrow_head_length,
        pad=_colormap_label_pad,
        label=False, **kwargs):
        """
        Initialises and plots figure.

        Parameters
        ----------
        dat : coll_dyn_activem.read.Dat
            Data object.
        frame : int
            Frame to render.
        box_size : float
            Length of the square box to render.
        centre : 2-uple like
            Centre of the box to render.
        pad : float
            Separation between label and colormap.
            (default: coll_dyn_activem.frame._colormap_label_pad)
        label : bool
            Write indexes of particles in circles. (default: False)

        Optional keyword parameters
        ---------------------------
        (see coll_dyn_activem.frame._Frame)
        """

        super().__init__(dat, frame, box_size, centre,
            **kwargs)   # initialise superclass

        self.localDensity = np.abs(
            Positions(
                dat.filename, corruption=dat.corruption).getLocalDensity(frame))

        self.colorbar(0, 1, cmap=plt.cm.inferno)    # add colorbar to figure
        self.colormap.set_label(                    # colorbar legend
            r'$\rho_{\mathrm{loc}}$' + ' '
                + r'$(f_{\rho}=%.3f)$'
                    % (self.localDensity.min()/self.localDensity.max()),
            labelpad=pad, rotation=270)

        self.label = label  # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, density in zip(self.particles, self.localDensity):    # for particle and particle's local density in rendered box
            self.draw_circle(particle,
                color=self.scalarMap.to_rgba(density),
                fill=True, border='black',
                label=self.label)                                           # draw particle circle with color corresponding to local density

class Bare(_Frame):
    """
    Plotting class specific to 'bare' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        label=False, **kwargs):
        """
        Initialises and plots figure.

        Parameters
        ----------
        dat : coll_dyn_activem.read.Dat
            Data object.
        frame : int
            Frame to render.
        box_size : float
            Length of the square box to render.
        centre : 2-uple like
            Centre of the box to render.
        label : bool
            Write indexes of particles in circles. (default: False)

        Optional keyword parameters
        ---------------------------
        (see coll_dyn_activem.frame._Frame)
        """

        super().__init__(dat, frame, box_size, centre, **kwargs)    # initialise superclass

        self.label = label  # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle in self.particles:                             # for particle in rendered box
            self.draw_circle(particle, color='black', fill=False,   # draw black circle
                label=self.label)

# SCRIPT

if __name__ == '__main__':  # executing as script

    startTime = datetime.now()

    # VARIABLE DEFINITIONS

    mode = get_env('MODE', default='orientation')           # plotting mode
    if mode == 'orientation':
        plotting_object = Orientation
    elif mode == 'displacement':
        plotting_object = Displacement
    elif mode == 'movement':
        plotting_object = Movement
    elif mode == 'overlap':
        plotting_object = Overlap
    elif mode == 'bond':
        plotting_object = Bond
    elif mode == 'd2min':
        plotting_object = D2min
    elif mode == 'velocity':
        plotting_object = Velocity
    elif mode == 'kinetic':
        plotting_object = Kinetic
    elif mode == 'interaction':
        plotting_object = Interaction
    elif mode == 'order':
        plotting_object = Order
    elif mode == 'density':
        plotting_object = Density
    elif mode == 'bare':
        plotting_object = Bare
    else: raise ValueError('Mode %s is not known.' % mode)  # mode is not known

    dat_file = get_env('DAT_FILE', default=joinpath(getcwd(), 'out.dat'))   # data file
    dat = Dat(dat_file, loadWork=False,
        corruption=get_env('CORRUPTION', default=None, vartype=str))        # data object

    init_frame = get_env('INITIAL_FRAME', default=-1, vartype=int)  # initial frame to render

    dt = get_env('DT', default=-1, vartype=int)     # displacement lag time (PLOT mode)
    jump = get_env('JUMP', default=1, vartype=int)  # jump when computing displacements
    a = get_env('A', default=0.3, vartype=float)    # length scale to compute overlaps
    a1 = get_env('A1', default=1.15, vartype=float) # distance relative to their diameters below which particles are considered bonded
    a2 = get_env('A2', default=1.5, vartype=float)  # distance relative to their diameters above which particles are considered unbonded

    rescale = get_env('RESCALE', default=False, vartype=bool)   # rescale values by root mean square

    box_size = get_env('BOX_SIZE', default=dat.L, vartype=float)    # size of the square box to consider
    centre = (get_env('X_ZERO', default=0, vartype=float),
        get_env('Y_ZERO', default=0, vartype=float))                # centre of the box

    try:
        Nentries = dat.frameIndices[-1]
        init_frame = (dat.frameIndices[init_frame] if init_frame < 0
            else init_frame)                                                # initial frame to draw
        dat._getFrameIndex(init_frame)                                      # throws IndexError if init_frame not in file
    except AttributeError:
        Nentries = dat.frames - 1
        init_frame = int(Nentries/2) if init_frame < 0 else init_frame      # initial frame to draw

    # FIGURE PARAMETERS

    vmin = get_env('V_MIN', vartype=float) # minimum value of the colorbar
    vmax = get_env('V_MAX', vartype=float) # maximum value of the colorbar

    frame_hor = get_env('FRAME_HORIZONTAL_SIZE', default=_frame_hor,
        vartype=float)  # horizontal size of the frame (in inches)
    frame_ver = get_env('FRAME_VERTICAL_SIZE', default=_frame_ver,
        vartype=float)  # vertical size of the frame (in inches)
    mpl.rcParams['figure.figsize'] = (frame_hor, frame_ver)

    frame_def = get_env('FRAME_DEFINITION', default=_frame_def,
        vartype=float)                                                  # definition of image (in dots per inches (dpi))
    font_size = get_env('FONT_SIZE', default=_font_size, vartype=float) # font size
    mpl.rcParams.update({'savefig.dpi': frame_def, 'font.size': font_size})

    arrow_width = get_env('ARROW_WIDTH', default=_arrow_width,
        vartype=float)  # width of the arrows
    arrow_head_width = get_env('HEAD_WIDTH', default=_arrow_head_width,
        vartype=float)  # width of the arrows' head
    arrow_head_length = get_env('HEAD_LENGTH', default=_arrow_head_length,
        vartype=float)  # length of the arrows' head
    arrow_factor = get_env('ARROW_FACTOR', default=1,
        vartype=float)  # arrow dilatation factor

    pad = get_env('PAD', default=_colormap_label_pad, vartype=float)    # separation between label and colormap

    # LEGEND SUPTITLE

    display_suptitle = get_env('SUPTITLE', default=True, vartype=bool)  # display suptitle

    def suptitle(frame, lag_time=None):
        """
        Returns figure suptitle.

        NOTE: Returns empty string if display_suptitle=False.

        Parameters
        ----------
        frame : int
            Index of rendered frame.
        lag_time : int
            Lag time between frames.

        Returns
        -------
        suptitle : string
            Suptitle.
        """

        if not(display_suptitle): return ''

        if dat._type == 'dat':
            suptitle = (
                str(r'$N=%.2e, \phi=%1.4f, l_p/\sigma=%.2e$'
        		% (dat.N, dat.phi, dat.lp)))
            Dr = 1/dat.lp
        else:
            suptitle = (
                str(r'$N=%.2e, \phi=%1.4f, D=%.2e, D_r=%.2e$'
        		% (dat.N, dat.phi, dat.D, dat.Dr))
                + '\n' + str(r'$\epsilon=%.2e, v_0=%.2e, I=%.2f$'
                % (dat.epsilon, dat.v0, dat.I)))
            Dr = dat.Dr

        suptitle += str(r'$, L=%.3e$' % dat.L)
        if 'BOX_SIZE' in envvar:
            suptitle += str(r'$, L_{new}=%.3e$' % box_size)
        suptitle += '\n'
        if 'X_ZERO' in envvar or 'Y_ZERO' in envvar:
            suptitle += str(r'$x_0 = %.3e, y_0 = %.3e$' % centre) + '\n'
        suptitle += str(r'$D_r t = %.5e$'
            % (frame*dat.dt*dat.dumpPeriod*Dr))
        if lag_time != None:
            suptitle += str(
                r'$, \Delta t = %.5e, D_r \Delta t = %.5e$'
                % (lag_time*dat.dt*dat.dumpPeriod,
                    lag_time*dat.dt*dat.dumpPeriod*Dr))

        return suptitle

    # MODE SELECTION

    if get_env('PLOT', default=False, vartype=bool):    # PLOT mode

        if mode in ('displacement', 'movement', 'overlap', 'bond', 'd2min'):
            try:
                dt = dt if dt >= 0 else dat.frameIndices[
                    dat.frameIndices.tolist().index(init_frame) + dt]
            except AttributeError:
                dt = dt if dt >=0 else Nentries - init_frame + dt
        else: dt = None

        figure = plotting_object(dat, init_frame, box_size, centre,
            arrow_width=arrow_width, arrow_head_width=arrow_head_width,
            arrow_head_length=arrow_head_length, arrow_factor=arrow_factor,
            pad=pad, dt=dt, jump=jump, a=a, a1=a1, a2=a2, rescale=rescale,
            vmin=vmin, vmax=vmax,
            label=get_env('LABEL', default=False, vartype=bool))
        figure.fig.suptitle(suptitle(init_frame, lag_time=dt))

        if get_env('SAVE', default=False, vartype=bool):    # SAVE mode
            figure_name = get_env('FIGURE_NAME', default='out')
            figure_ext = get_env('FIGURE_EXTENSION')
            if type(figure_ext) == type(None):
                figure.fig.savefig(figure_name + '.eps')
                figure.fig.savefig(figure_name + '.svg')
            else:
                figure.fig.savefig(figure_name + ('.%s' % figure_ext))

    if get_env('MOVIE', default=False, vartype=bool):   # MOVIE mode

        frame_fin = get_env('FINAL_FRAME', default=Nentries, vartype=int)       # final movie frame
        frame_per = get_env('FRAME_PERIOD', default=_frame_per, vartype=int)    # frame rendering period
        frame_max = get_env('FRAME_MAXIMUM', default=_frame_max, vartype=int)   # maximum number of frames

        movie_dir = get_env('MOVIE_DIR', default='out.movie')   # movie directory name
        mkdir(movie_dir)                                        # create movie directory
        mkdir(joinpath(movie_dir, 'frames'), replace=True)      # create frames directory (or replaces it if existing)

        frames = [init_frame + i*frame_per for i in range(frame_max)
            if init_frame + i*frame_per <= frame_fin]   # rendered frames

        for frame in frames:    # for rendered frames
            sys.stdout.write(
                'Frame: %d' % (frames.index(frame) + 1)
                + "/%d \r" % len(frames))

            if get_env('REMOVE_CM', default=True, vartype=bool):
                remove_cm = init_frame
            else: remove_cm = None

            plot_frame = frame
            lag = frame_per
            if mode in ('overlap', 'bond'):
                plot_frame = init_frame
                lag = frame - plot_frame

            figure = plotting_object(dat, plot_frame, box_size, centre,
                arrow_width=arrow_width, arrow_head_width=arrow_head_width,
                arrow_head_length=arrow_head_length, arrow_factor=arrow_factor,
                pad=pad, dt=lag, jump=jump, a=a, a1=a1, a2=a2, rescale=rescale,
                vmin=vmin, vmax=vmax,
                remove_cm=remove_cm,
                label=get_env('LABEL', default=False, vartype=bool))    # plot frame
            figure.fig.suptitle(suptitle(frame, frame_per))
            if remove_cm != None:
                figure.ax.set_xlabel(r'$\Delta^{\mathrm{CM}} x$')
                figure.ax.set_ylabel(r'$\Delta^{\mathrm{CM}} y$')

            tracer = get_env('TRACER', vartype=int)
            if tracer != None:
                if tracer in figure.particles:
                    figure.draw_circle(tracer,
                        color='black', fill=True, border=None,
                        label=False)
                    if get_env('NEIGHBOURS', default=False, vartype=bool):
                        for neighbour, _ in (
                            Positions(figure.dat.filename).getNeighbourList(
                                init_frame)[tracer]):
                            figure.draw_circle(neighbour,
                                color='red', fill=True, border=None,
                                label=False)
                        for neighbour, _ in (
                            Positions(figure.dat.filename).getNeighbourList(
                                frame)[tracer]):
                            figure.draw_circle(neighbour,
                                color='green', fill=True, border=None,
                                label=False)

            figure.fig.savefig(joinpath(movie_dir, 'frames',
                '%010d' % frames.index(frame) + '.png'))    # save frame
            del figure                                      # delete (close) figure

        subprocess.call([
            'ffmpeg',
            '-r', str(get_env('FRAME_RATE', default=5, vartype=int)),
            '-f', 'image2', '-s', '1280x960', '-i',
            joinpath(movie_dir , 'frames', '%10d.png'),
            '-pix_fmt', 'yuv420p', '-y',
            joinpath(movie_dir, get_env('FIGURE_NAME', default='out.mp4'))
            ])  # generate movie from frames

    # EXECUTION TIME
    print("Execution time: %s" % (datetime.now() - startTime))

    if get_env('SHOW', default=False, vartype=bool):    # SHOW mode
        plt.show()
