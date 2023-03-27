"""
Module frame renders images of the 2D system.

(modified from
https://github.com/yketa/active_particles/tree/master/analysis/frame.py)
"""

from coll_dyn_activem.init import get_env, mkdir
from coll_dyn_activem.read import Dat
from coll_dyn_activem.maths import normalise1D, amplogwidth, cooperativity,\
    relative_positions, angle, wo_mean
from coll_dyn_activem.flow import Displacements, Velocities
from coll_dyn_activem.structure import Positions
from coll_dyn_activem.force import Force
from coll_dyn_activem.pycpp import pairIndex, invPairIndex
from coll_dyn_activem._pycpp import getBondsBrokenBonds

from os import getcwd, cpu_count
from os import environ as envvar
from os.path import join as joinpath

import sys

from math import ceil

import numpy as np
np.seterr(divide='ignore')

import matplotlib as mpl
if get_env('NO_DISPLAY', default=False, vartype=bool): mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import Normalize as ColorsNormalise
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amsfonts}')

from datetime import datetime

from collections import OrderedDict

from operator import itemgetter

import subprocess

from pathos.multiprocessing import ProcessingPool as Pool

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
        WCA_diameter=False, confinement=False, rasterized=False,
        linewidth=1, colorbar_position='right', colorbar_label=None,
        **kwargs):
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
        confinement : bool
            Draw circle of confinement, at centre (dat.L/2, dat.L/2), with
            diameter dat.L - max(dat.diameters). (default: False)
        rasterized : bool
            Force rasterized (bitmap) drawing for vector graphics output.
            (default: False)
        linewidth : float or None
            Linewidth of circles and arrows. (default: None)
        colorbar_position : string
            Position of the colorbar ('right' or 'top'). (default: 'right')
        colorbar_label : string or None
            Label of the colorbar. (default: None)

        Optional keyword parameters
        ---------------------------
        remove_cm : int or None
            Remove centre of mass displacement from frame.
            NOTE: if remove_cm == None, nothing is done.
        """

        self.fig, self.ax = plt.subplots()
        self.box_size = box_size
        self.ax.set_xlim(np.array([-0.5, 0.5])*min(self.box_size, dat.Lxy[0]))
        self.ax.set_ylim(np.array([-0.5, 0.5])*min(self.box_size, dat.Lxy[1]))
        # self.ax.set_xlabel(r'$x$')
        # self.ax.set_ylabel(r'$y$')
        self.ax.set_aspect('equal')
        self.ax.tick_params(axis='both', which='both', direction='in',
            bottom=True, top=True, left=True, right=True)

        self.dat = dat
        self.frame = frame
        self.centre = centre
        self.positions = self.dat.getPositions(frame)
        if 'remove_cm' in kwargs:
            if kwargs['remove_cm'] != None:
                self.remove_cm = kwargs['remove_cm']
                # self.ax.set_xlabel(r'$\Delta^{\mathrm{CM}} x$')
                # self.ax.set_ylabel(r'$\Delta^{\mathrm{CM}} y$')
                self.positions -= self.dat.getDisplacements(
                    self.remove_cm, frame).mean(axis=0, keepdims=True)
                self.positions -= (self.positions//self.dat.Lxy)*self.dat.Lxy
        self.positions = relative_positions(self.positions, centre,
                self.dat.Lxy)                                                   # particles' positions at frame frame with centre as centre of frame
        self.diameters = self.dat.diameters*(2**(1./6.) if WCA_diameter else 1) # particles' diameters

        if confinement:
            self.ax.add_artist(
                plt.Circle(
                    relative_positions(
                        (self.dat.L/2, self.dat.L/2), centre, self.dat.L),
                    radius=(self.dat.L - self.dat.diameters.max())/2,
                    ec='red', fc=(1, 1, 1, 0)))

        self.particles = np.where(
            (np.abs(self.positions) - self.diameters.reshape((self.dat.N, 1))/2
                <= self.box_size/2).prod(axis=-1))[0]   # particles inside box of centre centre and length box_size
        # self.particles = [particle for particle in range(len(self.positions))
        #     if (np.abs(self.positions[particle]) <= self.box_size/2
        #         + self.diameters.reshape((self.dat.N, 1))/2).all()] # particles inside box of centre centre and length box_size

        self.arrow_width = arrow_width
        self.arrow_head_width = arrow_head_width
        self.arrow_head_length = arrow_head_length

        self.rasterized = rasterized

        self.linewidth = linewidth

        self.colorbar_position = colorbar_position
        self.colorbar_label = colorbar_label

    def __del__(self):
        """
        Closes figure.
        """

        plt.close(self.fig)

    def draw_circle(self, particle, color='black', fill=False, alpha=1,
        border=None, label=False):
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
        alpha : float
            Opacity. (default: 1)
        border : string or None
            Circle border color. (default: None)
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
                self.diameters[particle]/2, color=color, fill=fill, alpha=alpha,
                linewidth=self.linewidth,
                zorder=0, rasterized=self.rasterized)   # circle artist representing particle
            self.ax.add_artist(circle)
            if border != None:                          # add border
                circleBorder = plt.Circle(position,
                    self.diameters[particle]/2, color=border, fill=False,
                    linewidth=self.linewidth,
                    zorder=1, rasterized=self.rasterized)
                self.ax.add_artist(circleBorder)

        _draw_circle(self.positions[particle])

        for dim in range(2):
            if (np.abs(self.positions[particle][dim]) >
                self.dat.Lxy[dim]/2 - self.diameters[particle]/2):
                newPosition = self.positions[particle].copy()
                newPosition[dim] -= (np.sign(self.positions[particle][dim])
                    *self.dat.Lxy[dim])
                _draw_circle(newPosition)
        if (np.abs(self.positions[particle]) >
            self.dat.Lxy/2 - self.diameters[particle]/2).all():
            newPosition = self.positions[particle].copy()
            newPosition -= (np.sign(self.positions[particle])
                *self.dat.Lxy)
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
            head_length=length*self.arrow_head_length,
            length_includes_head=True,
            zorder=1,
            linewidth=self.linewidth,
            rasterized=self.rasterized)

    def draw_arrow_in(self, particle, orientation, color='black'):
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
            length_includes_head=True,
            linewidth=self.linewidth,
            rasterized=self.rasterized)

    def draw_link(self, particle1, particle2, width=4, color='black'):
        """
        Draw link between particles.

        Parameters
        ----------
        particle1 : int
            First particle index.
        particle2 : int
            Second particle index.
        width : int
            Line width. (default: 4)
        color : any matplotlib color
            Link color. (default: 'black')
        """

        ri, rj = self.positions[particle1], self.positions[particle2]
        self.ax.add_artist(Line2D(
            [ri[0], ri[0] + self.dat._diffPeriodic(ri[0], rj[0])],
            [ri[1], ri[1] + self.dat._diffPeriodic(ri[1], rj[1])],
            lw=width, color=color, rasterized=self.rasterized))
        self.ax.add_artist(Line2D(
            [rj[0], rj[0] + self.dat._diffPeriodic(rj[0], ri[0])],
            [rj[1], rj[1] + self.dat._diffPeriodic(rj[1], ri[1])],
            lw=width, color=color, rasterized=self.rasterized))

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

        self.cmap = cmap

        self.cmap_norm = ColorsNormalise(vmin=vmin, vmax=vmax)
        self.scalarMap = ScalarMappable(norm=self.cmap_norm, cmap=self.cmap)

        class Colormap(mpl.colorbar.ColorbarBase):
            def set_label(_, label, **kwargs):
                super().set_label(
                    label if self.colorbar_label == None
                        else self.colorbar_label,
                    **kwargs,
                    rotation=270 if self.colorbar_position == 'right' else 0)

        if self.colorbar_position == 'right':

            self.colormap_ax = make_axes_locatable(self.ax).append_axes('right',
                size='5%', pad=0.05)
            self.colormap = Colormap(
                self.colormap_ax, cmap=self.cmap,
                norm=self.cmap_norm, orientation='vertical')

        elif self.colorbar_position == 'top':

            self.colormap_ax = make_axes_locatable(self.ax).append_axes('top',
                size='5%', pad=0.05)
            self.colormap = Colormap(
            	self.colormap_ax, cmap=self.cmap,
                norm=self.cmap_norm, orientation='horizontal')
            self.colormap_ax.xaxis.set_label_position('top')
            self.colormap_ax.xaxis.set_ticks_position('top')

        else:
            raise ValueError(
                "Position '%s' is not recognised." % self.colorbar_position)

    def colorbar_discrete(self, bounds, cmap=plt.cm.jet):
        """
        Adds discrete colorbar to plot.

        Parameters
        ----------
        bounds : (*,) array-like
            Bounds of values to map to colors.
        cmap : matplotlib colorbar
            Matplotlib colorbar to be used. (default: matplotlib.pyplot.cm.jet)
        """

        self.colormap_bounds = np.array(sorted(bounds))

        _cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_cmap',
            list(map(cmap, range(cmap.N))), cmap.N)
        self.cmap_norm = mpl.colors.BoundaryNorm(self.colormap_bounds, cmap.N)
        self.scalarMap = mpl.cm.ScalarMappable(norm=self.cmap_norm, cmap=_cmap)

        class Colormap(mpl.colorbar.ColorbarBase):
            def set_label(_, label, **kwargs):
                super().set_label(
                    label if self.colorbar_label == None
                        else self.colorbar_label,
                    **kwargs,
                    rotation=270 if self.colorbar_position == 'right' else 0)

        if self.colorbar_position == 'right':

            self.colormap_ax = make_axes_locatable(self.ax).append_axes('right',
                size='5%', pad=0.05)
            self.colormap = Colormap(
                self.colormap_ax, cmap=_cmap,
                norm=self.cmap_norm, orientation='vertical')

        elif self.colorbar_position == 'top':

            self.colormap_ax = make_axes_locatable(self.ax).append_axes('top',
                size='5%', pad=0.05)
            self.colormap = Colormap(
            	self.colormap_ax, cmap=_cmap,
                norm=self.cmap_norm, orientation='horizontal')
            self.colormap_ax.xaxis.set_label_position('top')
            self.colormap_ax.xaxis.set_ticks_position('top')

        else:
            raise ValueError(
                "Position '%s' is not recognised." % self.colorbar_position)

    def _colorbar_discrete(self, nColors, vmin, vmax, cmap=plt.cm.jet):
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

        self.cmap_norm = BoundaryNorm(
            np.linspace(vmin, vmax, nColors + 1), nColors)
        _cmap = ListedColormap(
            [ScalarMappable(
                norm=ColorsNormalise(vmin=vmin, vmax=vmax), cmap=cmap
                ).to_rgba(x)
                for x in np.linspace(vmin, vmax, nColors)])
        self.scalarMap = ScalarMappable(norm=self.cmap_norm, cmap=_cmap)

        self.colormap_ax = make_axes_locatable(self.ax).append_axes('right',
            size='5%', pad=0.05)
        self.colormap = mpl.colorbar.ColorbarBase(self.colormap_ax, cmap=_cmap,
            norm=self.cmap_norm, orientation='vertical')
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
        self.colormap.set_label(r'$\theta_i/\pi$', labelpad=pad)  # colorbar legend

        self.label = label  # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, orientation in zip(self.particles,
            self.orientations):                                     # for particle and particle's displacement in rendered box
            self.draw_circle(particle,
                color=self.scalarMap.to_rgba(orientation/np.pi), fill=False,
                label=self.label)                                   # draw particle circle with color corresponding to displacement amplitude
            self.draw_arrow_in(particle, orientation,
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

        self.dt = dt
        self.zetad = cooperativity(
            dat.getDisplacements(frame, frame + dt, jump=jump, remove_cm=False))
        # self.displacements = dat.getDisplacements(
        #     frame, frame + dt, *self.particles, jump=jump, remove_cm=True)  # particles' displacements at frame
        self.displacements = Displacements(dat.filename).getDisplacements(
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
            r'$\log_{10} ||\delta \Delta \boldsymbol{r}_i(t, t + \Delta t)||$'
                + (r'$/\sqrt{\left<|\ldots|^2\right>_i}$'
                    if self.rescale else '')
                # + ' '
                # + r'$(\zeta_{\Delta \boldsymbol{r}} = %.4f)$' % self.zetad
                ,
            labelpad=pad)

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
            # self.draw_circle(particle,
            #     color='red' if np.linalg.norm(displacement) > 0.25 else 'white',
            #     fill=True,
            #     border='black',
            #     label=self.label)                                           # draw particle circle with color corresponding to displacement amplitude

class DispH(_Frame):
    """
    Plotting class specific to 'disph' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        arrow_width=_arrow_width,
        arrow_head_width=_arrow_head_width,
        arrow_head_length=_arrow_head_length,
        pad=_colormap_label_pad, jump=1,
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

        self.dt = dt

        self.hess = Force(dat.filename).getRAHessian(frame)
        self.dp = wo_mean(
            dat.getPropulsions(frame + dt) - dat.getPropulsions(frame))
        self.displacements = wo_mean(np.reshape(
            np.linalg.solve(self.hess, self.dp.flatten()),
            (dat.N, 2)))    # displacements from Hessian
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
            r'$\log_{10} ||\mathbb{H}^{-1} \delta \boldsymbol{p}_i||$'
                + (r'$/\sqrt{\left<|\ldots|^2\right>_i}$'
                    if self.rescale else ''),
            labelpad=pad)

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
            # self.draw_circle(particle,
            #     color='red' if np.linalg.norm(displacement) > 0.25 else 'white',
            #     fill=True,
            #     border='black',
            #     label=self.label)                                           # draw particle circle with color corresponding to displacement amplitude

class Odisplacement(_Frame):
    """
    Plotting class specific to 'odisplacement' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        pad=_colormap_label_pad, dt=1, jump=1, a1=1.15, a2=1.5,
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
        self.odisplacements = np.array(itemgetter(*self.particles)(
            Displacements(self.dat.filename).orientationNeighbours(
                frame, dt, A1=self.a1, jump=jump, remove_cm=True)[0]))
        self.brokenPairs = (
            Displacements(self.dat.filename).brokenPairs(
                frame, frame + dt, A1=self.a1, A2=self.a2))

        self.colorbar_discrete([0, 4, 5, 7], cmap=  # add colorbar to figure
            mpl.colors.LinearSegmentedColormap.from_list('custom_cmap', [
                (1, 1, 1),                          # white
                (255./255., 195./255., 0./255.),    # yellow
                (255./255., 87./255., 51./255.)]))  # orange
        self.colormap.set_label(                    # colorbar legend
            (r'$\sum_{j \in \mathcal{V}_i} \Theta($'
                + r'$\hat{\delta\Delta\boldsymbol{r}}_i(t, t + \Delta t) \cdot$'
                + r'$\hat{\delta\Delta\boldsymbol{r}}_j$'
                + r'$- 0.5), A_1=%.2f, A_2=%.2f$' % (self.a1, self.a2)),
            labelpad=pad)

        self.label = label  # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, odisplacement in zip(
            self.particles, self.odisplacements):   # for particle and particle's number of same displacement orientation neighbours
            self.draw_circle(particle,
                color=self.scalarMap.to_rgba(odisplacement),
                fill=True,
                border='black',
                label=self.label)                   # draw particle circle with color corresponding to number of same displacement orientation neighbours
        for i in range(self.dat.N):
            for j in range(i + 1, self.dat.N):
                if (i in self.particles and j in self.particles
                    and self.brokenPairs[pairIndex(i, j, self.dat.N)]):
                    self.draw_link(i, j) # draw broken bond line from i to j

class Oridisplacement(_Frame):
    """
    Plotting class specific to 'oridisplacement' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        pad=_colormap_label_pad, dt=1, jump=1,
        brokenPairs=False, a1=1.15, a2=1.5, minimum=0,
        border=True, label=False, **kwargs):
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
        brokenPairs : bool
            Compute and display broken bonds between. (default: True)
        a1 : float
            Distance relative to their diameters below which particles are
            considered bonded. (default: 1.15)
        a2 : float
            Distance relative to their diameters above which particles are
            considered unbonded. (default: 1.5)
        minimum : float
            Display particles with squared displcaments greater than this
            minimum. (default: 0)
        border : bool
            Draw black border of particles. (default: True)
        label : bool
            Write indexes of particles in circles. (default: False)

        Optional keyword parameters
        ---------------------------
        (see coll_dyn_activem.frame._Frame)
        """

        super().__init__(dat, frame, box_size, centre,
            **kwargs)   # initialise superclass

        self.a1, self.a2 = a1, a2
        self.minimum = minimum
        self.displacements = dat.getDisplacements(frame, frame + dt,
            jump=jump, remove_cm=True)
        self.oridisplacements = np.array(list(map(
            angle,
            *np.transpose(itemgetter(*self.particles)(self.displacements)))))
        if brokenPairs:
            self.brokenPairs = (
                Displacements(self.dat.filename).brokenPairs(
                    frame, frame + dt, A1=self.a1, A2=self.a2))
        else:
            self.brokenPairs = np.zeros((int(self.dat.N*(self.dat.N - 1)/2),))

        self.colorbar(-1, 1, cmap=plt.cm.hsv)   # add colorbar to figure
        self.colormap.set_label(
            r'$\mathrm{arg}($'
                + r'$\delta \Delta \boldsymbol{r}_i(t, t + \Delta t))/\pi$'
                + ((' ' + r'$(\mathrm{minimum} = %.2f)$' % self.minimum)
                    if self.minimum > 0 else '')
                + (r'$, A_1=%.2f, A_2=%.2f$' % (self.a1, self.a2)
                    if brokenPairs else ''),
            labelpad=pad)                       # colorbar legend

        self.border = border    # draw borders
        self.label = label      # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, oridisplacement in zip(
            self.particles, self.oridisplacements): # for particle and particle's orientation of displacement
            self.draw_circle(particle,
                color=(
                    self.scalarMap.to_rgba(oridisplacement/np.pi)
                        if ((self.displacements[particle]**2).sum(axis=-1)
                            >= self.minimum**2)
                        else 'white'),
                fill=True,
                border='black' if self.border else None,
                label=self.label)                   # draw particle circle with color corresponding to orientation of displacement
        if self.brokenPairs.sum() == 0: return
        for i in range(self.dat.N):
            for j in range(i + 1, self.dat.N):
                if (i in self.particles and j in self.particles
                    and self.brokenPairs[pairIndex(i, j, self.dat.N)]):
                    self.draw_link(i, j) # draw broken bond line from i to j

class Movement(_Frame):
    """
    Plotting class specific to 'movement' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        arrow_width=_arrow_width,
        arrow_head_width=_arrow_head_width,
        arrow_head_length=_arrow_head_length,
        dt=1, jump=1,
        arrow_factor=1, minimum=None,
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
        minimum : float
            Display particles with squared displcaments greater than this
            minimum. (default: None)
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
        self.minimum = (
            minimum if type(minimum) == type(None) or minimum > 0 else None)
        self.side_text = self.fig.text(1.03, 0,
            ((r'$\mathrm{minimum} = %.2f,$' % self.minimum + ' ')
                if type(self.minimum) != type(None) else '')
                + r'$\mathrm{factor} = %.2e$' % self.arrow_factor,
            rotation=270, transform=self.ax.transAxes)

        self.displacements = dat.getDisplacements(
            frame, frame + dt, *self.particles, jump=jump, remove_cm=True)  # particles' displacements at frame

        for i in range(len(self.displacements)):
            for dim in range(2):
                while self.displacements[i, dim] > self.dat.L/2:
                    self.displacements[i, dim] -= self.dat.L
                while self.displacements[i, dim] < -self.dat.L/2:
                    self.displacements[i, dim] += self.dat.L

        self.label = label  # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, displacement in zip(
            self.particles, self.displacements):                            # for particle and particle's displacement in rendered box
            if (type(self.minimum) == type(None)
                or (np.abs(displacement) < self.minimum).all()):
                self.draw_circle(particle,                                      # draw particle
                    color='black',
                    fill=False,
                    label=self.label)
                self.draw_arrow(particle, *(self.arrow_factor*displacement))
            else:
                self.draw_circle(particle,                                      # draw particle
                    color='red',
                    fill=True,
                    border='black',
                    label=self.label)
            # if (displacement**2).sum(axis=-1) < self.minimum**2: continue
            # self.draw_arrow(particle, *(self.arrow_factor*displacement))    # draw displacement arrow

class LemaitrePatinet(_Frame):
    """
    Plotting class specific to 'lemaitrepatinet' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        arrow_width=_arrow_width,
        arrow_head_width=_arrow_head_width,
        arrow_head_length=_arrow_head_length,
        dt=1, arrow_factor=1,
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
        dt : int
            Lag time for displacement. (default: 1)
        arrow_factor : float
            Displacement arrow dilatation factor. (default: 1)
        pad : float
            Separation between label and colormap.
            (default: coll_dyn_activem.frame._colormap_label_pad)
        label : bool
            Write indexes of particles in circles. (default: False)
        """

        super().__init__(dat, frame, box_size, centre,
            arrow_width=arrow_width,
            arrow_head_width=arrow_head_width,
            arrow_head_length=arrow_head_length,
            **kwargs)   # initialise superclass

        self.arrow_factor = arrow_factor    # displacement arrow dilatation factor
        self.side_text = self.fig.text(1.10, 0,
            r'$\mathrm{factor} = %.2e$' % self.arrow_factor,
            rotation=270, transform=self.ax.transAxes)

        self.displacements = dat.getDisplacements(
            frame, frame + dt, remove_cm=True)                                  # particles' displacements at frame
        self.d_propulsions = wo_mean(
            dat.getPropulsions(frame + dt) - dat.getPropulsions(frame))
        self.hessian = Force(dat.filename).getRAHessian(frame, a=12, rcut=1.25) # hessian matrix at frame
        self.residual_force = np.sqrt((Force(dat.filename).getRAResidualForce(  # residual force
            frame, frame + dt, a=12, rcut=1.25)**2).sum(axis=-1))

        self.colorbar(20, 200, cmap=plt.cm.hot.reversed())  # add colorbar to figure
        self.colormap.set_label(
            r'$|\boldsymbol{f}_{\mathrm{res}, i}(t, t + \Delta t)|$',
            labelpad=pad)                                   # colorbar legend

        self.label = label  # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, displacement, force in zip(
            self.particles, self.displacements, self.residual_force):    # for particle, particle's displacement and particle's residual in rendered box
            self.draw_circle(particle,                                      # draw particle
                color=self.scalarMap.to_rgba(force),
                border='black',
                fill=True,
                label=self.label)
            self.draw_arrow(particle, *(self.arrow_factor*displacement))

class Eigenmode(_Frame):
    """
    Plotting class specific to 'eigenmode' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        n=0,
        arrow_width=_arrow_width,
        arrow_head_width=_arrow_head_width,
        arrow_head_length=_arrow_head_length,
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
        n : int
            Index of the eigenmode to draw, in increasing order of eigenvalue.
            (default: 0)
        arrow_width : float
            Width of the arrows.
        arrow_head_width : float
            Width of the arrows' head.
        arrow_head_length : float
            Length of the arrows' head.
        arrow_factor : float
            Eigenvector arrow dilatation factor. (default: 1)
        label : bool
            Write indexes of particles in circles. (default: False)
        """

        super().__init__(dat, frame, box_size, centre,
            arrow_width=arrow_width,
            arrow_head_width=arrow_head_width,
            arrow_head_length=arrow_head_length,
            **kwargs)   # initialise superclass

        self.n = n  # eigenmode index

        self.arrow_factor = arrow_factor    # displacement arrow dilatation factor

        eval, evec = np.linalg.eig(
            Force(dat.filename).getRAHessian(frame, a=12, rcut=1.25))
        self.mode = evec[np.argsort(eval)[self.n]]
        self.mode = np.reshape(self.mode, (dat.N, 2))
        self.mode /= (self.mode**2).sum(axis=-1).mean()*np.sqrt(dat.N)
        print((self.mode**2).sum(axis=-1).mean())

        self.label = label  # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, mode in zip(self.particles, self.mode):   # for particle and particle's projection on mode
            self.draw_circle(particle,                          # draw particle
                color='black',
                fill=False,
                label=self.label)
            self.draw_arrow(particle, *(self.arrow_factor*mode))

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
        self.overlaps = (dat.getDisplacements(
            frame, frame + dt, *self.particles, jump=jump, remove_cm=True,
            norm=True)/self.a > 1)*1.0  # particles' overlaps between frame and frame + dt

        self.colorbar_discrete((0, 0.5, 1), cmap=plt.cm.binary) # add colorbar to figure
        self.colormap.set_label(                                # colorbar legend
            (r'$\Theta($'
                + r'$||\delta \Delta \boldsymbol{r}_i(t, t + \Delta t)||/a$'
                + r'$-1),$' + ' ' + r'$a=%.2f$' % self.a),
            labelpad=pad)

        self.label = label  # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, overlap in zip(
            self.particles, self.overlaps):                                 # for particle and particle's overlap in rendered box
            self.draw_circle(particle,
                border='black',
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
        self.brokenBonds = np.array(itemgetter(*self.particles)(Displacements(
            dat.filename, corruption=dat.corruption).brokenBonds(
                np.min([frame, frame + dt]), np.abs(dt),
                A1=self.a1, A2=self.a2)[0]))    # particles' number of broken bonds between frame and frame + dt

        self.colorbar_discrete([0, 1, 2, 3, 4]) # add colorbar to figure
        self.colormap.set_label(                # colorbar legend
            # (r'$B_i(t, t + \Delta t, A_1=%.2f, A_2=%.2f)$'
            #     % (self.a1, self.a2))
            (r'$\mathcal{B}_i(t, t + \Delta t, A_1=%.2f, A_2=%.2f)$'
                % (self.a1, self.a2)),
            labelpad=pad)

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
                fill=True, border='black',
                label=self.label)               # draw particle circle with color corresponding to number of broken bond

class Cbond(_Frame):
    """
    Plotting class specific to 'cbond' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        pad=_colormap_label_pad, dt=1, a1=1.15, a2=1.5, minimum=0.5,
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
        minimum : float
            Show in red particles with minimum proportion of broken bonds.
            (default: 0.5)
        label : bool
            Write indexes of particles in circles. (default: False)

        Optional keyword parameters
        ---------------------------
        (see coll_dyn_activem.frame._Frame)
        """

        super().__init__(dat, frame, box_size, centre,
            **kwargs)   # initialise superclass

        frame, dt = np.min([frame, frame + dt]), np.abs(dt)

        self.a1, self.a2 = a1, a2
        self.minimum = minimum if minimum > 0 else 0.5
        self.initialBonds, self.brokenBonds = getBondsBrokenBonds(
            dat.getPositions(frame), dat.getDisplacements(frame, frame + dt),
            dat.diameters, dat.L, self.a1, self.a2) # number of initial and broken bonds
        self.initialBonds = np.array(itemgetter(*self.particles)(
            self.initialBonds))
        self.brokenBonds = np.array(itemgetter(*self.particles)(
            self.brokenBonds))

        cvals = [0, self.minimum - 0.05, self.minimum, self.minimum + 0.05, 1]
        colors = ["blue", "blue", "magenta", "red", "red"]
        norm = plt.Normalize(min(cvals),max(cvals))
        tuples = list(zip(map(norm,cvals), colors))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

        self.colorbar(0, 1, cmap=cmap.reversed())   # add colorbar to figure
        self.colormap.set_label(                    # colorbar legend
            (r'$C_b^i(t, a_1=%.2f, a_2=%.2f)$'
                % (self.a1, self.a2)),
            labelpad=pad)

        self.label = label  # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, ib, bb in zip(
            self.particles, self.initialBonds, self.brokenBonds):   # for particle and particle's number of initial and broken bonds in rendered box
            self.draw_circle(particle,
                color=self.scalarMap.to_rgba(1 - bb/ib),
                fill=True,
                border='black',
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
        self.d2min = np.array(itemgetter(*self.particles)(Displacements(
            dat.filename, corruption=dat.corruption).d2min(
                frame, frame + dt, A1=self.a1)))    # particles' nonaffine squared displacements between frame and frame + dt

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
            labelpad=pad)

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
        direction=True,
        arrow_width=_arrow_width,
        arrow_head_width=_arrow_head_width,
        arrow_head_length=_arrow_head_length,
        pad=_colormap_label_pad,
        border=False, label=False, **kwargs):
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
        direction : bool
            Draw arrow in the direction of the velocity. (default: True)
        arrow_width : float
            Width of the arrows.
        arrow_head_width : float
            Width of the arrows' head.
        arrow_head_length : float
            Length of the arrows' head.
        pad : float
            Separation between label and colormap.
            (default: coll_dyn_activem.frame._colormap_label_pad)
        border : bool
            Draw black border of particles. (default: True)
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

        self.direction = direction

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
            r'$\log_{10}||\delta \boldsymbol{v}_i(t)||$' + ' '
                + r'$(\zeta_{\boldsymbol{v}} = %.4f)$' % self.zetav,
            labelpad=pad)

        self.border = border    # draw borders
        self.label = label      # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, velocity in zip(
            self.particles, self.velocities):                               # for particle and particle's velocity in rendered box
            self.draw_circle(particle,
                color=self.scalarMap.to_rgba(
                    np.log10(np.linalg.norm(velocity))),
                fill=True,
                border='black' if self.border else None,
                label=self.label)                                           # draw particle circle with color corresponding to velocity amplitude
            if self.direction:
                self.draw_arrow(particle,
                    *normalise1D(velocity)*0.75*self.diameters[particle])   # draw velocity direction arrow

class Bvelocity(_Frame):
    """
    Plotting class specific to 'odisplacement' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        pad=_colormap_label_pad, dt=1, jump=1, a1=1.15, a2=1.5,
        arrow_width=_arrow_width,
        arrow_head_width=_arrow_head_width,
        arrow_head_length=_arrow_head_length,
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
        a1 : float
            Distance relative to their diameters below which particles are
            considered bonded. (default: 1.15)
        a2 : float
            Distance relative to their diameters above which particles are
            considered unbonded. (default: 1.5)
        arrow_width : float
            Width of the arrows.
        arrow_head_width : float
            Width of the arrows' head.
        arrow_head_length : float
            Length of the arrows' head.
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
            r'$\log_{10}||\delta \boldsymbol{v}_i(t)||$',
            labelpad=pad)

        self.a1, self.a2 = a1, a2
        self.brokenPairs = (
            Displacements(self.dat.filename).brokenPairs(
                frame, frame + dt, A1=self.a1, A2=self.a2))

        self.label = label  # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, velocity in zip(
            self.particles, self.velocities):                               # for particle and particle's velocity in rendered box
            self.draw_circle(particle,
                color=self.scalarMap.to_rgba(
                    np.log10(np.linalg.norm(velocity))),
                fill=True,
                border='black',
                label=self.label)                                           # draw particle circle with color corresponding to velocity amplitude
            self.draw_arrow(particle,
                *normalise1D(velocity)*0.75*self.diameters[particle])       # draw velocity direction arrow
        for pair in np.where(self.brokenPairs)[0]:
            i, j = invPairIndex(pair, self.dat.N)
            if i in self.particles and j in self.particles:
                self.draw_link(i, j)                                        # draw broken bond line from i to j

class Ovelocity(_Frame):
    """
    Plotting class specific to 'ovelocity' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        pad=_colormap_label_pad, a1=1.15,
        border=True, label=False, **kwargs):
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
        a1 : float
            Distance relative to their diameters below which particles are
            considered bonded. (default: 1.15)
        border : bool
            Draw black border of particles. (default: True)
        label : bool
            Write indexes of particles in circles. (default: False)

        Optional keyword parameters
        ---------------------------
        (see coll_dyn_activem.frame._Frame)
        """

        super().__init__(dat, frame, box_size, centre,
            **kwargs)   # initialise superclass

        self.a1 = a1
        self.ovelocities = np.array(itemgetter(*self.particles)(
            Velocities(self.dat.filename).orientationNeighbours(
                frame, A1=self.a1, remove_cm=True)[0]))

        self.colorbar_discrete([0, 4, 5, 7], cmap=  # add colorbar to figure
            mpl.colors.LinearSegmentedColormap.from_list('custom_cmap', [
                (1, 1, 1),                          # white
                (255./255., 195./255., 0./255.),    # yellow
                (255./255., 87./255., 51./255.)]))  # orange
        self.colormap.set_label(                    # colorbar legend
            (r'$\sum_{j \in \mathcal{V}_i} \Theta($'
                + r'$\hat{\delta\dot{\boldsymbol{r}}}_i(t, t + \Delta t) \cdot$'
                + r'$\hat{\delta\dot{\boldsymbol{r}}}_j(t, t + \Delta t)$'
                + r'$- 0.5), A_1=%.2f$' % (self.a1,)),
            labelpad=pad)

        self.border = border    # draw borders
        self.label = label      # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, ovelocity in zip(
            self.particles, self.ovelocities):   # for particle and particle's number of same velocity orientation neighbours
            self.draw_circle(particle,
                color=self.scalarMap.to_rgba(ovelocity),
                fill=True,
                border='black' if self.border else None,
                label=self.label)                   # draw particle circle with color corresponding to number of same velocity orientation neighbours

class Orivelocity(_Frame):
    """
    Plotting class specific to 'orivelocity' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        pad=_colormap_label_pad,
        border=True, label=False, **kwargs):
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
        border : bool
            Draw black border of particles. (default: True)
        label : bool
            Write indexes of particles in circles. (default: False)

        Optional keyword parameters
        ---------------------------
        (see coll_dyn_activem.frame._Frame)
        """

        super().__init__(dat, frame, box_size, centre,
            **kwargs)   # initialise superclass

        self.velocities = dat.getVelocities(frame, remove_cm=True)
        self.orivelocities = np.array(list(map(
            angle,
            *np.transpose(itemgetter(*self.particles)(self.velocities)))))

        self.colorbar(-1, 1, cmap=plt.cm.hsv)   # add colorbar to figure
        self.colormap.set_label(
            r'$\mathrm{arg}($'
                + r'$\delta \dot{\boldsymbol{r}}_i(t, t + \Delta t))/\pi$',
            labelpad=pad)                       # colorbar legend

        self.border = border    # draw borders
        self.label = label      # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, oridisplacement in zip(
            self.particles, self.orivelocities):    # for particle and particle's orientation of velocity
            self.draw_circle(particle,
                color=self.scalarMap.to_rgba(oridisplacement/np.pi),
                fill=True,
                border='black' if self.border else None,
                label=self.label)                   # draw particle circle with color corresponding to orientation of velocity

class Pvelocity(_Frame):
    """
    Plotting class specific to 'pvelocity' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        direction=True, minimum=0,
        arrow_width=_arrow_width,
        arrow_head_width=_arrow_head_width,
        arrow_head_length=_arrow_head_length,
        pad=_colormap_label_pad,
        border=False, label=False, **kwargs):
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
        direction : bool
            Draw arrow in the direction of the velocity. (default: True)
        minimum : float
            Display particles with bond orientation order parameter norm
            greater than this minimum. (default: 0)
        arrow_width : float
            Width of the arrows.
        arrow_head_width : float
            Width of the arrows' head.
        arrow_head_length : float
            Length of the arrows' head.
        pad : float
            Separation between label and colormap.
            (default: coll_dyn_activem.frame._colormap_label_pad)
        border : bool
            Draw black border of particles. (default: True)
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

        self.direction = direction

        self.minimum = minimum

        self.order = np.abs(Positions(
            dat.filename, corruption=dat.corruption).getBondOrderParameter(
                frame, *self.particles))    # particles' bond order parameter at frame

        self.velocities = dat.getVelocities(
            frame, *self.particles, remove_cm=True) # particles' velocities at frame

        velocities = np.array(
            [velocity for order, velocity in zip(self.order, self.velocities)
            if order >= self.minimum])
        self.vmin, self.vmax = amplogwidth(velocities)
        try:
            self.vmin = np.log10(kwargs['vmin'])
        except (KeyError, AttributeError, TypeError): pass  # 'vmin' not in keyword arguments or None
        try:
            self.vmax = np.log10(kwargs['vmax'])
        except (KeyError, AttributeError, TypeError): pass  # 'vmax' not in keyword arguments or None

        self.colorbar(self.vmin, self.vmax) # add colorbar to figure
        self.colormap.set_label(            # colorbar
            r'$\log_{10}||\delta \boldsymbol{v}_i(t)||$' + ' '
                + r'$(\min |\psi_{6,i}| = %.2f)$' % self.minimum,
            labelpad=pad)

        self.border = border    # draw borders
        self.label = label      # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, order, velocity in zip(
            self.particles, self.order, self.velocities):                    # for particle and particle's velocity in rendered box
            self.draw_circle(particle,
                color=self.scalarMap.to_rgba(
                    np.log10(np.linalg.norm(velocity))),
                fill=(order >= self.minimum),
                border=(
                    'black' if (self.border or order < self.minimum) else None),
                label=self.label)                                           # draw particle circle with color corresponding to velocity amplitude
            if self.direction and order >= self.minimum:
                self.draw_arrow(particle,
                    *normalise1D(velocity)*0.75*self.diameters[particle])   # draw velocity direction arrow

class Vorticity(_Frame):
    """
    Plotting class specific to 'vorticity' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        a=1, pad=_colormap_label_pad, **kwargs):
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
        a : float
            Scale on which to coarse-grain the velocity field. (default: 1)
        pad : float
            Separation between label and colormap.
            (default: coll_dyn_activem.frame._colormap_label_pad)

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

        try:
            centre = (centre
                + self.dat.getDisplacements(self.remove_cm, frame,
                    remove_cm=False).mean(axis=0))
        except AttributeError:
            pass
        self.p, self.s, self.w, _, _ = self.dat.getVelocityVorticity(
            frame, nBoxes=None, sigma=a, centre=centre)
        self.p -= self.dat.L/2
        self.w /= np.abs(self.w).max()

        self.vmin, self.vmax = -0.25, 0.25
        try:
            self.vmin = float(kwargs['vmin'])
        except (KeyError, AttributeError, TypeError): pass  # 'vmin' not in keyword arguments or None
        try:
            self.vmax = float(kwargs['vmax'])
        except (KeyError, AttributeError, TypeError): pass  # 'vmax' not in keyword arguments or None

        self.colorbar(self.vmin, self.vmax) # add colorbar to figure
        self.colormap.set_label(            # colorbar
            r'$\omega/\mathrm{max}(|\omega|)$',
            labelpad=pad)

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        self.ax.imshow(self.w.T[::-1],                              # vorticity field
            vmin=self.vmin, vmax=self.vmax,
            extent=(
                self.p[:, :, 0].min(), self.p[:, :, 0].max(),
                self.p[:, :, 1].min(), self.p[:, :, 1].max()),
            cmap=self.cmap,
            rasterized=self.rasterized)
        self.ax.streamplot(self.p[:, :, 0].T, self.p[:, :, 1].T,    # stream lines
            self.s[:, :, 0].T, self.s[:, :, 1].T,
            color='black', density=5, linewidth=self.linewidth)

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
            r'$||\delta \boldsymbol{v}_i(t)||^2$' + ' '
                + (r'$(\left<||\delta \boldsymbol{v}_i(t)||^2\right>_i = %.4e)$'
                    % self.mke),
            labelpad=pad)

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
            r'$\log_{10}||\boldsymbol{F}_i(t)||$',
            labelpad=pad)

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
            labelpad=pad)

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

class Polar(_Frame):
    """
    Plotting class specific to 'polar' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        minimum=0,
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
        minimum : float
            Display particles with bond orientation order parameter norm
            greater than this minimum. (default: 0)
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

        self.minimum = minimum

        self.bond = Positions(
            dat.filename, corruption=dat.corruption).getBondOrderParameter(
                frame, *self.particles, arg=False)  # particles' bond order parameter at frame

        self.colorbar(-1, 1, cmap=plt.cm.hsv)   # add colorbar to figure
        self.colormap.set_label(                # colorbar legend
            r'$\mathrm{arg}(\psi_{6,i})/\pi$'
                + ((' ' + r'$(\min |\psi_{6,i}| = %.2f)$' % self.minimum)
                    if self.minimum > 0 else ''),
            labelpad=pad)

        self.label = label  # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, polar, order in zip(
            self.particles, np.angle(self.bond)/np.pi, np.abs(self.bond)):  # for particle and particle's bond orientation real part of the order parameter in rendered box
            self.draw_circle(particle,
                color=self.scalarMap.to_rgba(polar),
                fill=(order >= self.minimum), border='black',
                label=self.label)                               # draw particle circle with color corresponding to bond orientation order parameter norm

class Density(_Frame):
    """
    Plotting class specific to 'density' mode.
    """

    def __init__(self, dat, frame, box_size, centre, a=None,
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
        a : float or None
            Size of the box in which to compute local packing fractions.
            (default: None)
            NOTE: if a == None, then a = dat.L.
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

        # localDensity = np.abs(
        #     Positions(
        #         dat.filename, corruption=dat.corruption).getLocalDensityVoronoi(
        #             frame))
        # self.localDensity = np.array(itemgetter(*self.particles)(localDensity))
        #
        # self.colorbar(0, 1, cmap=plt.cm.tab10)      # add colorbar to figure
        # self.colormap.set_label(                    # colorbar legend
        #     r'$\phi_{v,i}$' + ' '
        #         + r'$(\phi_{v,i}^{\mathrm{min}}/\phi_{v,i}^{\mathrm{max}}=$'
        #         + r'$%.3f)$'
        #             % (localDensity.min()/localDensity.max()),
        #     labelpad=pad)

        localDensity = Positions(dat.filename, corruption=dat.corruption
            ).getLocalParticleDensity(
                frame, dat.L if type(a) == type(None) else a)
        self.localDensity = np.array(itemgetter(*self.particles)(localDensity))

        self.colorbar(0, 1, cmap=plt.cm.tab20)      # add colorbar to figure
        self.colormap.set_label(                    # colorbar legend
            r'$\phi_{i}$' + ' '
                + r'$(a=%.1f, \phi_{i}^{\mathrm{min}}/\phi_{i}^{\mathrm{max}}=$'
                    % a
                + r'$%.3f)$'
                    % (localDensity.min()/localDensity.max()),
            labelpad=pad)

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

class Voronoi(_Frame):
    """
    Plotting class specific to 'voronoi' mode.
    """

    def __init__(self, dat, frame, box_size, centre,
        pad=_colormap_label_pad,
        **kwargs):
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

        self.voronoi = (
            Positions(dat.filename, corruption=dat.corruption)._voronoi(
                frame, centre=centre))

        self.vmin = self.voronoi.volumes.min()
        self.vmax = self.voronoi.volumes.max()
        try:
            if kwargs['vmin'] == None: raise TypeError
            self.vmin = kwargs['vmin']
        except (KeyError, AttributeError, TypeError): pass  # 'vmin' not in keyword arguments or None
        try:
            if kwargs['vmax'] == None: raise TypeError
            self.vmax = kwargs['vmax']
        except (KeyError, AttributeError, TypeError): pass  # 'vmax' not in keyword arguments or None

        self.colorbar(self.vmin, self.vmax, cmap=plt.cm.PiYG)   # add colorbar to figure
        self.colormap.set_label(                                # colorbar legend
            r'$V_i$',
            labelpad=pad)

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        pc = PatchCollection(
            [Polygon(poly[:, :2]) for poly in self.voronoi.polytopes],
            edgecolors='black')
        pc.set_fc(list(map(self.scalarMap.to_rgba, self.voronoi.volumes)))
        pc.set_rasterized(self.rasterized)

        self.ax.add_collection(pc)

class Defects(_Frame):
    """
    Plotting class specific to 'defects' mode.
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

        self.neighbours = (
            Positions(dat.filename, corruption=dat.corruption).getNeighbourList(
                frame))

        self.bond = np.abs(Positions(
            dat.filename, corruption=dat.corruption).getBondOrderParameter(
                frame, *self.particles, arg=False)) # particles' bond order parameter at frame

        self.cmap_norm = ColorsNormalise(vmin=0, vmax=1)
        self.scalarMap = ScalarMappable(norm=self.cmap_norm, cmap=plt.cm.Greys)

        self.label = label  # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle, bond in zip(self.particles, self.bond):   # for particle and hexatic ordere parameter in rendered box
            self.draw_circle(particle,
                color=(
                    'blue' if len(self.neighbours[particle]) == 5
                    else 'red' if len(self.neighbours[particle]) == 7
                    else self.scalarMap.to_rgba(1 - bond)),
                border='black',
                fill=True,
                label=self.label)

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

        # for particle in self.particles:                             # for particle in rendered box
        #     # self.draw_circle(particle, color='black', fill=False,   # draw black circle
        #     #     label=self.label)
        #     # if particle >= self.dat.N - 10:
        #     #     self.draw_circle(particle, color='black', fill=True,
        #     #         label=self.label)
        #     self.draw_circle(particle,
        #         color='red' if particle < self.dat.N/2 else 'blue',
        #         border=None, fill=True, label=self.label)
        #     # self.draw_circle(particle,
        #     #     color='red', border=None, fill=True

        # cpus = get_env('SLURM_CPUS_ON_NODE', default=cpu_count(), vartype=int)
        # with Pool(cpus) as pool:
            # circles = pool.map(
        circles = list(map(
            lambda particle:
                plt.Circle(
                    self.positions[particle], self.diameters[particle]/2,
                    color='black', fill=False,
                    # color='red' if particle < self.dat.N/2 else 'blue',
                    # fill=True,
                    linewidth=self.linewidth,
                    zorder=0, rasterized=self.rasterized),
            self.particles))
        coll = PatchCollection(
            circles,
            edgecolors=list(map(lambda c: c.get_edgecolor(), circles)),
            facecolors=list(map(lambda c: c.get_facecolor(), circles)),
            linewidth=self.linewidth,
            rasterized=self.rasterized)
        self.ax.add_collection(coll)

        if self.label:
            for particle in self.particles:
                self.ax.annotate(
                    "%i" % particle, xy=self.positions[particle], ha="center")

class FRAP(_Frame):
    """
    Plotting class specific to 'frap' mode.
    """

    def __init__(self, dat, frame, box_size, centre, dt=1, n=4,
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
        dt : int
            Lag time for displacement. (default: 1)
        n : int
            Number of checkerboard boxes in each direction. (default: 4)
        label : bool
            Write indexes of particles in circles. (default: False)

        Optional keyword parameters
        ---------------------------
        (see coll_dyn_activem.frame._Frame)
        """

        super().__init__(dat, frame, box_size, centre, **kwargs)    # initialise superclass

        self.fill = np.array([
            (True if (
                (position[0]//(self.box_size/n))%2
                    == (position[1]//(self.box_size/n))%2)
            else False)
            for position in self.dat.getPositions(frame + dt)])

        self.label = label  # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        for particle in self.particles:                                         # for particle in rendered box
            self.draw_circle(particle, color='black', fill=self.fill[particle], # draw black circle and fill according to checkerboard
                label=self.label)

class FRAPc(_Frame):
    """
    Plotting class specific to 'frapc' mode.
    """

    def __init__(self, dat, frame, box_size, centre, dt=1, n=1,
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
        dt : int
            Lag time for displacement. (default: 1)
        n : int
            Number of rainbows. (default: 1)
        label : bool
            Write indexes of particles in circles. (default: False)

        Optional keyword parameters
        ---------------------------
        (see coll_dyn_activem.frame._Frame)
        """

        super().__init__(dat, frame, box_size, centre, **kwargs)    # initialise superclass

        self.cmap_norm = ColorsNormalise(vmin=0, vmax=self.dat.L/n)
        self.scalarMap = ScalarMappable(norm=self.cmap_norm, cmap=plt.cm.hsv)
        self.colors = np.array(list(map(
            lambda position: self.scalarMap.to_rgba(position[0]%(self.dat.L/n)),
            self.dat.getPositions(frame + dt))))
        # positions = relative_positions(
        #     self.dat.getPositions(frame + dt), centre, self.dat.Lxy)
        # self.colors = np.array(list(map(
        #     lambda p: ['white', 'red'][
        #         1*(np.sqrt((p**2).sum(axis=-1)) < 30).all()],
        #     relative_positions(
        #         self.dat.getPositions(frame + dt), centre, self.dat.Lxy))))

        self.label = label  # write labels

        self.draw()

    def draw(self):
        """
        Plots figure.
        """

        circles = list(map(
            lambda particle:
                plt.Circle(
                    self.positions[particle], self.diameters[particle]/2,
                    color=self.colors[particle], fill=True,
                    linewidth=self.linewidth,
                    zorder=0, rasterized=self.rasterized),
            self.particles))
        coll = PatchCollection(
            circles,
            edgecolors=list(map(lambda c: c.get_edgecolor(), circles)),
            facecolors=list(map(lambda c: c.get_facecolor(), circles)),
            linewidth=self.linewidth,
            rasterized=self.rasterized)
        self.ax.add_collection(coll)

        # for particle in self.particles:                                 # for particle in rendered box
        #     self.draw_circle(particle,
        #         color=self.colors[particle], fill=True, border='black', # draw color
        #         label=self.label)

# SCRIPT

if __name__ == '__main__':  # executing as script

    startTime = datetime.now()

    # VARIABLE DEFINITIONS

    mode = get_env('MODE', default='orientation')           # plotting mode
    if mode == 'orientation':
        plotting_object = Orientation
    elif mode == 'displacement':
        plotting_object = Displacement
    elif mode == 'disph':
        plotting_object = DispH
    elif mode == 'odisplacement':
        plotting_object = Odisplacement
    elif mode == 'oridisplacement':
        plotting_object = Oridisplacement
    elif mode == 'movement':
        plotting_object = Movement
    elif mode == 'lemaitrepatinet':
        plotting_object = LemaitrePatinet
    elif mode == 'eigenmode':
        plotting_object = Eigenmode
    elif mode == 'overlap':
        plotting_object = Overlap
    elif mode == 'bond':
        plotting_object = Bond
    elif mode == 'cbond':
        plotting_object = Cbond
    elif mode == 'polar':
        plotting_object = Polar
    elif mode == 'd2min':
        plotting_object = D2min
    elif mode == 'velocity':
        plotting_object = Velocity
    elif mode == 'bvelocity':
        plotting_object = Bvelocity
    elif mode == 'ovelocity':
        plotting_object = Ovelocity
    elif mode == 'orivelocity':
        plotting_object = Orivelocity
    elif mode == 'pvelocity':
        plotting_object = Pvelocity
    elif mode == 'vorticity':
        plotting_object = Vorticity
    elif mode == 'kinetic':
        plotting_object = Kinetic
    elif mode == 'interaction':
        plotting_object = Interaction
    elif mode == 'order':
        plotting_object = Order
    elif mode == 'density':
        plotting_object = Density
    elif mode == 'voronoi':
        plotting_object = Voronoi
    elif mode == 'defects':
        plotting_object = Defects
    elif mode == 'bare':
        if get_env('MIXTURE', default=False, vartype=bool):
            plotting_object = coll_dyn_activem.mixture_pa.Bare
        else:
            plotting_object = Bare
    elif mode == 'frap':
        plotting_object = FRAP
    elif mode == 'frapc':
        plotting_object = FRAPc
    else: raise ValueError('Mode %s is not known.' % mode)  # mode is not known

    dat_file = get_env('DAT_FILE', default=joinpath(getcwd(), 'out.dat'))   # data file
    dat = Dat(dat_file, loadWork=False,
        corruption=get_env('CORRUPTION', default=None, vartype=str))        # data object

    init_frame = get_env('INITIAL_FRAME', default=-1, vartype=int)  # initial frame to render

    dt = get_env('DT', default=-1, vartype=int)                     # displacement lag time (PLOT mode)
    jump = get_env('JUMP', default=1, vartype=int)                  # jump when computing displacements
    a = get_env('A', default=0.3, vartype=float)                    # length scale to compute overlaps or local packing fractions
    a1 = get_env('A1', default=1.15, vartype=float)                 # distance relative to their diameters below which particles are considered bonded
    a2 = get_env('A2', default=1.5, vartype=float)                  # distance relative to their diameters above which particles are considered unbonded
    minimum = get_env('MINIMUM', default=0, vartype=float)          # minimum squared displacement
    n = get_env('N', default=4, vartype=int)                        # number of checkerboard boxes in each direction

    rescale = get_env('RESCALE', default=False, vartype=bool)   # rescale values by root mean square

    box_size = get_env('BOX_SIZE', default=dat.L, vartype=float)    # size of the square box to consider
    centre = (get_env('X_ZERO', default=dat.L/2, vartype=float),
        get_env('Y_ZERO', default=dat.L/2, vartype=float))          # centre of the box

    confinement = get_env('CONFINEMENT', default=False, vartype=bool)   # plot circle of confinement

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

    # subplot layout
    figure_top = get_env('FIGURE_TOP', vartype=float)
    figure_bottom = get_env('FIGURE_BOTTOM', vartype=float)
    figure_left = get_env('FIGURE_LEFT', vartype=float)
    figure_right = get_env('FIGURE_RIGHT', vartype=float)

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

    rasterized = get_env('RASTERIZED', default=False, vartype=bool) # force rasterized (bitmap) drawing for vector graphics output

    colorbar_position = get_env('COLORBAR_POSITION', default='right')   # colorbar position
    colorbar_label = get_env('COLORBAR_LABEL')                          # colorbar label

    # LEGEND SUPTITLE

    display_suptitle = get_env('SUPTITLE', default=True, vartype=bool)      # display suptitle
    time_suptitle = get_env('TIME_SUPTITLE', default=False, vartype=bool)   # display elapsed time since initial frame in suptitle

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

        if time_suptitle:
            if True or type(lag_time) == type(None):
                if dat.Dr != 0:
                    return (r'$t/\tau_p = %.2f$'
                        % ((frame - init_frame)*dat.dt*dat.Dr))
                else:
                    return (r'$t = %.2f$'
                        % ((frame - init_frame)*dat.dt))
            else:
                if dat.Dr != 0:
                    return (r'$\Delta t/\tau_p = %.2f$'
                        % (lag_time*dat.dt*dat.Dr))
                else:
                    return (r'$\Delta t = %.2f$'
                        % (lag_time*dat.dt))

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
        if Dr == 0:
            suptitle += str(r'$t = %.5e$'
                % (frame*dat.dt*dat.dumpPeriod))
        else:
            suptitle += str(r'$D_r t = %.5e$'
                % (frame*dat.dt*dat.dumpPeriod*Dr))
        if lag_time != None:
            if Dr == 0:
                suptitle += str(
                    r'$, \Delta t = %.5e$'
                    % (lag_time*dat.dt*dat.dumpPeriod))
            else:
                suptitle += str(
                    r'$, \Delta t = %.5e, D_r \Delta t = %.5e$'
                    % (lag_time*dat.dt*dat.dumpPeriod,
                        lag_time*dat.dt*dat.dumpPeriod*Dr))

        return suptitle

    # MODE SELECTION

    if get_env('PLOT', default=False, vartype=bool):    # PLOT mode

        if mode in (
            'displacement', 'disph', 'odisplacement', 'oridisplacement',
            'movement', 'lemaitrepatinet', 'overlap', 'bond', 'cbond', 'polar',
            'd2min', 'frap', 'frapc', 'bvelocity'):
            if get_env('NEGATIVE_DT', default=False, vartype=bool): pass
            else:
                try:
                    dt = dt if dt >= 0 else dat.frameIndices[
                        dat.frameIndices.tolist().index(init_frame) + dt]
                except AttributeError:
                    dt = dt if dt >=0 else Nentries - init_frame + dt
        else: dt = None

        figure = plotting_object(dat, init_frame, box_size, centre,
            arrow_width=arrow_width, arrow_head_width=arrow_head_width,
            arrow_head_length=arrow_head_length, arrow_factor=arrow_factor,
            rasterized=rasterized, confinement=confinement,
            colorbar_position=colorbar_position, colorbar_label=colorbar_label,
            pad=pad, dt=dt, jump=jump, a=a, a1=a1, a2=a2, rescale=rescale,
            minimum=minimum, n=n, vmin=vmin, vmax=vmax,
            remove_cm=get_env('REMOVE_CM', vartype=int),
            label=get_env('LABEL', default=False, vartype=bool))
        figure.fig.suptitle(suptitle(init_frame, lag_time=dt))
        figure.fig.subplots_adjust(
            top=figure_top, bottom=figure_bottom,
            left=figure_left, right=figure_right)

        if get_env('TIGHT', default=False, vartype=bool): plt.tight_layout()

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
            if mode in (
                'displacement', 'disph', 'overlap', 'bond', 'cbond', 'frap',
                'movement', 'lemaitrepatinet', 'frapc'):
                if get_env('MOVE', default=True, vartype=bool):
                    remove_cm = init_frame
                    plot_frame = frame
                    lag = init_frame - plot_frame
                else:
                    plot_frame = init_frame
                    lag = frame - plot_frame

            figure = plotting_object(dat, plot_frame, box_size, centre,
                arrow_width=arrow_width, arrow_head_width=arrow_head_width,
                arrow_head_length=arrow_head_length, arrow_factor=arrow_factor,
                confinement=confinement,
                pad=pad, dt=lag, jump=jump, a=a, a1=a1, a2=a2, rescale=rescale,
                minimum=minimum, n=n, vmin=vmin, vmax=vmax,
                colorbar_position=colorbar_position,
                colorbar_label=colorbar_label,
                remove_cm=remove_cm,
                label=get_env('LABEL', default=False, vartype=bool))    # plot frame
            figure.fig.suptitle(suptitle(frame, frame_per))

            tracer = get_env('TRACER', vartype=int)
            if tracer != None:
                if tracer in figure.particles:
                    figure.draw_circle(tracer,
                        color='black', fill=True, border=None,
                        label=False)
                    if get_env('DRAW_PATH', default=False, vartype=bool):
                        try:
                            if (
                                ((figure.positions[tracer] - path[-1])**2
                                    ).sum(axis=-1)
                                > (figure.box_size/2)**2):
                                path += [(np.nan, np.nan)]
                            path += [figure.positions[tracer]]
                        except NameError: path = [figure.positions[tracer]]
                        figure.ax.plot(*np.transpose(path), color='black', lw=4)
                    if get_env('NEIGHBOURS', default=False, vartype=bool):
                        for neighbour, _ in (
                            Positions(figure.dat.filename).getNeighbourList(
                                init_frame)[tracer]):
                            figure.draw_circle(neighbour,
                                color='#0173b2', fill=True, border=None,
                                label=False)
                        for neighbour, _ in (
                            Positions(figure.dat.filename).getNeighbourList(
                                frame)[tracer]):
                            figure.draw_circle(neighbour,
                                color='#de8f05', fill=True, border=None,
                                label=False)

            figure.fig.subplots_adjust(
                top=figure_top, bottom=figure_bottom,
                left=figure_left, right=figure_right)

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
