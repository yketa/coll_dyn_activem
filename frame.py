"""
Module frame renders images of the 2D system.

(modified from
https://github.com/yketa/active_particles/tree/master/analysis/frame.py)
"""

from coll_dyn_activem.init import get_env, mkdir
from coll_dyn_activem.read import Dat
from coll_dyn_activem.maths import normalise1D, amplogwidth
from coll_dyn_activem.structure import Positions

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
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

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
        arrow_head_length=_arrow_head_length):
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

        self.positions = dat.getPositions(frame, centre=centre) # particles' positions at frame frame with centre as centre of frame
        self.diameters = dat.diameters                          # particles' diameters

        self.particles = [particle for particle in range(len(self.positions))
            if (np.abs(self.positions[particle]) <= box_size/2).all()]  # particles inside box of centre centre and length box_size

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
                self.box_size/2 - self.diameters[particle]/2):
                newPosition = self.positions[particle].copy()
                newPosition[dim] -= (np.sign(self.positions[particle][dim])
                    *self.box_size)
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
        """

        super().__init__(dat, frame, box_size, centre,
            arrow_width=arrow_width,
            arrow_head_width=arrow_head_width,
            arrow_head_length=arrow_head_length)    # initialise superclass

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
        pad=_colormap_label_pad, dt=1,jump=1,
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
            Lag time for displacement. (default=1)
        jump : int
            Period in number of frames at which to check if particles have
            crossed any boundary. (default: 1)
            NOTE: `jump' must be chosen so that particles do not move a distance
                  greater than half the box size during this time.
        label : bool
            Write indexes of particles in circles. (default: False)

        Optional keyword parameters
        ---------------------------
        vmin : float
            Minimum value of the colorbar.
        vmax : float
            Maximum value of the colorbar.
        """

        super().__init__(dat, frame, box_size, centre,
            arrow_width=arrow_width,
            arrow_head_width=arrow_head_width,
            arrow_head_length=arrow_head_length)    # initialise superclass

        self.displacements = (
            dat.getDisplacements(frame, frame + dt, *self.particles, jump=jump))    # particles' displacements at frame

        self.vmin, self.vmax = amplogwidth(self.displacements)
        try:
            self.vmin = np.log10(kwargs['vmin'])
        except (KeyError, AttributeError, TypeError): pass  # 'vmin' not in keyword arguments or None
        try:
            self.vmax = np.log10(kwargs['vmax'])
        except (KeyError, AttributeError, TypeError): pass  # 'vmax' not in keyword arguments or None

        self.colorbar(self.vmin, self.vmax) # add colorbar to figure
        self.colormap.set_label(            # colorbar legend
            r'$\log_{10} ||\vec{r}_i(t + \Delta t) - \vec{r}_i(t)||$',
            # r'$\log_{10}$'
            #     + r'$||\boldsymbol{r}_i(t + \Delta t) - \boldsymbol{r}_i(t)||$',
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
        vmin : float
            Minimum value of the colorbar.
        vmax : float
            Maximum value of the colorbar.
        """

        super().__init__(dat, frame, box_size, centre,
            arrow_width=arrow_width,
            arrow_head_width=arrow_head_width,
            arrow_head_length=arrow_head_length)    # initialise superclass

        self.velocities = dat.getVelocities(frame, *self.particles) # particles' displacements at frame

        self.vmin, self.vmax = amplogwidth(self.velocities)
        try:
            self.vmin = np.log10(kwargs['vmin'])
        except (KeyError, AttributeError, TypeError): pass  # 'vmin' not in keyword arguments or None
        try:
            self.vmax = np.log10(kwargs['vmax'])
        except (KeyError, AttributeError, TypeError): pass  # 'vmax' not in keyword arguments or None

        self.colorbar(self.vmin, self.vmax) # add colorbar to figure
        self.colormap.set_label(            # colorbar
            r'$\log_{10}||\vec{v}_i(t)||$',
            # r'$\log_{10}||\boldsymbol{v}_i(t)||$',
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
        """

        super().__init__(dat, frame, box_size, centre,
            arrow_width=arrow_width,
            arrow_head_width=arrow_head_width,
            arrow_head_length=arrow_head_length)    # initialise superclass

        self.bondOrder = np.abs(
            Positions(dat.filename).getBondOrderParameter(frame))

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
        """

        super().__init__(dat, frame, box_size, centre,
            arrow_width=arrow_width,
            arrow_head_width=arrow_head_width,
            arrow_head_length=arrow_head_length)    # initialise superclass

        self.localDensity = np.abs(
            Positions(dat.filename).getLocalDensity(frame))

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
        """

        super().__init__(dat, frame, box_size, centre)  # initialise superclass

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
    elif mode == 'velocity':
        plotting_object = Velocity
    elif mode == 'order':
        plotting_object = Order
    elif mode == 'density':
        plotting_object = Density
    elif mode == 'bare':
        plotting_object = Bare
    else: raise ValueError('Mode %s is not known.' % mode)  # mode is not known

    dat_file = get_env('DAT_FILE', default=joinpath(getcwd(), 'out.dat'))   # data file
    dat = Dat(dat_file, loadWork=False)                                     # data object

    init_frame = get_env('INITIAL_FRAME', default=-1, vartype=int)  # initial frame to render

    dt = get_env('DT', default=-1, vartype=int)     # displacement lag time (PLOT mode)
    jump = get_env('JUMP', default=1, vartype=int)  # jump when computing displacements

    box_size = get_env('BOX_SIZE', default=dat.L, vartype=float)    # size of the square box to consider
    centre = (get_env('X_ZERO', default=0, vartype=float),
        get_env('Y_ZERO', default=0, vartype=float))                # centre of the box

    Nentries = dat.frames - 1
    init_frame = int(Nentries/2) if init_frame < 0 else init_frame    # initial frame to draw

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

        Nframes = Nentries - init_frame  # number of frames available for the calculation
        if mode == 'displacement': dt = Nframes + dt if dt < 0 else dt
        else: dt = None

        figure = plotting_object(dat, init_frame, box_size, centre,
            arrow_width=arrow_width, arrow_head_width=arrow_head_width,
            arrow_head_length=arrow_head_length, pad=pad, dt=dt, jump=jump,
            vmin=vmin, vmax=vmax,
            label=get_env('LABEL', default=False, vartype=bool))
        figure.fig.suptitle(suptitle(init_frame, lag_time=dt))

        if get_env('SAVE', default=False, vartype=bool):    # SAVE mode
            figure_name = get_env('FIGURE_NAME', default='out')
            figure.fig.savefig(figure_name + '.eps')
            figure.fig.savefig(figure_name + '.svg')

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

            figure = plotting_object(dat, frame, box_size, centre,
                arrow_width=arrow_width, arrow_head_width=arrow_head_width,
                arrow_head_length=arrow_head_length, pad=pad, dt=frame_per,
                jump=jump, vmin=vmin, vmax=vmax,
                label=get_env('LABEL', default=False, vartype=bool))    # plot frame
            figure.fig.suptitle(suptitle(frame, frame_per))

            figure.fig.savefig(joinpath(movie_dir, 'frames',
                '%010d' % frames.index(frame) + '.png'))    # save frame
            del figure                                      # delete (close) figure

        subprocess.call([
            'ffmpeg', '-r', '5', '-f', 'image2', '-s', '1280x960', '-i',
            joinpath(movie_dir , 'frames', '%10d.png'),
            '-pix_fmt', 'yuv420p', '-y',
            joinpath(movie_dir, get_env('FIGURE_NAME', default='out.mp4'))
            ])  # generate movie from frames

    # EXECUTION TIME
    print("Execution time: %s" % (datetime.now() - startTime))

    if get_env('SHOW', default=False, vartype=bool):    # SHOW mode
        plt.show()
