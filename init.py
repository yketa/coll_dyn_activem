"""
Module init provides functions useful when initialising simulations or
analysis.

(adapted from https://github.com/yketa/active_particles/tree/master/init.py)
"""

from os.path import join as joinpath
from os.path import exists as pathexists
from os import makedirs
from os import environ as envvar
from shutil import rmtree as rmr

import sys

import atexit

import pickle

from collections import OrderedDict

from numbers import Number

import numpy as np

from datetime import datetime
from pytz.reference import LocalTimezone as timezone

def to_vartype(input, default=None, vartype=str):
    """
    Returns input converted to vartype or default if the conversion fails.

    Parameters
    ----------
    input : *
        Input value (of any type).
    default : *
        Default value to return if conversion fails.
    vartype : data type
        Desired data type. (default: string)

    Returns
    -------
    output : vartype or type(default)
        Input converted to vartype data type or default.
    """

    if vartype == bool and input == 'False': return False   # special custom case

    try:
        try:
            return vartype(input)
        except ValueError: return vartype(eval(input))
    except: return default

def set_env(var_name, var_value):
    """
    Sets environment variable var_name value to str(var_value).

    Parameters
    ----------
    var_name : string
        Environment variable name.
    var_value : *
        Environment variable value.
        NOTE: if var_value=None, then the environment variable is unset
    """

    if var_value == None:
        if var_name in envvar: del envvar[var_name]
    else: envvar[var_name] = str(var_value)

def get_env(var_name, default=None, vartype=str):
    """
    Returns environment variable with desired data type.

    WARNING: get_env function uses eval function to evaluate environment
    variable strings if necessary, therefore extra cautious is recommended when
    using it.

    Parameters
    ----------
    var_name : string
        Name of environment variable.
    default : *
        Default value to return if environment variable does not exist or if
        conversion fails. (default: None)
    vartype : data type
        Desired data type. (default: string)

    Returns
    -------
    var : vartype or type(default)
        Environment variable converted to vartype data type of default.
    """

    try:
        return to_vartype(envvar[var_name], default=default, vartype=vartype)
    except: return default

def get_env_list(var_name, delimiter=':', default=None, vartype=str):
    """
    Returns list from environment variable containing values delimited with
    delimiter to be converted to vartype data type or taken to be default if
    the conversion fails.
    NOTE: Returns empty list if the environment variable does not exist or is
    an empty string.

    Parameters
    ----------
    var_name : string
        Name of environment variable.
    delimiter : string
        Pattern which delimits values to be evaluated in environment variable.
    default : *
        Default value to return if individual value in environment variable
        does not exist or if conversion fails. (default: None)
    vartype : data type
        Desired data type. (default: string)

    Returns
    -------
    var_list : list of vartype of type(default)
        List of individual environment variables values converted to vartype
        data type or default.
    """

    if not(var_name in envvar) or envvar[var_name] == '': return []
    return list(map(
        lambda var: to_vartype(var, default=default, vartype=vartype),
        envvar[var_name].split(delimiter)
        ))

class StdOut:
    """
    Enables to set output stream to file and revert this setting.
    """

    def __init__(self):
        """
        Saves original standard output as attribute.
        """

        self.stdout = sys.stdout    # original standard output

    def set(self, output_file):
        """
        Sets output to file.

        Parameters
        ----------
        output_file : file object
            Output file.
        """

        try:
            self.output_file.close()    # if output file already set, close it
        except AttributeError: pass

        self.output_file = output_file  # output file
        sys.stdout = self.output_file   # new output stream

        atexit.register(self.revert)    # close file when exiting script

    def revert(self):
        """
        Revers to original standard output.
        """

        try:
            self.output_file.close()
            sys.stdout = self.stdout    # revert to original standart output
        except AttributeError: pass     # no custom output was set

def mkdir(directory, replace=False):
    """
    Creates directory if not existing, erases and recreates it if replace is
    set to True.

    Parameters
    ----------
    directory : string
        Name of directory.
    replace : bool
        Erase and recreate directory. (default: False)
    """

    if pathexists(directory) and replace: rmr(directory)
    makedirs(directory, exist_ok=True)

def isnumber(variable):
    """
    Returns True if variable is a number, False otherwise.

    Parameters
    ----------
    variable : *
        Variable to check.

    Returns
    -------
    variableisnumber : bool
        Is variable a number?
    """

    return isinstance(variable, Number)

def linframes(init_frame, tot_frames, max_frames):
    """
    Returns linearly spaced indexes in [|init_frame; tot_frames - 1|], with a
    maximum of max_frames indexes.

    Parameters
    ----------
    init_frame : int
        Index of initial frame.
    tot_frames : int
        Total number of frames.
    max_frames : int
        Maximum number of frames.

    Returns
    -------
    frames : 1D Numpy array
        Array of frame numbers in ascending order.
    """

    return np.array(list(OrderedDict.fromkeys(map(
        int,
        np.linspace(init_frame, tot_frames - 1, max_frames, dtype=int)
        ))))

class Time:
    """
    Get initial, final and elapsed times.
    """

    class _Time(datetime):
        """
        Subclass of datetime.datetime with different string syntax.
        """

        def __str__(self):
            """
            Returns string of date as:
                "Weekday Day Month Hour:Minutes:Seconds Timezone Year"

            Returns
            -------
            time : str
                Time.
            """

            return self.strftime(
                "%a %d %b %H:%M:%S {} %Y".format(timezone().tzname(self)))

    def __init__(self):
        """
        Sets initial time.
        """

        self.initial_time = self._Time.now()

    def end(self):
        """
        Sets final and elapsed times.
        """

        self.final_time = self._Time.now()
        self.elapsed_time = self.final_time - self.initial_time

    def getInitial(self):
        """
        Returns initial time.

        Returns
        -------
        initial : str
            Initial time.
        """

        return str(self.initial_time)

    def getFinal(self):
        """
        Returns final time.

        NOTE: Final time is set by calling self.end(). If self.getFinal() is
              called before, it will first call self.end().

        Returns
        -------
        final : str
            Final time.
        """

        try: self.final_time
        except AttributeError: self.end()

        return str(self.final_time)

    def getElapsed(self):
        """
        Returns elapsed time.

        NOTE: Final time is set by calling self.end(). If self.getElapsed() is
              called before, it will first call self.end().

        Returns
        -------
        elapsed : str
            Elapsed time.
        """

        try: self.final_time
        except AttributeError: self.end()

        return str(self.elapsed_time)
