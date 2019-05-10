# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 21:45:19 2019

@author: brodi
"""

from abc import ABCMeta, abstractmethod
import copy
import matplotlib.pyplot as plt
import numpy as np

from utils import asCartesian, isVector, normalise, wavelengthToHex, air, \
                  angle_between, unit_vector

class Ray(object):

    """Ray object, access is through subclasses of Source"""

    def __init__(self, p, k, wavelength=550):
        super(Ray, self).__init__()
        self.k_history = []
        self.vertices = isVector(p).reshape((1, 3))
        self.k = isVector(k)        
        self.wavelength = wavelength
        self.isTerminated = False
        self._n = air(self.wavelength)  # Index of refraction of the medium currently in
    
        # TODO make less of hack

    @property
    def p(self):
        """The current point of the ray, i.e. most recent vertex"""
        return self.vertices[-1]

    @property
    def k(self):
        """The current direction vector of the ray"""
        return self._k

    @k.setter
    def k(self, new):
        self.k_history.append(normalise(new))
        self._k = normalise(new)

    @property
    def n(self):
        """The refractive index of the medium the ray is currently travelling in"""
        return self._n

    @n.setter
    def n(self, new):
        self._n = new
        # print "ray.n = %s" %(self._n)
    
    def draw(self, ax = None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_axes([.04,.05,.9,.9])
#        else:
#            ax = fig.get_axes()[0]
        
        ax.plot(self.vertices[:,0], self.vertices[:,1], '.-', color=wavelengthToHex(self.wavelength))
#        return fig

    def append(self, p, k):
        """Update the position and direction of the ray"""
        self.vertices = np.vstack((self.vertices, isVector(p)))
        self.k = isVector(k)
#        print("k = %s, p = %s" % (str(float(self.k)), str(float(self.p))))
        
    def copy(self):
        return copy.deepcopy(self)
    

    
#    def __repr__(self):
#        s = "Ray:\n Vertices: " + str(self.vertices) + "\nVector: " + str(self.k) + "\nMedium: " + str(self.n)
#        return s
    
class Source(object):

    """ABC defining common interface for all source like objects
        
        ray_list - stores all rays associated with this source, exposed for direct access/iteration
        _initRays - populates ray_list 
        _reset - re-populates ray_list """
        
    __metaclass__ = ABCMeta

    def __init__(self, center):
        super(Source, self).__init__()

        self.center = center
        self.ray_list = []

    def __getitem__(self, key):
        """Allows for direct access and iteration over the rays emitted from this source"""
        return self.ray_list[key]
    
    def _reset(self):
        """Returns source to its original state ie before call to Render()"""
        self.ray_list = self._initRays()
        
    @abstractmethod
    def _initRays(self):
        pass
    
    
class SingleRay(Source):

    """Wrapper for a  single ray object
    
       center -  center point of the beam 
       direction -  direction of propagation vector
       wavelength -  wavelength in nm"""

    def __init__(self, center, direction,  wavelength, polar = False):
        super(SingleRay, self).__init__(center)
        if polar:
            self.direction = asCartesian(direction)
        self.direction = isVector(self.direction)
        self.wavelength = wavelength

        self.ray_list = self._initRays()

    def _initRays(self):
        """docstring for _initRays"""
        return([Ray(self.center, self.direction, self.wavelength)])
        
    def closedAngle(self):
        return [r.closedAngle() for r in self.ray_list]