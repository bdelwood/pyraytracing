# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 21:29:52 2019

@author: brodi
"""
from functools import partial
from abc import ABCMeta, abstractmethod
from sympy import Plane, Point3D
from matplotlib.patches import Polygon, Circle, Arrow, Ellipse
import matplotlib.pyplot as plt
import numpy as np
eps = np.finfo(np.float64).eps

from interactions import Reflection, Transmission, Absorption
from utils import validDistanceArray, isVector, asCartesian, \
                  asSpherical, normalise, z, arbRot

import sympy as sym

from scipy.spatial.transform import Rotation as R
import quaternion as qt
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore", message="Setting the 'color' property will override") 


def hitsMirror(mirror, p, d):
    if sym.Point3D(mirror.center).distance(sym.Point3D(p)) > mirror.arad or d == 0:
        return False
    else:
        return True

class OpticalElement(object):

    """Abstract base class defining the common interface that all optical elements
        must implement and subclass.
        Also defines draw() which should be called first in all derived draw
        implementations to handle setting up new axes if none are provided"""
    __metaclass__ = ABCMeta
    

    def __init__(self, center, draw_kwargs={}, name = None):
        super(OpticalElement, self).__init__()
        self.draw_kwargs = {'closed':True, 'facecolor':'none', 'edgecolor':'r'}
        self.draw_kwargs.update(draw_kwargs)
        self.center = center
        self.polygon = partial(Polygon, **self.draw_kwargs)
        self.name = name
        
#        self.init()
    
    @property
    @abstractmethod
    def center(self):
        return self._center
    
    @center.setter
    @abstractmethod
    def center(self, val):
        self._center = isVector(val)   
        self._set()
    
    @abstractmethod
    def _set(self):
        pass
    
    @abstractmethod
    def _reset(self):
        pass
        
    @abstractmethod
    def distance(self):
        pass

    @abstractmethod
    def propagate_ray(self):
        pass
    

    @abstractmethod
    def draw(self, ax = None):
        if ax is None:
            fig, ax = plt.subplots()
        return ax
    

class OpticalPlane(OpticalElement):
    def __init__(self, center, a = [0,0,0], b = z, polar = False, draw_kwargs = {}, passThru = None, **kwargs):
        super().__init__(center, draw_kwargs, **kwargs)
        self.a = a if not polar else asCartesian(a)
        self.b = b
        
        self._passes = passThru
        self.passes = passThru
        
    def _set(self):
        if all(hasattr(self, attr) for attr in ['a','b','center']):
            self.normal = normalise(np.cross(self.a,self.b))
            
            self.Plane = Plane(Point3D(self.center), tuple(self.normal))
            
            self.points = np.array([self.center + self.a + self.b,
                                    self.center - self.a + self.b,
                                    self.center - self.a - self.b,
                                    self.center + self.a - self.b])
            
            self.vertices = np.array([(self.center + self.a)[:-1], (self.center - self.a)[:-1]])
            
    def _reset(self):
        self.passes = self._passes
        
    @property
    def a(self):
        return self._a
    
    @a.setter
    def a(self, val):
        self._a = isVector(val)
        self._set()
    
    @property
    def b(self):
        return self._b
    
    @b.setter
    def b(self, val):
        self._b = isVector(val)
        self._set()
    
    def draw(self, ax = None):
        ax = super(OpticalPlane, self).draw(ax)
        poly = self.polygon([x[:2] for x in self.points]) # only takes first 2 coords
        ax.add_patch(poly)
        ax.autoscale_view()

    
    def distance(self, ray):
        """Distance from ray to self along the ray's path"""
        r = self.center - ray.p
        top = np.dot(self.normal, r)
        bottom = np.dot(ray.k, self.normal)
        if np.abs(top) < eps or np.abs(bottom) < eps:
            return float('inf')

        d = top / bottom
        p = ray.p + d * ray.k
        d = d if self.isIntersect(p) else float('inf')
        if self.passes:
            self.passes -= 1
            return float('inf')
        else:
            return d

    def isIntersect(self, p):
        """Decides if p is in the plane or not by projecting the intersection point
            on the infinite plane onto the a,b axis than span the finite plane
            and seeing if either component is >1"""
        alpha = np.dot(self.a, (p - self.center)) / np.linalg.norm(self.a)**2
        beta = np.dot(self.b, (p - self.center)) / np.linalg.norm(self.b)**2

        return np.logical_and(np.abs(alpha) <= 1, np.abs(beta) <= 1)
    
class TestShape(OpticalPlane):
    
    def __init__(self, center, a = [0,0,0], b = z, polar = False, draw_kwargs = {}, **kwargs):
        super().__init__(center, a, b, polar, draw_kwargs, **kwargs)
        self.draw_kwargs.pop('closed')
        
    def draw(self, ax = None):
        ax = super(OpticalPlane, self).draw(ax)
        ell = partial(Ellipse, **self.draw_kwargs)
        poly = ell(np.mean([self.points[2][:2],self.points[0][:2]], axis = 0), 10, 5)
        ax.add_patch(poly)
        ax.autoscale_view()
        

class Mirror(OpticalPlane, Reflection):

    """A totally reflective plane (both sides)"""

    def propagate_ray(self, ray):
        """
        Implements propagate_ray for Mirror by reflecting the wavevector of the ray
            about the normal of the plane.
        """

        return self._reflect(ray, self.normal)

class Window(OpticalPlane, Transmission):
    """A transparent plane"""
    
    def propagate_ray(self, ray):
        """Implements propagate_ray for Window by transmitting the ray"""

        return self._transmit(ray, self.normal)
    
    
class Wall(OpticalPlane, Absorption):

    """A plane with total absorption"""
    
    def __init__(self, center, a = [0,0,0], b = z, polar = False, draw_kwargs = {}):
        draw_kwargs.update({'edgecolor':'grey'})
        super().__init__(center,  a, b, polar, draw_kwargs)

    def propagate_ray(self, ray):
        """Implements propagate_ray for Wall by terminating ray"""
        return self._absorb(ray)

class CompositeObject(OpticalElement):

    """docstring for CompositeObject"""

    def __init__(self, centre, draw_kwargs={}, **kwargs):
        super(CompositeObject, self).__init__(centre, **kwargs)
        self.components = []
        self.interactions = defaultdict(list)

    def add(self, comp):
        """Add a new component to the composite object"""
        if not isinstance(comp, OpticalElement):
            raise Exception("Component must be an optical element!")

        self.components.append(comp)
        
    def distance(self, ray):
        """Distance of the composite is the min distance of its components"""
        d = np.array([s.distance(ray) for s in self.components])
        d = validDistanceArray(d)
        
#        distances = validDistanceArray(distances)
#        closest = self.components[np.argmin(d)]
#        updated_ray = closest.propagate_ray(ray)

        return np.min(d)
    
#    def closest(self, ray):
#        d = self.distance(ray)
#        closest = self.components[np.argmin(d)]

    def propagate_ray(self, ray):
        """Pass on propagate_ray to the component closest to the ray"""
        d = np.array([s.distance(ray) for s in self.components])
        d = validDistanceArray(d)

        new_ray = self.components[np.argmin(d)].propagate_ray(ray)
        return new_ray

    def draw(self, ax=None):
        """docstring for draw"""
        ax = super(CompositeObject, self).draw(ax)
        for s in self.components:
            s.draw(ax)

        return ax
    
    def _reset(self):
        [c._set() for c in self.components]
#    
#    def _set(self):
#        [c._reset for c in self.components]
    

class RotationStage(CompositeObject):
    def __init__(self, center, radius, draw_kwargs = {}, **kwargs):
        super().__init__(center, draw_kwargs, **kwargs)
        self.radius = radius
    
    def draw(self, ax = None):
        ax = super().draw(ax)
        circ = Circle(self.center, self.radius, edgecolor = 'grey', linestyle = '--', fill = False, zorder = 0)
        ax.scatter(*self.center[:2], marker = '+', color = 'lightskyblue', s = 300)
        ax.add_patch(circ)
        
        return ax

    def rotate(self, xyz):
#        q = qt.from_rotation_vector([x,y,z])
#        rotM = R.from_quat([q.x,q.y,q.z,q.w])
        rotM = R.from_rotvec(xyz)
        tranM = np.eye(4)
        tranM[:-1,-1] = np.asmatrix(-np.array(self.center))

        for s in self.components:
            cent = s.center.reshape(1,3)
            cent = np.hstack((cent, np.ones((cent.shape[0], 1))))
            cent = rotM.apply(cent.dot(tranM.T)[:,:3])
            cent = (np.hstack((cent, np.ones((cent.shape[0], 1))))).dot(np.linalg.inv(tranM).T)[:,:3][0]

            s.center = cent
            vec = rotM.apply(np.array([s.a,s.b]).reshape(2,3))
            s.a = vec[0,:]
            s.b = vec[1,:]
    
            
        
            