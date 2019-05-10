# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 21:30:05 2019

@author: brodi
"""
import numpy as np
import sympy as sym

#def reflection(mirror, ray):
#    '''Reflects an incident ray, ray in the plane with normal n'''
#    p = mirror.Plane.intersection(sym.Ray(ray.p, ray.k))[0]
#    k_prime = ray.k - 2 * np.dot(ray.k, mirror.normal) * mirror.normal
#    
#    if hitsMirror(mirror, p, d):
#        d = mirror.Plane.distance(sym.Point3D(ray.p))
#        ray.append(p, k_prime)
#        return d, ray
#
#def transmission(mirror, ray):
#    p = mirror.Plane.intersection(sym.Ray(ray.p, ray.k))[0]
#
#    if hitsMirror(mirror, p):
#        d = mirror.Plane.distance(sym.Point3D(ray.p))
#        ray.append(p, ray.k)
#        return d, ray

class Reflection(object):
    """Reflection provides geometry independent reflection"""
    def _reflect(self, ray, n):
        '''Reflects an incident ray, ray in the plane with normal n'''
        d = self.distance(ray)
        p = ray.p + d * ray.k
        k_prime = ray.k - 2 * np.dot(ray.k, n) * n
        ray.append(p, k_prime)
        ray.isTerminated = False
        return ray

class Transmission(object):
    """Transmission transmits rays through object"""
    def _transmit(self, ray, n):
        d = self.distance(ray)
        p = ray.p + d * ray.k
        k_prime = ray.k
        ray.append(p, k_prime)
        ray.isTerminated = False
        return ray

class Absorption(object):
    """Terminates rays"""
    def _absorb(self, ray):
        d = self.distance(ray)
        ray.append(ray.p + d * ray.k, ray.k)
        ray.isTerminated = True
        return ray
    
