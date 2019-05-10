# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 21:15:33 2019

@author: brodi
"""
import numpy as np
from sympy import Point3D

from numba import jit

from scipy.spatial.transform import Rotation as R
import quaternion as qt

air = lambda l: 1.

z = [0, 0, 1]

eps = np.finfo(np.float64).eps

@jit(cache=True)
def vecAngle(v0, v1):
    """
    Get the angle between 2 vectors
    """
    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

@jit(cache=True)
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

@jit(cache=True)
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return float(np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))))

@jit(cache=True)
def validDistanceArray(d):
    """Masks potential invalid distances with +inf"""
    d = np.array(d)
    mask = np.logical_or(np.logical_or(np.isnan(d), np.isclose(d, 0.)), d < 0)
    d[mask] = float('inf')
    return d

def rotPlane(a, b, t):
    """Helper function to create orthogonal sets of basis vectors for a plane with 
        sides a,b at angle t to the x axis"""
    return ([a * np.cos(t), a * np.sin(t), 0], [0, 0, b])

@jit(cache=True)
def arbRot(v, o, rotV):
    """
    Arbitraty rotation of v at o about rotV.
    
        Inputs:
            v -- (3, 1) vector, list or array
            o -- center point of rotation
            rotV -- (2,1) vector, passed to quaternion.from_spherical_coords
        Returns
            rotated -- returns rotated vector
    """
    q = qt.from_spherical_coords(rotV)
    rotM = R.from_quat([q.x,q.y,q.z,q.w])
    tranM = np.eye(4)
    tranM[:-1,-1] = -np.array(o)
    
    vec = np.hstack((v, np.ones((v.shape[0], 1))))
    vec = rotM.apply(vec.dot(tranM.T)[:,:3])
    vec = (np.hstack((vec, np.ones((vec.shape[0], 1))))).dot(-tranM.T)
        
    return vec[:,:3]

#@jit(cache=True)
def asCartesian(rthetaphi):
    #takes list rthetaphi (single coord)
    r       = rthetaphi[0]
    theta   = np.radians(rthetaphi[1]) # to radian
    phi     = np.radians(rthetaphi[2])
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.cos(theta)
    z = r * np.sin(theta) * np.sin(phi)

    return np.array([x,y,z])


def asSpherical(xyz):
    #takes list xyz (single coord)
    x       = xyz[0]
    y       = xyz[1]
    z       = xyz[2]
    r       =  np.sqrt(x*x + y*y + z*z)
    phi   =  np.degrees(np.arccos(z/r)) #to degrees
    theta     =  np.degrees(np.arctan2(y,x))
    return np.array([r,theta,phi])

@jit(cache=True)
def isVector(x):
    if isinstance(x, (list, tuple, np.ndarray, Point3D)) and len(x) != 3:
        raise Exception("Invalid 3d vector")
    return(np.array(x))
    
def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

@jit(cache=True)
def normalise(n):
    """Returns the n normalised"""
    if np.linalg.norm(n) == 0:
        raise Exception("Attempt to normalise 0 vector")
    return np.array(n) / np.linalg.norm(n)

@jit(cache=True)
def wavelengthToHex(wl):
    """Calculates an approximation to the RGB representation of a ray with
        a given wavelength"""
    if (wl >= 380 and wl < 440):
        R = -1 * (wl - 440.) / (440. - 380.)
        G = 0
        B = 1
    elif (wl >= 440. and wl < 490.):
        R = 0
        G = (wl - 440.) / (490 - 440.)
        B = 1
    elif (wl >= 490 and wl < 510):
        R = 0
        G = 1
        B = -1 * (wl - 510.) / (510. - 490.)
    elif (wl >= 510 and wl < 580):
        R = (wl - 510.) / (580. - 510.)
        G = 1
        B = 0
    elif (wl >= 580. and wl < 645.):
        R = 1
        G = -1 * (wl - 645.) / (645. - 580.)
        B = 0.0
    elif (wl >= 645 and wl <= 780):
        R = 1
        G = 0
        B = 0
    else:
        R = 0
        G = 0
        B = 0

    if (wl > 780 or wl < 380):
        alpha = 0
    elif (wl > 700):
        alpha = (780. - wl) / (780. - 700.)
    elif (wl < 420):
        alpha = (wl - 380.) / (420. - 380.)
    else:
        alpha = 1
    return (R, G, B)


def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=eps):
    """
    Calculates intersection point of line with plane. If no intersection, return None. 
    """
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        return None # no intersection or line is within plane
 
    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi
 
# Example use
#if __name__=="__main__":
#	#Define plane
#	planeNormal = np.array([0, 0, 1])
#	planePoint = np.array([0, 0, 5]) #Any point on the plane
# 
#	#Define ray
#	rayDirection = np.array([0, -1, -1])
#	rayPoint = np.array([0, 0, 10]) #Any point along the ray
# 
#	Psi = LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint)
#	print ("intersection at", Psi)