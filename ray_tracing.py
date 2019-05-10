# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 00:32:24 2019

@author: brodi
"""
from elements import OpticalElement, Mirror, Wall, RotationStage
from sources import Source

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from numba import jit

from collections import defaultdict
import scipy.spatial as spatial
    
from utils import angle_between

@jit(cache = True)
def validDistanceArray(d):
    """Masks potential invalid distances with +inf"""
    d = np.array(d)
    mask = np.logical_or(np.logical_or(np.isnan(d), np.isclose(d, 0.)), d < 0)
    d[mask] = float('inf')
    return d

       

class Bench(object):
    def __init__(self, x, y, z, render_limit = 20):
        self.x = x
        self.y = y
        self.z = z
        
        self.boundary_list = self._makeBoundaries()
        self.source_list = []
#        self.screen_list = []
        self.element_list = []
        
        self.rotation_list = []
        
        self.interactions = defaultdict(list)

        self.render_limit = render_limit
    def _makeBoundaries(self):
        """Private function. Builds the edges of the simulation area"""

        x = self.x
        y = self.y
        z = self.z

        return [Wall([2 * x, y, z], [0, y, 0], [0, 0, z]),
                Wall([0, y, z], [0, y, 0], [0, 0, z]),
                Wall([x, 2 * y, z], [x, 0, 0], [0, 0, z]),
                Wall([x, 0, z], [x, 0, 0], [0, 0, z]),
                Wall([x, y, 2 * z], [x, 0, 0], [0, y, 0]),
                Wall([x, y, 0], [x, 0, 0], [0, y, 0]), ]
        

        
    def _trace(self, ray):
        """Private. Computes the distance along the ray to each interactor
            then propagates the ray through the one it hits first"""

        distances = []
        for obj in self.interactors:
#            if isinstance(obj, RotationStage):
#                print('here')
            distances.append(obj.distance(ray))
        distances = np.array(distances)
#        if self.verbose:
#            print "Distances: " + str(distances)
        # Assert that distances must be >= 0 and not nan
        distances = validDistanceArray(distances)
        closest = self.interactors[np.argmin(distances)]
        updated_ray = closest.propagate_ray(ray)
        if closest.name:
            self.interactions[closest.name].append(ray.copy())
#            print(ray.p)
        return updated_ray
    
    def _propogate(self, ray):
        pass
    
    def Render(self):
        """Main render loop to perform ray tracing"""
#        rot_list = []
#        for i, elem in enumerate(self.element_list):
#            if type(elem) == RotationStage:
#                self.element_list.pop(i)
#                rot_list += elem.components

        self.interactors = self.boundary_list + self.element_list #self.screen_list + self.element_list
        self.reset()
        for source in self.source_list:
#            source._reset()
            for ray in source:
                i = 0
                while ray.isTerminated is False:
                    if i > self.render_limit:
#                        raise Exception("Render limit exceeded")
                        break
#                    if self.verbose:
#                        print "\n"
#                        print "Loop " + str(i)
#                        print ray
                    ray = self._trace(ray)
                    i += 1
#                if self.verbose:
#                    print "*" * 30
#        [self.interactions.update(d.interactions) if hasattr(d, 'interactions') else None for d in self.interactors]
            
    def add(self, element):
        """Adds an optical element object to the bench"""
        if isinstance(element, Source):
            self.source_list.append(element)
#        elif isinstance(element, Screen):
#            self.screen_list.append(element)
        elif isinstance(element, OpticalElement) and not isinstance(element, RotationStage):
            self.element_list.append(element)
        elif isinstance(element, RotationStage):
            self.element_list += element.components
            self.rotation_list.append(element)
        else:
            raise Exception("Object must be subclassed from OpticalElement or Source")
    
    def removeSources(self):
        self.source_list = []
        self.interactions.clear()
        
    def reset(self):
        """Calls individual reset functions"""
        [s._reset() for s in self.source_list]
        [e._reset() for e in self.element_list]
        [r._reset() for r in self.rotation_list]
            
    def closedPathDist(self, name):
        if len(self.interactions[name]) < 2:
            dist =  np.nan
        else:
            p1 = self.interactions[name][0].p.reshape(1, -1)
            p2 = self.interactions[name][-1].p.reshape(1,-1)
            dist = spatial.distance.cdist(p1, p2)
        
        return dist

    def rayAngle(self, n1, n2, rays = (-1, -1)):
        """ Give a named interactor and a ray number, defults to last """
        if len(self.interactions[n1]) ==0 or self.interactions[n2] == 0:
            return np.nan
        else:
            k1 = self.interactions[n1][rays[0]].k
            k2 = self.interactions[n2][rays[1]].k
            return angle_between(k1, k2)
    
    def optimize(self):
        raise NotImplementedError
        

    def draw(self, ax = None):
        """Draws a xy projection of the OpticalBench. Generally call Render() first to
            populate with rays, although can be called before render to check the placement of objects etc"""
        if ax is None:
            fig = plt.figure()
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
            ax.set_xlim(-0.4 * self.x, 2.4 * self.x)
            ax.set_ylim(-0.4 * self.y, 2.4 * self.y)
            ax.set_aspect('equal')

        for b in self.boundary_list[:4]:
            b.draw(ax)

        for e in self.element_list[:4]:
            e.draw(ax)
            
        for r in self.rotation_list[:4]:
            r.draw(ax)

#        for e in self.screen_list:
#            e.draw(ax)
#            e.fig.show()

        for s in self.source_list:
            for r in s:
                r.draw(ax)

        return ax



#r1 = SingleRay([-1,1,0],[-6,100,0])   
#
#l = 100
#s = 6
#mL = 2.54
#
#m1 = Mirror([0,0,0], [mL,45,0], polar=True)
#m2 = Mirror([l,-s,0],[mL,85,0], polar=True)
#m3 = Mirror([l,0,0],[mL,300,0], polar=True)
#m4 = Mirror([0,-s,0],[mL,120,0], polar=True)

#m1.transmit(r1)
#rays = [m.interact(r1) for m in [m2,m3,m4]]
##m1.transmit(r1)
#fig = m1.draw()
#fig = m2.draw(fig = fig)
#fig = m3.draw(fig = fig)
#fig = m4.draw(fig = fig)
#fig = r1.draw(fig = fig)
#figs = [ray.draw() for ray in rays]
#for ray in rays:
#    fig = ray[0].draw(fig = fig)
#    fig = ray[1].draw(fig = fig)
#fig.show()
#