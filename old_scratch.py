# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 17:45:50 2019

@author: brodi
"""
from functools import partial
from matplotlib.patches import Polygon
import threading
import queue


#    def rotate(self, th, phi):
#        q = qt.from_spherical_coords([th, phi])
#        rotM = R.from_quat([q.x,q.y,q.z,q.w])
#        tranM = np.eye(4)
#        tranM[:-1,-1] = -np.array(self.center)
#
#        for i, s in enumerate(self.components):
#            points = np.hstack((s.points, np.ones((s.points.shape[0], 1))))
#            points = rotM.apply(points.dot(tranM.T)[:,:3])
#            points = (np.hstack((points, np.ones((points.shape[0], 1))))).dot(-tranM.T)
#            
#            s.points = points[:,:3]

#   Found out why the roatations aren't doing anything to rays: need to rebuild 
#   object, not merely edit its points; the points don't factor into dist calculations
#   perform the transformation on all input vectors (center, a, b), and then rebuild the object
#   compare with old method via graph 

def read_kbd_input(inputQueue):
    print('Angle: ')
    while (True):
        input_str = input()
        inputQueue.put(input_str)
        
#inputQueue = queue.Queue()
#inputThread = threading.Thread(target=read_kbd_input, args=(inputQueue,), daemon=True)
#inputThread.start()
#if (inputQueue.qsize() > 0):
#    th = float(inputQueue.get())
        
draw_kwargs = {'closed':False, 'facecolor':'none', 'edgecolor':'r'}
polygon = partial(Polygon, **draw_kwargs)
#xs = [np.array([100, 100]), np.array([100,   0]), np.array([100,   0]), np.array([100, 100])]
#fig, ax = plt.subplots()
#poly = polygon(np.array([x[:2] for x in xs]))
#verts = [(a,0), (b,0)]
#poly = Polygon(verts, facecolor='0.9', edgecolor='none', closed = False)
#poly = polygon(xs)
#ax.add_patch(poly)
#ax.autoscale_view()
#fig.show()

#plt.show()
