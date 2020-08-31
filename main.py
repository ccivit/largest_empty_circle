# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:58:48 2019

@author: ccivit
"""
import matplotlib.pyplot as plt
from math import pi
import numpy as np
from numpy import cos,sin
import itertools



a = np.array([-200,0])
b = np.array([0,300])
c = np.array([200,0])
f = np.array([200,-100])

def mid(a,b):
    return (a + b)/2

def dist(a,b):
    return np.linalg.norm(a-b)

def rad(a,b):
    return dist(a,b)

def norm_v(a,b):
    return (a-b)/dist(a,b)

def point_is_inside_circle(p,center,radius):
    return dist(p,center) < radius * 0.999999
#    return (p[0] - center[0])**2 + (p[1] - center[0])**2 < radius**2

def moss_egg(a,b,c,t):
    # euclidean egg, from https://en.wikipedia.org/wiki/Moss%27s_Egg
    # find d and e
    d = c + norm_v(b,c)*rad(a,c) 
    e = a + norm_v(b,a)*rad(a,c)
    
    if t >= 0.5:
        x = mid(a,c)[0] + 0.5*rad(a,c)*cos(2*pi*t)
        y = mid(a,c)[1] + 0.5*rad(a,c)*sin(2*pi*t)
    elif t < 0.5:
        x = a[0] + rad(a,c)*cos(2*pi*t)
        y = a[1] + rad(a,c)*sin(2*pi*t)
        if x < e[0]:
            x = b[0] + rad(b,e)*cos(2*pi*t)
            y = b[1] + rad(b,e)*sin(2*pi*t)
        if x < d[0]:
            x = c[0] + rad(c,d)*cos(2*pi*t)
            y = c[1] + rad(c,d)*sin(2*pi*t)            
    return np.array([x,y])

def define_circle(p1, p2, p3):
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)
    

def sample_cloud():
    XY = np.array([[0.068007,0.028853],
                   [0.06118,0.039598],
                   [0.05819,0.048729],
                   [0.056347,0.060423],
                   [0.056954,0.072519],
                   [0.060822,0.083259],
                   [0.066346,0.091569],
                   [0.0742,0.098495],
                   [0.084365,0.10486],
                   [0.093805,0.108234],
                   [0.104132,0.109049],
                   [0.113885,0.106744],
                   [0.122792,0.103579],
                   [0.131627,0.098305],
                   [0.139191,0.091442],
                   [0.145915,0.082297],
                   [0.150513,0.072264],
                   [0.152981,0.062045],
                   [0.152886,0.051365],
                   [0.149622,0.041181],
                   [0.14415,0.031515],
                   [0.13715,0.023157],
                   [0.128693,0.016971],
                   [0.118273,0.012602],
                   [0.107582,0.010847],
                   [0.096283,0.011264],
                   [0.084782,0.015423],
                   [0.075385,0.021257]])
    return XY



def generate_non_circle(a,b,c,f):
    step_size = 0.05
    XY = np.zeros((int(1/step_size),2))
    t_range = np.arange(0,1,step_size)
    for i,t in enumerate(t_range):
        XY[i] = moss_egg(a,b,c,t)
    return XY

def generate_circle(center,radius):
    step_size = 0.005
    circle = np.zeros((int(1/step_size),2))
    t_range = np.arange(0,1,step_size)
    for i,t in enumerate(t_range):
        circle[i] = np.array([center[0] + radius*cos(2*pi*t),
                              center[1] + radius*sin(2*pi*t)])
    return circle

# from https://gist.github.com/bert/1188638

#import numpy
#from PIL import Image

#def voronoi(points,shape=(500,500)):
#    depthmap = np.ones(shape,numpy.float)*1e308
#    colormap = np.zeros(shape,numpy.int)
#
#    def hypot(X,Y):
#        return (X-x)**2 + (Y-y)**2
#
#    for i,(x,y) in enumerate(points):
#        paraboloid = numpy.fromfunction(hypot,shape)
#        colormap = numpy.where(paraboloid < depthmap,i+1,colormap)
#        depthmap = numpy.where(paraboloid <
#depthmap,paraboloid,depthmap)
#
#    for (x,y) in points:
#        print(x)
#        print(y)
#        x = int(x)
#        y = int(y)
#        colormap[x-1:x+2,y-1:y+2] = 0
#
#    return colormap

def distance_to_point_set(array_of_points,point2):
    distances = []
    for point in array_of_points:
        distances.append(dist(point,point2))
    return distances

def top_smallest(numbers,nb_of_top=3):
    # Returns index of the 3 smallest items in an list of numbers
    m1_v, m2_v, m3_v = float('inf'), float('inf'), float('inf')
    m1_i, m2_i, m3_i = float('inf'), float('inf'), float('inf')
    for i,x in enumerate(numbers):
        if x <= m1_v:
            m1_i, m2_i = i, m1_i
            m1_v, m2_v = x, m1_v
        elif x < m2_v:
            m2_v = x
            m2_i = i
        elif x < m3_v:
            m3_v = x
            m3_i = i
    return m1_i,m2_i,m3_i

#def draw_voronoi_map(colormap):
#    shape = colormap.shape
#
#    palette = numpy.array([
#            0x000000FF,
#            0xFF0000FF,
#            0x00FF00FF,
#            0xFFFF00FF,
#            0x0000FFFF,
#            0xFF00FFFF,
#            0x00FFFFFF,
#            0xFFFFFFFF,
#            0x000000FF,
#            0xFF0000FF,
#            0x00FF00FF,
#            0xFFFF00FF,
#            0x0000FFFF,
#            0xFF00FFFF,
#            0x00FFFFFF,
#            0xFFFFFFFF,
#            ])
#
#    colormap = numpy.transpose(colormap)
#    pixels = numpy.empty(colormap.shape+(4,),numpy.int8)
#
#    pixels[:,:,3] = palette[colormap] & 0xFF
#    pixels[:,:,2] = (palette[colormap]>>8) & 0xFF
#    pixels[:,:,1] = (palette[colormap]>>16) & 0xFF
#    pixels[:,:,0] = (palette[colormap]>>24) & 0xFF
#
#    image = Image.frombytes("RGBA",shape,pixels)
#    image.save('voronoi.png')



center= [0.1,0.05]
radius = 0
#print(point_is_inside_circle(a,center,radius))


XY = generate_non_circle(a,b,c,f)
XY = sample_cloud()
#print(XY)
print(len(XY),'points')
print(np.math.factorial(len(XY)), 'iterations')



#draw_voronoi_map(voronoi(XY))
    
perm = itertools.permutations(range(len(XY)),3) 
  
for n,i in enumerate(list(perm)):
    new_center, new_radius = define_circle(XY[i[0]],XY[i[1]],XY[i[2]])
    #if i == (0, 14, 20):
        #print(new_center,new_radius)
        #print(XY[i[0]])
        #print(XY[i[1]])
        #print(XY[i[2]])
    if new_radius > radius:
        p_inside = False
        for p_i,p in enumerate(XY):
            p_inside = p_inside or point_is_inside_circle(p,new_center,new_radius)
            if  i == (0, 14, 20):
                if True:
                    print(p_i, point_is_inside_circle(p,new_center,new_radius))
                
        if not p_inside:# or  i == (0, 14, 15):
            center = new_center
            radius = new_radius
            #print('Found new circle:',radius)
            #print(i)

print('Center:',center)
print('diameters:',radius*2)

circle = generate_circle(center,radius)

points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
                   [2, 0], [2, 1], [2, 2]])
from scipy.spatial import Voronoi, voronoi_plot_2d
vor = Voronoi(XY,incremental=False)


fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
                line_width=2, line_alpha=0.5, point_size=5)
#print(type(fig))
#fig = voronoi_plot_2d(vor,figsize=plot_dims)
#print('vertices:',vor.vertices)
#print('regions:',vor.regions)
#print('Number of regions:',len(vor.regions))
#for region in vor.regions:
##    print(region)
##    print(XY[region[0]])
#    if len(region) == 3:
#        print(region)
#        new_center, new_radius = define_circle(XY[region[0]],XY[region[1]],XY[region[2]])
#        print(new_center, new_radius)
     


#print(top_smallest([1, 5, 3, 4, -1]))

center2= [0.1,0.05]
radius2 = 0
for i,vertex in enumerate(vor.vertices):
#    print(vertex)
#    print(XY[vertex[0]])
    distances = distance_to_point_set(XY,vertex)
    #print(vor.regions[i])
    #print(distances)
    top3_points = top_smallest(distances)
    #print(top3_points)
    #new_center,new_radius = define_circle(XY[top3_points[0]],XY[top3_points[1]],XY[top3_points[2]])
    new_center,new_radius = vertex, dist(vertex,XY[top3_points[0]])
    if new_radius > radius2:
        radius2,center2 = new_radius,new_center
        print(center2,radius2)

print(center2,radius2)
ideal = generate_circle(center,radius)
current = generate_circle(center2,radius2)

fig = plt.gcf()
plt.scatter(center[0],center[1],marker='x')
plt.scatter(center2[0],center2[1],marker = 'x')
plt.scatter(ideal[:,0],ideal[:,1],marker=".",linewidths = 0.05)
plt.scatter(current[:,0],current[:,1],marker=".",linewidths = 0.05)
for i,n in enumerate(XY):
    plt.annotate(str(i), (XY[i,0], XY[i,1]))
fig.set_size_inches(10, 10)
fig.savefig('test2png.png', dpi=300)

plt.show()

if False:
    plot_dims = (10,10)
    plt.figure(1,figsize=plot_dims)
    plt.scatter(circle[:,0],circle[:,1])
    plt.scatter(XY[:,0],XY[:,1])
    for i,n in enumerate(XY):
        plt.annotate(str(i), (XY[i,0], XY[i,1]))
    plt.ylim(0,.2)
    plt.xlim(0,.2)
    plt.show()
