import matplotlib.pyplot as plt
from math import pi
import numpy as np
from numpy import cos,sin



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


def dist(a,b):
    return np.linalg.norm(a-b)


def generate_circle(center,radius):
    step_size = 0.005
    circle = np.zeros((int(1/step_size),2))
    t_range = np.arange(0,1,step_size)
    for i,t in enumerate(t_range):
        circle[i] = np.array([center[0] + radius*cos(2*pi*t),
                              center[1] + radius*sin(2*pi*t)])
    return circle


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


center= [0,0]
radius = 0


XY = sample_cloud()
print(len(XY),'obstacles')


circle = generate_circle(center,radius)
print('Initialization circle. Center:',center,'Radius:',radius)

from scipy.spatial import Voronoi, voronoi_plot_2d
vor = Voronoi(XY,incremental=False)


fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
                line_width=2, line_alpha=0.5, point_size=5)


center2= [0.1,0.05]
radius2 = 0
for i,vertex in enumerate(vor.vertices):
    distances = distance_to_point_set(XY,vertex)

    top3_points = top_smallest(distances)

    new_center,new_radius = vertex, dist(vertex,XY[top3_points[0]])
    if new_radius > radius2:
        radius2,center2 = new_radius,new_center
        print('Iterating... Candidate',i,center2,radius2)

print('Largest empty circle. Center:',center2,'Radius:',radius2)
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

