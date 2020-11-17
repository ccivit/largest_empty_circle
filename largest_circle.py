from math import pi
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import csv

def load_point_cloud(filepath):
    point_cloud = np.empty([0, 2])
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile, delimiter="\t", quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            point_cloud = np.append(point_cloud,[row],axis=0)
    return point_cloud

def dist(a,b):
    return np.linalg.norm(a-b)

def draw_circle(center,radius):
    step_size = 0.005
    circle = np.zeros((int(1/step_size),2))
    t_range = np.arange(0,1,step_size)
    for i,t in enumerate(t_range):
        circle[i] = np.array([center[0] + radius*np.cos(2*pi*t),
                              center[1] + radius*np.sin(2*pi*t)])
    return circle

def distance_to_point_set(array_of_points,pivot_point):
    distances = []
    for point in array_of_points:
        distances.append(dist(point,pivot_point))
    return distances

def top_smallest(list_of_numbers,nb_of_top=3):
    # Returns index of the 3 smallest items in an list of numbers
    # mn_v stands for value of number n, and mn_i stands for index of number n
    # in list_of_numbers
    m1_v, m2_v, m3_v = float('inf'), float('inf'), float('inf')
    m1_i, m2_i, m3_i = float('inf'), float('inf'), float('inf')
    for i,x in enumerate(list_of_numbers):
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

def save_results_fig(point_cloud, center, radius, output_image):
    fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
                    line_width=2, line_alpha=0.5, point_size=5)
    current = draw_circle(center,radius)
    plt.scatter(center[0],center[1],marker = 'x')
    plt.scatter(current[:,0],current[:,1],marker=".",linewidths = 0.05)
    for i,n in enumerate(point_cloud):
        plt.annotate(str(i), (point_cloud[i,0], point_cloud[i,1]))
    fig.set_size_inches(10, 10)
    fig.savefig(output_image, dpi=300)

if __name__ == "__main__":
    data_file = 'sample.csv'
    output_image = 'example.png'
    point_cloud = load_point_cloud(data_file)
    print("Data has:",len(point_cloud),'points')

    vor = Voronoi(point_cloud,incremental=False)

    radius = 0 # We initialize with zero since we're looking for maximum radius
    for i,vertex in enumerate(vor.vertices):
        distances = distance_to_point_set(point_cloud,vertex)
        top3_points = top_smallest(distances)
        new_center,new_radius = vertex, dist(vertex,point_cloud[top3_points[0]])
        if new_radius > radius:
            radius,center = new_radius,new_center
            print('Iterating... Candidate',i,center,radius)

    print('Largest empty circle. Center:',center,'Radius:',radius)
    save_results_fig(vor, point_cloud, center, radius, output_image)
