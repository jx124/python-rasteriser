from cow import *
from math import floor, fmod
import numpy as np
from PIL import Image

def compute_screen_coordinates(film_aperture_width, film_aperture_height, image_width, image_height,
                               fit_film, near_clipping_plane, focal_length):
    '''Conputes the dimensions of the screen using a physically-based camera model.'''
    film_aspect_ratio = film_aperture_width / film_aperture_height
    device_aspect_ratio = image_width / image_height

    top = ((film_aperture_height * inch_to_mm / 2) / focal_length) * near_clipping_plane
    right = ((film_aperture_width * inch_to_mm / 2) / focal_length) * near_clipping_plane

    fov = 2 * 180/np.pi * np.arctan(right / near_clipping_plane)
    print(f"Field of view: {fov:.2f}°")

    xscale, yscale = 1, 1

    if fit_film:
        if (film_aspect_ratio > device_aspect_ratio):
            xscale = device_aspect_ratio / film_aspect_ratio
        else:
            yscale = film_aspect_ratio / device_aspect_ratio
    else:
        if (film_aspect_ratio > device_aspect_ratio):
            yscale = film_aspect_ratio / device_aspect_ratio
        else:
            xscale = device_aspect_ratio / film_aspect_ratio

    right *= xscale
    top *= yscale

    return (top, -top, -right, right)

def edge_function(a,b,c):
    #Returns ab cross ac
    return - (c[1] - a[1]) * (b[0] - a[0]) + (c[0] - a[0]) * (b[1] - a[1]) 

def convert_to_raster(vertex_world, world_to_camera, l, r, t, b, near, image_width, image_height):
    #Convert from world to camera coordinates
    vertex_camera = np.append(vertex_world,1) @ world_to_camera

    #Convert from camera to screen coordinates
    vertex_screen_x = near * vertex_camera[0] / -vertex_camera[2]
    vertex_screen_y = near * vertex_camera[1] / -vertex_camera[2]

    #Convert from screen to normalised coordinates in [-1,1]
    vertex_NDC_x = 2 * vertex_screen_x / (r - l) - (r + l) / (r - l)
    vertex_NDC_y = 2 * vertex_screen_y / (t - b) - (t + b) / (t - b)

    #Convert from normalised coordinates to raster coordinates
    vertex_raster_x = (vertex_NDC_x + 1) / 2 * image_width
    vertex_raster_y = (1 - vertex_NDC_y) / 2 * image_height
    vertex_raster_z = -vertex_camera[2]

    return np.array([vertex_raster_x, vertex_raster_y, vertex_raster_z])



focal_length = 35.
film_aperture_width = 0.980
film_aperture_height = 0.735
inch_to_mm = 25.4
near_clipping_plane = 1
far_clipping_plane = 1000
image_width = 640
image_height = 480
fit_film = 0 #fit_film: 0 for overscan 1 for fill
n_triangles = 3156 #number of triangles

world_to_camera = np.array([
    0.707107, -0.331295, 0.624695, 0, 
    0, 0.883452, 0.468521, 0, 
    -0.707107, -0.331295, 0.624695, 0,
    -1.63871, -5.747777, -40.400412, 1
]).reshape((4,4))
camera_to_world = np.linalg.inv(world_to_camera)

def main():
    
    t, b, l, r = compute_screen_coordinates(film_aperture_width, film_aperture_height, image_width, image_height,
                                            fit_film, near_clipping_plane, focal_length)

    convert_to_raster(vertices[0,:], world_to_camera, l, r, t, b, near_clipping_plane, image_width, image_height)

    #Initialise framebuffer and depthbuffer
    framebuffer = np.zeros((image_width * image_height, 3))
    depthbuffer = np.ones(image_width * image_height) * far_clipping_plane

    for i in range(n_triangles):
        #Extract vertices
        v0 = vertices[nvertices[i * 3]]
        v1 = vertices[nvertices[i * 3 + 1]]
        v2 = vertices[nvertices[i * 3 + 2]]

        #Convert to raster space
        v0_raster = convert_to_raster(v0, world_to_camera, l, r, t, b, near_clipping_plane, image_width, image_height)
        v1_raster = convert_to_raster(v1, world_to_camera, l, r, t, b, near_clipping_plane, image_width, image_height)
        v2_raster = convert_to_raster(v2, world_to_camera, l, r, t, b, near_clipping_plane, image_width, image_height)

        #Precompute reciprocal
        v0_raster[2] = 1./v0_raster[2]
        v1_raster[2] = 1./v1_raster[2]
        v2_raster[2] = 1./v2_raster[2]
        
        #Extract vertex attributes
        st0 = st[stindices[i * 3]]
        st1 = st[stindices[i * 3 + 1]]
        st2 = st[stindices[i * 3 + 2]]

        #Multiply by the reciprocals of their vertex z-coordinates
        st0 *= v0_raster[2]
        st1 *= v1_raster[2]
        st2 *= v2_raster[2]

        x_min = min(v0_raster[0], v1_raster[0], v2_raster[0])
        y_min = min(v0_raster[1], v1_raster[1], v2_raster[1])
        x_max = max(v0_raster[0], v1_raster[0], v2_raster[0])
        y_max = max(v0_raster[1], v1_raster[1], v2_raster[1])

        if (x_min > image_width -1 or x_max < 0 or y_min > image_height -1 or y_max < 0):
            continue
        
        #Calculate bounding box
        x0 = max(0, int(floor(x_min)))
        x1 = min(image_width - 1, int(floor(x_max)))
        y0 = max(0, int(floor(y_min)))
        y1 = min(image_height - 1, int(floor(y_max)))
        
        area = edge_function(v0_raster, v1_raster, v2_raster)

        #Inner loop
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                pixel_sample = (x + 0.5, y + 0.5)
                w0 = edge_function(v1_raster, v2_raster, pixel_sample)
                w1 = edge_function(v2_raster, v0_raster, pixel_sample)
                w2 = edge_function(v0_raster, v1_raster, pixel_sample)

                if (w0 >= 0 and w1 >= 0 and w2 >=0):
                    w0 /= area
                    w1 /= area
                    w2 /= area

                    #Interpolate 1/depth
                    z_reciprocal = v0_raster[2] * w0 + v1_raster[2] * w1 + v2_raster[2] * w2
                    z = 1. / z_reciprocal

                    #Depth buffer test
                    if (z < depthbuffer[y * image_width + x]):
                        depthbuffer[y * image_width + x] = z

                        #Interpolate vertex attributes
                        st_temp = (st0 * w0 + st1 * w1 + st2 * w2) * z
                        
                        #Computing point in camera space by interpolating point coordinates divided by vertex z-coordinate
                        v0_camera = np.append(v0, 1) @ world_to_camera
                        v1_camera = np.append(v1, 1) @ world_to_camera
                        v2_camera = np.append(v2, 1) @ world_to_camera

                        point_x = (v0_camera[0]/-v0_camera[2]) * w0 + (v1_camera[0]/-v1_camera[2]) *\
                                  w1 + (v2_camera[0]/-v2_camera[2]) * w2
                        point_y = (v0_camera[1]/-v0_camera[2]) * w0 + (v1_camera[1]/-v1_camera[2]) *\
                                  w1 + (v2_camera[1]/-v2_camera[2]) * w2
                        point = np.array([point_x * z, point_y * z, -z])

                        #Computing the normal and taking its dot product with the camera ray
                        normal = np.cross(v1_camera[0:3] - v0_camera[0:3], v2_camera[0:3] - v0_camera[0:3])
                        normal /= np.linalg.norm(normal) 
                        view_direction = -point
                        view_direction /= np.linalg.norm(view_direction)
                        normal_dot_view = max(0., np.dot(normal, view_direction))

                        #Multiply colour by checkerboard
                        M = 10.0
                        checker = (fmod(st_temp[0] * M, 1.0) > 0.5) ^ (fmod(st_temp[1] * M, 1.0) < 0.5)
                        c = 0.3 * (1 - checker) + 0.7 * checker
                        normal_dot_view *= c

                        framebuffer[y * image_width + x, 0] = normal_dot_view * 255 
                        framebuffer[y * image_width + x, 1] = normal_dot_view * 255
                        framebuffer[y * image_width + x, 2] = normal_dot_view * 255
        if (i % int(n_triangles/20) == 0):
            print(f"Progress: {(i/n_triangles) * 100:.2f}% complete")

    im = Image.fromarray(framebuffer.reshape((image_height, image_width, 3)).astype(np.uint8))
    im.show()

if __name__ == "__main__":
    main()