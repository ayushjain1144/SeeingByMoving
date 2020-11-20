
import skimage
from skimage.measure import marching_cubes as mcl
from mayavi import mlab
import numpy as np
import ipdb
st = ipdb.set_trace

def save_voxel_to_mesh(voxel_grid, output_fname):

    verts, faces, normals, values = mcl(voxel_grid, 0.0)


    mlab.triangular_mesh([vert[0] for vert in verts],
                            [vert[1] for vert in verts],
                            [vert[2] for vert in verts],
                            faces)
    faces = faces + 1
    mlab.show()

    thefile = open(output_fname, 'w')
    for item in verts:
        thefile.write("v {0} {1} {2}\n".format(item[0],item[1],item[2]))
    for item in normals:
        thefile.write("vn {0} {1} {2}\n".format(item[0],item[1],item[2]))
    for item in faces:
        thefile.write("f {0}/{0} {1}/{1} {2}/{2}\n".format(item[0],item[1],item[2]))  
    thefile.close()

def main():
    # load voxel grid

    voxel_grid = np.load('car_72_1775_1.npy' ,allow_pickle=True)
    #st()
    print(f'shape of voxel grid: {voxel_grid.shape}')
    save_voxel_to_mesh(voxel_grid[0][0], "couch_mesh.obj")

if __name__ == "__main__":
    main()
