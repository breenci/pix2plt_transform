# %%
import numpy as np
import matplotlib.pyplot as plt
import glob


def matrix_transform(coords, trans_mat):
    """Do the matrix transformation"""

    # pad input coords with ones to allow translation
    pad_coords = np.vstack((coords.T, np.ones_like(coords[:, 0])))

    # do transformation
    pad_trans_coords = np.matmul(trans_mat, pad_coords)

    # unpad
    trans_coords = pad_trans_coords[:3, :].T

    return trans_coords


def pix2plt(coords_arr, tmat_arr):
    """Transform from pixel to plate coordinate systems"""

    # pad z
    pad_coords = np.insert(coords_arr, 2, 0, axis=1)

    # do transform
    out_coords = matrix_transform(pad_coords, tmat_arr)

    return out_coords


def plt2pix(coords_arr, mat_arr):
    """Transform from plate to pixel coordinate systems"""

    # take matrix inverse. Note: Assumes input matrix describes pixel to
    # plate transformation
    plt2pix_mat = np.linalg.inv(mat_arr)

    # do transform
    plt_in_pix = matrix_transform(coords_arr, plt2pix_mat)

    # remove z axis
    out_coords = plt_in_pix[:, 0:2]

    return out_coords


def transform_from_file(coords_fn, mat_fn, out_fn=None, mode='pix2plt'):
    """Apply a coordinate transformation where coordinates and transformation
    matrices are in text file format

    :param coords_fn: Name of file containing coordinates to be transformed
    :type coords_fn: Numpy array with xyz values in columns
    :param mat_fn: Name of file containing transformation matrix
    :type mat_fn: 4 x 4 numpy array
    :param out_fn: Path to saved output file
    :type out_fn: str, optional
    :param mode: Specifies the direction of transformation, defaults to 'pix2plt'
    :type mode: str, optional
    :return: Transformed coordinates
    :rtype: Numpy array with xyz (pix2plt) or xy (plt2pix) values in columns
    """

    # load files
    coords = np.loadtxt(coords_fn)
    trans_mat = np.loadtxt(mat_fn)

    # choose direction of transformation. Pixel to plate (pix2plt) or
    # plt2pix. Note: This assumes that the input matrix describes the
    # pix2plt mode
    if mode == 'pix2plt':
        out_coords = pix2plt(coords, trans_mat)

    if mode == 'plt2pix':
        out_coords = plt2pix(coords, trans_mat)

    # save output as .txt file
    if out_fn is not None:
        np.savetxt(out_fn, out_coords)

    return out_coords


if __name__ == '__main__':
    # Test transformations using mask centroid data
    fig, (ax1,ax2) = plt.subplots(1,2)

    # open transform the mask centroid file with each of the transformation matrices
    for mask in np.sort(glob.glob('transformation_mats/t_mat*')):
        # extract the mask id and use to identify saved tramsformed coordinate files
        mask_id = mask[26:28]
        plt_coords = transform_from_file('mask_AC_01.txt', mask, out_fn='transformed_coords/t_coords_'+ mask_id + '.txt')
        print(mask)
        
        # plotting
        ax1.scatter(plt_coords[:, 0], plt_coords[:, 1], s=1, c='b')
        ax1.text(plt_coords[1,0]-50, plt_coords[1,1]+20, mask_id)
        ax2.scatter(plt_coords[:, 0], plt_coords[:, 2], s=1, c='b')

    ax1.set_title('XY Plane')
    ax2.set_title('XZ Plane')
    plt.show()

    # do inverse transform
    mask_from_coords = transform_from_file('transformed_coords/t_coords_02.txt', 'transformation_mats/t_mat_02.txt', mode='plt2pix')
    AC_mask = np.loadtxt('mask_AC_01.txt')
    # isclose returns true if inverse transform reproduces original coords
    print(np.isclose(mask_from_coords, AC_mask, rtol=1e-10))


# %%
