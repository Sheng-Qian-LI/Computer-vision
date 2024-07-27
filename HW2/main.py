###
### This homework is modified from CS231.
###


import sys
import numpy as np
import os
from scipy.optimize import least_squares
from scipy.optimize import minimize
import math
from copy import deepcopy
from skimage.io import imread
from sfm_utils import *

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''

def estimate_initial_RT(E):
    U, S, VT = np.linalg.svd(E)

    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    Z = np.array([[0, 1, 0],
                  [-1, 0, 0],
                  [0, 0, 0]])

    M = np.dot(U, np.dot(Z, U.T))

    Q1 = np.dot(U, np.dot(W, VT))
    Q2 = np.dot(U, np.dot(W.T, VT))

    R1 = (np.linalg.det(Q1)) * Q1
    R2 = (np.linalg.det(Q2)) * Q2

    u3 = U[:, 2]
    T1 = u3
    T2 = -u3

    RT1 = np.hstack((R1, T1[:, np.newaxis]))
    RT2 = np.hstack((R1, T2[:, np.newaxis]))
    RT3 = np.hstack((R2, T1[:, np.newaxis]))
    RT4 = np.hstack((R2, T2[:, np.newaxis]))

    RTs = []
    RTs.append(RT1)
    RTs.append(RT2)
    RTs.append(RT3)
    RTs.append(RT4)

    return RTs

'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''

def linear_estimate_3d_point(image_points, camera_matrices):
    M = len(image_points)
    A = np.zeros((2 * M, 4))

    for i in range(M):
        # u
        A[2 * i, 0] = image_points[i][1] * camera_matrices[i][2, 0] - camera_matrices[i][1, 0]
        A[2 * i, 1] = image_points[i][1] * camera_matrices[i][2, 1] - camera_matrices[i][1, 1]
        A[2 * i, 2] = image_points[i][1] * camera_matrices[i][2, 2] - camera_matrices[i][1, 2]
        A[2 * i, 3] = image_points[i][1] * camera_matrices[i][2, 3] - camera_matrices[i][1, 3]
        # v
        A[2 * i + 1, 0] = camera_matrices[i][0, 0] - image_points[i][0] * camera_matrices[i][2, 0]
        A[2 * i + 1, 1] = camera_matrices[i][0, 1] - image_points[i][0] * camera_matrices[i][2, 1]
        A[2 * i + 1, 2] = camera_matrices[i][0, 2] - image_points[i][0] * camera_matrices[i][2, 2]
        A[2 * i + 1, 3] = camera_matrices[i][0, 3] - image_points[i][0] * camera_matrices[i][2, 3]
        
        
    U, S, VT = np.linalg.svd(A)
    # 取最後一行
    point_3d = VT[-1, :-1] / VT[-1, -1]

    return point_3d


'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2M reprojection error vector
'''

def reprojection_error(point_3d, image_points, camera_matrices):
    M = len(image_points)
    error = np.empty((2 * M,))

    for i in range(M):
        Mi = camera_matrices[i]
        pi = image_points[i]
        
        P = np.append(point_3d, 1)
        
        y = np.dot(Mi, P)

        y3_inv = 1.0 / y[2]
        p_prime_i = y[:2] * y3_inv
        ei = p_prime_i - pi
        error[2 * i:2 * (i + 1)] = ei

    return error



'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    
    M = camera_matrices.shape[0]
    jacobian = np.zeros((2 * M, 3))

    for i in range(M):
        P = camera_matrices[i]
        projected_point = np.dot(P, np.append(point_3d, 1))

        u, v, w = projected_point[0], projected_point[1], projected_point[2]
        
        dZ = P[2, 0:3]

        du_dX = P[0, 0] / w
        du_dY = P[0, 1] / w
        du_dZ = P[0, 2] / w
        
        dv_dX = P[1, 0] / w
        dv_dY = P[1, 1] / w
        dv_dZ = P[1, 2] / w

        jacobian[2 * i, :] = [du_dX, du_dY, du_dZ] - u * dZ / (w ** 2)
        jacobian[2 * i + 1, :] = [dv_dX, dv_dY, dv_dZ] - v * dZ / (w ** 2)

    return jacobian


'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''

def nonlinear_estimate_3d_point(image_points, camera_matrices):

    linear_ppoint = linear_estimate_3d_point(image_points, camera_matrices)
    point_3d = linear_ppoint

    for iteration in range(10):

        error = reprojection_error(point_3d, image_points, camera_matrices)
        J = jacobian(point_3d, camera_matrices)

        Hessian = np.dot(J.T, J)

        Hessian_inv = np.linalg.inv(Hessian)
        gradient = np.dot(J.T, error)
        step = np.dot(Hessian_inv, gradient)

        point_3d -= step

    return point_3d


'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''

def estimate_RT_from_E(E, image_points, K):

    U, S, Vt = np.linalg.svd(E)

    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    Q1 = np.dot(U, np.dot(W, Vt))
    Q2 = np.dot(U, np.dot(W.T, Vt))

    R1 = (np.linalg.det(Q1)) * Q1
    R2 = (np.linalg.det(Q2)) * Q2

    t = U[:, 2]
    T1 = t
    T2 = -t

    RT1 = np.dot(-(R1.T) , T1)
    RT2 = np.dot(-(R1.T) , T2)
    RT3 = np.dot(-(R2.T) , T1)
    RT4 = np.dot(-(R2.T) , T2)

    RT5 = np.hstack((R1.T, RT1[:, np.newaxis]))
    RT6 = np.hstack((R1.T, RT2[:, np.newaxis]))
    RT7 = np.hstack((R2.T, RT3[:, np.newaxis]))
    RT8 = np.hstack((R2.T, RT4[:, np.newaxis]))

    M1= np.dot(np.linalg.inv(K) , RT5)
    M2= np.dot(np.linalg.inv(K) , RT6)
    M3= np.dot(np.linalg.inv(K) , RT7)
    M4= np.dot(np.linalg.inv(K) , RT8)

    X, Y, Z = image_points.shape
    count_R1_T1 = 0
    count_R1_T2 = 0
    count_R2_T1 = 0
    count_R2_T2 = 0

    for i in range(X):
        for j in range(Y):
            M = np.append(image_points[i, j, :], 1)
            R1T1 = M1[:, :-1]
            R1_T1 = M - M1[:, -1]
            R1T2 = M2[:, :-1]
            R1_T2 = M - M2[:, -1]
            R2T1 = M3[:, :-1]
            R2_T1 = M - M3[:, -1]
            R2T2 = M4[:, :-1]
            R2_T2 = M - M4[:, -1]

            Z1= np.linalg.lstsq(R1T1, R1_T1, rcond=None)[0]
            Z2 = np.linalg.lstsq(R1T2, R1_T2, rcond=None)[0]
            Z3 = np.linalg.lstsq(R2T1, R2_T1, rcond=None)[0]
            Z4 = np.linalg.lstsq(R2T2, R2_T2, rcond=None)[0]
        
            if Z1[2] > 0:
                count_R1_T1 += 1
            if Z2[2] > 0:
                count_R1_T2 += 1
            if Z3[2] > 0:
                count_R2_T1 += 1
            if Z4[2] > 0:
                count_R2_T2 += 1

    max_z = max(count_R1_T1, count_R1_T2, count_R2_T1, count_R2_T2)

    if max_z == count_R1_T1 :
        R = R1
        T = T2
    elif max_z == count_R1_T2 :
        R = R1
        T = T2
    elif max_z == count_R2_T1 :
        R = R2
        T = T1
    elif max_z == count_R2_T2 :
        R = R2
        T = T2 

    RT = np.hstack((R, T.reshape(3, 1)))
    return RT


if __name__ == '__main__':
    run_pipeline = True

    # Load the data
    image_data_dir = 'data/statue/'
    unit_test_camera_matrix = np.load('data/unit_test_camera_matrix.npy')
    unit_test_image_matches = np.load('data/unit_test_image_matches.npy')

    image_paths = [os.path.join(image_data_dir, 'images', x) for x in sorted(os.listdir('data/statue/images')) if '.jpg' in x]

    focal_length = 719.5459  ###########################################################調整焦距

    matches_subset = np.load(os.path.join(image_data_dir, 'matches_subset.npy'), allow_pickle=True, encoding='latin1')[0,:]
    
    dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches.npy'), allow_pickle=True, encoding='latin1')
    fundamental_matrices = np.load(os.path.join(image_data_dir, 'fundamental_matrices.npy'), allow_pickle=True, encoding='latin1')[0,:]

    #############################################################################
    # Part A: Computing the 4 initial R,T transformations from Essential Matrix #
    #############################################################################
    print('-' * 80)
    print("Part A: Check your matrices against the example R,T")
    print('-' * 80)
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length ###########################################################調整焦距

    E = K.T.dot(fundamental_matrices[0]).dot(K)  ########################################################### K轉置dot fundamental_matrices 得E本質矩陣, 
    im0 = imread(image_paths[0])
    im_height, im_width, _ = im0.shape
    example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
                           [0.1019, 0.9948, 0.0045, -0.0089],
                           [0.2041, -0.0254, 0.9786, 0.0331]])
    print("Example RT:\n", example_RT)
    estimated_RT = estimate_initial_RT(E)
    print('')
    print("Estimated RT:\n", estimated_RT)

    ##############################################################
    # Part B: Determining the best linear estimate of a 3D point #
    ##############################################################
    print('-' * 80)
    print('Part B: Check that the difference from expected point ')
    print('is near zero')
    print('-' * 80)
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(example_RT)
    unit_test_matches = matches_subset[0][:,0].reshape(2,2) ########################################################### unit_test_matches就是image_points
    
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(), camera_matrices.copy()) 
    # unit_test_matches.shape = (2,2) camera_matrices.shape = (2,3,4)

    expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    print("Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum())

    ###############################################################
    # Part C: Calculating the reprojection error and its Jacobian #
    ###############################################################
    print('-' * 80)
    print('Part C: Check that the difference from expected error/Jacobian ')
    print('is near zero')
    print('-' * 80)
    estimated_error = reprojection_error(expected_3d_point, unit_test_matches, camera_matrices)
    estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631))
    print("Error Difference: ", np.fabs(estimated_error - expected_error).sum())
    expected_jacobian = np.array([[ 154.33943931, 0., -22.42541691],
                                  [0., 154.33943931, 36.51165089],
                                  [141.87950588, -14.27738422, -56.20341644],
                                  [21.9792766, 149.50628901, 32.23425643]])
    print("Jacobian Difference: ", np.fabs(estimated_jacobian - expected_jacobian).sum())

    #################################################################
    # Part D: Determining the best nonlinear estimate of a 3D point #
    #################################################################
    print('-' * 80)
    print('Part D: Check that the reprojection error from nonlinear method')
    print('is lower than linear method')
    print('-' * 80)

    estimated_3d_point_linear = linear_estimate_3d_point(unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    # unit_test_image_matches.shape = (4,2) unit_test_camera_matrix.shape = (4,3,4)
    
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    
    error_linear = reprojection_error(estimated_3d_point_linear, unit_test_image_matches, unit_test_camera_matrix)
    print("Linear method error:", np.linalg.norm(error_linear))
    error_nonlinear = reprojection_error(estimated_3d_point_nonlinear, unit_test_image_matches, unit_test_camera_matrix)
    print("Nonlinear method error:", np.linalg.norm(error_nonlinear))

    ##############################################################
    # Part E: Determining the correct R, T from Essential Matrix #
    ##############################################################
    print('-' * 80)
    print("Part E: Check your matrix against the example R,T")
    print('-' * 80)
    estimated_RT = estimate_RT_from_E(E, np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    print("Example RT:\n", example_RT)
    print('!')
    print("Estimated RT:\n", estimated_RT)

    #########################################################
    # Part F: Run the entire Structure from Motion pipeline #
    #########################################################
    if not run_pipeline:
        sys.exit()
    print('-' * 80)
    print('Part F: Run the entire SFM pipeline')
    print('-' * 80)
    frames = [0] * (len(image_paths) - 1)
    
    for i in range(len(image_paths)-1):
        frames[i] = Frame(matches_subset[i].T, focal_length, fundamental_matrices[i], im_width, im_height)
        bundle_adjustment(frames[i])

    # print('frame',frames)
    merged_frame = merge_all_frames(frames) ###########################################################有缺東西
 
    # Construct the dense matching
    camera_matrices = np.zeros((2,3,4))
    dense_structure = np.zeros((0,3))

    # print('!!!!!!!!!!!!!')
    for i in range(len(frames)-1):
        matches = dense_matches[i]
        camera_matrices[0,:,:] = merged_frame.K.dot(merged_frame.motion[i,:,:])
        camera_matrices[1,:,:] = merged_frame.K.dot(merged_frame.motion[i+1,:,:])
        points_3d = np.zeros((matches.shape[1], 3))
        use_point = np.array([True]*matches.shape[1])
        for j in range(matches.shape[1]):
            points_3d[j,:] = nonlinear_estimate_3d_point(matches[:,j].reshape((2,2)), camera_matrices)
        dense_structure = np.vstack((dense_structure, points_3d[use_point,:]))

    np.save('results.npy', dense_structure)
    print ('Save results to results.npy!')
