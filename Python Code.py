#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:22:22 2024

@author: nandani
"""

import numpy as np
from scipy.linalg import expm

# Helper function: Skew-symmetric matrix for 3D vectors
def skew_symmetric(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

# Function 1: Compute the Spatial Jacobian (J_s)
def calculate_spatial_jacobian(S_list, thetas):
    """
    Calculate the spatial Jacobian J_s based on screw axes (S_list) and joint angles (thetas).
    
    Arguments:
    S_list -- A list of screw axes for the robot's joints (6xN for N joints)
    thetas -- A list of joint angles (length N)
    
    Returns:
    J_s -- Spatial Jacobian (6xN)
    """
    n = len(thetas)
    J_s = np.zeros((6, n))
    
    T = np.eye(4)  # Identity matrix (base frame at the start)
    
    for i in range(n):
        # Update only the angular part of the screw axis with the current transformation
        R = T[:3, :3]  # Rotation matrix from the current transformation
        J_s[:3, i] = np.dot(R, S_list[:3, i])  # Angular part (first 3 rows)
        J_s[3:, i] = np.dot(R, S_list[3:, i])  # Linear part (last 3 rows)
        
        # Calculate the transformation matrix for the next joint
        S_skew = skew_symmetric(S_list[:3, i])
        v = S_list[3:, i]
        
        # Construct the matrix for the exponential map
        se3_mat = np.vstack([
            np.hstack([S_skew, v.reshape(3, 1)]),
            np.array([0, 0, 0, 0])
        ])
        T = np.dot(T, expm(se3_mat * thetas[i]))
    
    return J_s

# Function 2: Compute the Body Jacobian (J_b)
def calculate_body_jacobian(B_list, thetas):
    """
    Calculate the body Jacobian J_b based on body screw axes (B_list) and joint angles (thetas).
    
    Arguments:
    B_list -- A list of body screw axes (6xN for N joints)
    thetas -- A list of joint angles (length N)
    
    Returns:
    J_b -- Body Jacobian (6xN)
    """
    n = len(thetas)
    J_b = np.zeros((6, n))
    
    T = np.eye(4)  # Identity matrix (end-effector frame at the start)
    
    for i in range(n):
        R = T[:3, :3]
        J_b[:3, i] = np.dot(R, B_list[:3, i])
        J_b[3:, i] = np.dot(R, B_list[3:, i])
        
        S_skew = skew_symmetric(B_list[:3, i])
        v = B_list[3:, i]
        
        se3_mat = np.vstack([
            np.hstack([S_skew, v.reshape(3, 1)]),
            np.array([0, 0, 0, 0])
        ])
        T = np.dot(T, expm(-se3_mat * thetas[i]))
    
    return J_b

# Function 3: Compute Manipulability Measures
def manipulability_measures(J):
    U, S, Vt = np.linalg.svd(J)
    
    # u1: Volume of manipulability ellipsoid
    u1 = np.prod(S)
    
    # u2: Condition number (largest/smallest singular values)
    u2 = np.max(S) / np.min(S)
    
    # u3: Minimum singular value
    u3 = np.min(S)
    
    return u1, u2, u3

# Function 4: Check for Singularity
def is_singular(J):
    _, S, _ = np.linalg.svd(J)
    return np.min(S) < 1e-6

# Function 5: Compute Torques to Hold an Object
def compute_gravity_torque(J, m):
    F_gravity = np.array([0, 0, -m * 9.81, 0, 0, 0])
    torques = np.dot(J.T, F_gravity)
    return torques

# ---- TEST CASES ----
if __name__ == "__main__":
    # Example: 3-DOF robot with screw axes
    S_list = np.array([
        [0, 0, 1, 0, 0, 0],  # Screw axis for joint 1 (rotation about z-axis)
        [0, 1, 0, 0, 0, 1],  # Screw axis for joint 2 (rotation about y-axis)
        [1, 0, 0, 0, 1, 0]   # Screw axis for joint 3 (rotation about x-axis)
    ]).T

    # Joint angles for the robot (radians)
    thetas = [0.5, 1.0, -0.5]

    # 1. Compute the Spatial Jacobian
    J_s = calculate_spatial_jacobian(S_list, thetas)
    print("\nSpatial Jacobian J_s:\n", J_s)

    # 2. Compute the Body Jacobian
    J_b = calculate_body_jacobian(S_list, thetas)
    print("\nBody Jacobian J_b:\n", J_b)

    # 3. Compute Manipulability Measures
    u1, u2, u3 = manipulability_measures(J_s)
    print(f"\nManipulability Measures:\n u1 = {u1:.4f}, u2 = {u2:.4f}, u3 = {u3:.4f}")

    # 4. Check for Singular Configurations
    singular = is_singular(J_s)
    print(f"\nIs the configuration singular? {'Yes' if singular else 'No'}")

    # 5. Compute Torques to Hold a 150g Object
    mass = 0.150  # 150 grams in kilograms
    torques = compute_gravity_torque(J_s, mass)
    print(f"\nRequired Torques to Hold 150g Object: {torques}")
