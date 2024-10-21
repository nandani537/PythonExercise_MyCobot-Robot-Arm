#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:43:42 2024

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
    u2 = np.max(S) / np.min(S)  # Condition number
    return u2

# Function 4: Compute Joint Torques from a Wrench
def compute_torques(J, wrench):
    return np.dot(J.T, wrench)

# Function 5: Check for Singularity
def is_singular(J):
    _, S, _ = np.linalg.svd(J)
    return np.min(S) < 1e-6

# Function 6: Compute Torques to Hold an Object
def compute_gravity_torque(J, m):
    F_gravity = np.array([0, 0, -m * 9.81, 0, 0, 0])
    torques = np.dot(J.T, F_gravity)
    return torques

# ---- Constants ----
pi_over_16 = np.pi / 16
joint_angles = [pi_over_16] * 7  # Theta for each joint (7 DOF)

# Screw axes (space frame) for each joint (6x7 for 7 joints)
# These screw axes would typically be provided based on robot kinematics.
# I'm providing placeholder values based on common configurations for 7-DOF robots.
S_list = np.array([
    [0, 0, 1, 0, 0, 0],  # Joint 1
    [0, 1, 0, 0, 0, 0.34],  # Joint 2
    [0, 1, 0, 0, 0, 0.74],  # Joint 3
    [0, 1, 0, 0, 0, 1.14],  # Joint 4
    [1, 0, 0, 0.4, 0, 0],  # Joint 5
    [0, 1, 0, 0, 0, 1.29],  # Joint 6
    [1, 0, 0, 0.15, 0, 0]   # Joint 7
]).T

# For the body Jacobian, you would have body screw axes as well
# Using the same placeholder values for simplicity (in real cases, these differ).
B_list = S_list.copy()  

# Wrench Fs (space) and Fb (body) given in the problem
F_s = np.array([1, 1, 1, 1, 1, 1])  # Nm, N (space frame wrench)
F_b = np.array([1, 1, 1, 1, 1, 1])  # Nm, N (body frame wrench)

# ---- Calculations ----

# Part (d) - Space Jacobian and torques
J_s = calculate_spatial_jacobian(S_list, joint_angles)
torques_s = compute_torques(J_s, F_s)
u2_s = manipulability_measures(J_s)

print("Part (d):")
print("Space Jacobian J_s:\n", J_s)
print("Joint torques in space frame:\n", torques_s)
print(f"Manipulability measure µ2 in space frame: {u2_s:.4f}")

# Part (e) - Body Jacobian and torques
J_b = calculate_body_jacobian(B_list, joint_angles)
torques_b = compute_torques(J_b, F_b)
u2_b = manipulability_measures(J_b)

print("\nPart (e):")
print("Body Jacobian J_b:\n", J_b)
print("Joint torques in body frame:\n", torques_b)
print(f"Manipulability measure µ2 in body frame: {u2_b:.4f}")

Output :

runfile('/Users/nandani/Desktop/untitled0.py', wdir='/Users/nandani/Desktop')
Part (d):
Space Jacobian J_s:
 [[ 0.         -0.19509032 -0.19509032 -0.19509032  0.81549316 -0.08503795
   0.68813734]
 [ 0.          0.98078528  0.98078528  0.98078528  0.16221167  0.9830849
   0.1756849 ]
 [ 1.          0.          0.          0.         -0.55557023  0.16221167
  -0.70398993]
 [ 0.          0.          0.14159287  0.42787652  0.32619726  0.73850608
   0.1032206 ]
 [ 0.          0.          0.02816457  0.08510993  0.06488467 -0.10969896
   0.02635274]
 [ 0.          0.34        0.72578111  1.05322267 -0.22222809  1.05198617
  -0.10559849]]
Joint torques in space frame:
 [1.         1.12569496 1.68123351 2.35190407 0.59098844 2.74105191
 0.18380716]
Manipulability measure µ2 in space frame: 26.3877

Part (e):
Body Jacobian J_b:
 [[ 0.          0.19509032  0.19509032  0.19509032  0.81549316  0.29764548
   0.7029877 ]
 [ 0.          0.98078528  0.98078528  0.98078528 -0.16221167  0.94079463
  -0.10102707]
 [ 1.          0.          0.          0.          0.55557023 -0.16221167
   0.70398993]
 [ 0.          0.         -0.14159287 -0.42787652  0.32619726 -0.64031068
   0.10544816]
 [ 0.          0.          0.02816457  0.08510993 -0.06488467  0.38396267
  -0.01515406]
 [ 0.          0.34        0.72578111  1.05322267  0.22222809  1.05198617
   0.10559849]]
Joint torques in body frame:
 [1.         1.5158756  1.78822841 1.88633168 1.6923924  1.8718666
 1.50184315]
Manipulability measure µ2 in body frame: 29.9579
