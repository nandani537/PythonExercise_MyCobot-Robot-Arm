# Robot Arm Python Code for Assignment 3 :
For the remaining questions, assume the angles of the joints are ίπ/16 for joints i=1...7.

(d)	What is the space Jacobian? What joint torques are needed to generate the wrench Fs = (1 Nm,1 Nm,1 Nm,1 N,1 N, 1 N)? What is the manipulability measure µ₂ for the angular velocity manipulability ellipsoid in the space frame? What is the manipulability measure μ₂ for the linear manipulability ellipsoid in the space frame?

(e)	What is the body Jacobian? What joint torques are needed to generate the wrench Ft = (1 Nm,1 Nm, 1 Nm, 1 N, 1 N, 1 N)? What is the manipulability measure µ₂ for the angular velocity manipulability ellipsoid in the body frame? What is the manipulability measure µ₂ for the linear manipulability ellipsoid in the body frame?


5.Write python functions that computes the spatial and body Jacobians as a function of robot joint angles.

6.Write python functions to determine manipulability measures u1, u2, and u3.

7.Are you able to identify some singular configurations? Can you verify singularity by inspection of the jacobian?

8.Suppose that you need the robot to hold an object of weight 150g. Write a python function that, given the configuration of the arm as an input (the joint angles), returns the neccessary torques to be applied to each joint motor in order to hold the object weight. Test your function with some example configurations and verify that result is correct. 

