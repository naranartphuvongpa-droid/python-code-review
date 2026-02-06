import numpy as np
import math

def translation_matrix_binder_parabolic(start, end, amplitude, num_nodes, window=2):
    """
    Parabolic binder translation matrix calculator
    
    Parameters:
    - start, end: x-axis start and end positions
    - amplitude: maximum z displacement
    - num_nodes: number of nodes
    - window: local neighborhood for slope calculation
    
    Returns:
    - theta_deg: list of angles (degree) at each segment
    - length: list of segment lengths
    """
    # x positions
    x = np.linspace(start, end, num_nodes)
    # z positions - parabolic curve
    z = amplitude * ((x - start) / (end - start))**2
    
    theta = []
    length = []
    
    for i in range(num_nodes - 1):
        # Local window for slope
        i0 = max(0, i - window)
        i1 = min(num_nodes, i + window + 1)
        
        # Local slope
        slope, _ = np.polyfit(x[i0:i1], z[i0:i1], 1)
        theta_i = math.atan(slope) # Don't forget to take data from TexGen
        
        # Segment length
        dx = x[i+1] - x[i]
        dz = z[i+1] - z[i]
        length_i = math.sqrt(dx**2 + dz**2)
        
        theta.append(theta_i)
        length.append(length_i)
    
    T_tensor_binder = []
    T_eps_binder = []
    T_tensor_binder_inv = []
    T_eps_binder_inv_transpose = []
    for i in range(len(theta)):
        c = np.cos(theta[i])
        s = np.sin(theta[i])
        T_tensor_binder_i = np.array([
            [c**2, 0, s**2, 0, 2*c*s, 0],
            [0, 1, 0, 0, 0, 0],
            [s**2, 0, c**2, 0, -2*c*s, 0],
            [0, 0, 0, c, 0, -s],
            [-c*s, 0, c*s, 0, c**2 - s**2, 0],
            [0, 0, 0, s, 0, c]
        ])  
        T_eps_binder_i = np.array([
            [c**2, 0, s**2, 0, c*s, 0],
            [0, 1, 0, 0, 0, 0],
            [s**2, 0, c**2, 0, -c*s, 0],
            [0, 0, 0, c, 0, -s],
            [-c*s, 0, c*s, 0, c**2 - s**2, 0],
            [0, 0, 0, s, 0, c]
        ])
        T_tensor_binder.append(T_tensor_binder_i)
        T_eps_binder.append(T_eps_binder_i)
        T_tensor_binder_inv.append(np.linalg.inv(T_tensor_binder_i))
        T_eps_binder_inv_transpose.append(np.linalg.inv(T_eps_binder_i).T)
    return theta, length, T_tensor_binder, T_eps_binder, T_tensor_binder_inv, T_eps_binder_inv_transpose

def binder_fixed_angle_theta(binder_angle_deg, num_nodes=20):
    """
    Return theta list and length list for a fixed-angle binder.

    binder_angle_deg: angle measured from x-axis (x-z plane)
    num_nodes: number of discretization points along binder path
    """
    th = math.radians(float(binder_angle_deg))
    theta = [th for _ in range(num_nodes - 1)]
    length = [1.0 for _ in range(num_nodes - 1)]  # constant weights
    return theta, length

import math

def binder_fixed_angle_theta(binder_angle_deg, num_nodes=20):
    """
    Fixed binder inclination angle (measured from x-axis) for all segments.
    Returns:
      theta (list of radians), length (list of weights)
    """
    th = math.radians(float(binder_angle_deg))
    theta = [th] * (num_nodes - 1)
    length = [1.0] * (num_nodes - 1)
    return theta, length





