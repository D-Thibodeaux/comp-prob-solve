import numpy as np 
#define function to calculate distance given 2 3D points
def compute_bond_length(coord1, coord2):
    """"
    Given two Cartesian points representing two atoms, calculate the distance between them. 
    Reject any distance greater than 2 as being unreasonable for a covalent bond

    Parameters:
    coord1 (list): list containg position of first atom in angstroms
    coord2 (list): list containg position of second atom in angstroms

    Returns:
    None if distange is greater than 2
    distance  in angstroms if distance is less than 2
    """
    #define vector from point 1 to point 2
    dst = np.array(coord2)-np.array(coord1)

    #find magnitude of vector
    dst = np.sqrt(dst.dot(dst))

    #print warning if distance if over 2
    if dst > 2:
        #print("Warning: bond distance is greater than is expected for a covalent bond")
        return dst
    else:
        return dst
    
# define function to measue bond angle given three atoms 
def compute_bond_angle(coord1, coord2, coord3):
    """
    Compute the angle between three cartesian points.

    Parameters:
    coord1 (list): position of the first non-central atom
    coord2 (list): position of the central atom
    coord3 (list): position of the secons non-central atom

    Returns:
    Value of the angle between the three points in degrees
    """
    #define 2 vectors originating at the middle atom
    v1 = np.array(coord1) - np.array(coord2)
    v2 = np.array(coord3) - np.array(coord2)

    #find angle and convert to degrees
    cos = v1.dot(v2) / (np.sqrt(v1.dot(v1)) * np.sqrt(v2.dot(v2)))
    ang = np.degrees(np.arccos(cos))

    return ang

