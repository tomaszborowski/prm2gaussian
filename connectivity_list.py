#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Set of functions useful when generating connectivity list section of Gaussian input file
Currently gen_connectivity_line asscribes only bond order = 1.0 (which is sufficient for Amber FF)

Created on Wed Oct 14 11:20:42 2020

@author: borowski
"""
import numpy as np

BOND_RADII_SCALE_FACTOR = 1.3 # 130% of (R_i + R_j) as a threshold to consider atoms bonded

# cov_radii - a dictionary with element symbols as keys and cavalent radii [A] as values
# (up to 86 - Rn) (source: https://en.wikipedia.org/wiki/Covalent_radius)
cov_radii = {\
'H':0.31,'He':0.28,\
'Li':1.28,'Be':0.96,'B':0.84,'C':0.76,'N':0.71,'O':0.66,'F':0.57,'Ne':0.58,\
'Na':1.66,'Mg':1.41,'Al':1.21,'Si':1.11,'P':1.07,'S':1.05,'Cl':1.02,'Ar':1.06,\
'K':2.03,'Ca':1.76,'Sc':1.70,'Ti':1.60,'V':1.53,'Cr':1.39,'Mn':1.61,'Fe':1.52,'Co':1.50,'Ni':1.24,'Cu':1.32,'Zn':1.22,'Ga':1.22,'Ge':1.20,'As':1.19,'Se':1.20,'Br':1.20,'Kr':1.16,\
'Rb':2.20,'Sr':1.95,'Y':1.90,'Zr':1.75,'Nb':1.64,'Mo':1.54,'Tc':1.47,'Ru':1.46,'Rh':1.42,'Pd':1.39,'Ag':1.45,'Cd':1.44,'In':1.42,'Sn':1.39,'Sb':1.39,'Te':1.38,'I':1.39,'Xe':1.40,\
'Cs':2.44,'Ba':2.15,'Hf':1.87,'Ta':1.70,'W':1.62,'Re':1.51,'Os':1.44,'Ir':1.41,'Pt':1.36,'Au':1.36,'Hg':1.32,'Tl':1.45,'Pb':1.46,'Bi':1.48,'Po':1.40,'At':1.50,'Rn':1.50,\
'La':2.07,'Ce':2.04,'Pr':2.03,'Nd':2.01,'Pm':1.99,'Sm':1.98,'Eu':1.98,'Gd':1.96,'Tb':1.94,'Dy':1.92,'Ho':1.92,'Er':1.89,'Tm':1.90,'Yb':1.87,'Lu':1.75} 

     
def coord_diffs(x1,y1,z1,x2,y2,z2):
    """
    calculates absolute value differences in coordinates

    Parameters
    ----------
    xi, yi, zi - float, (x,y,z) coordinates of point i

    Returns
    -------
    abs(delta_x), abs(delta_y), abs(delta_z)

    """
    dx = np.abs(x1-x2)
    dy = np.abs(y1-y2)
    dz = np.abs(z1-z2)
    return dx, dy, dz


def coord_lists_diffs(L1, L2):
    """
    calculates absolute value differences in coordinates

    Parameters
    ----------
    L1, L2 - lists of coordinates: Li = [xi, yi, zi]

    Returns
    -------
    abs(delta_x), abs(delta_y), abs(delta_z)

    """    
    dx = np.abs(L1[0]-L2[0])
    dy = np.abs(L1[1]-L2[1])
    dz = np.abs(L1[2]-L2[2])
    return dx, dy, dz
 
    
def diff_dist(dx,dy,dz):
    """
    calculates distance from coordinate differences

    Parameters
    ----------
    float dx : (x1-x2), dy : (y1-y2), dz : (z1-z2)

    Returns
    -------
    TYPE : float
        distance between two points

    """
    return np.sqrt(dx**2 + dy**2 + dz**2)


def diff_dist2(dxdydz):
    """
    calculates distance from coordinate differences

    Parameters
    ----------
    tuple of floats dxdydz : (x1-x2, y1-y2, z1-z2)

    Returns
    -------
    TYPE : float
        distance between two points

    """
    return np.sqrt(dxdydz[0]**2 + dxdydz[1]**2 + dxdydz[2]**2)


def are_atoms_bonded(symbol1, coords1, symbol2, coords2):
    """ checks if two atoms are bonded or not based on their distance
    Parameters
    ----------
    symboli - string, element symbol
    coordsi - list of coordinates of atom i [x_i, y_i, z_i]
    Returns
    -------
    boolean: TRUE or FALSE
    """
    R = cov_radii[symbol1] + cov_radii[symbol2]
    Rs = BOND_RADII_SCALE_FACTOR * R
    DxDyDz = coord_lists_diffs(coords1, coords2)
    dist = 100.0
    if max(DxDyDz) <= Rs:
        dist = diff_dist2(DxDyDz)
    if dist <= Rs:
        return True
    else:
        return False
    

def gen_connectivity_line(ind, lista):
    """
    takes a list lista of indices (zero-based) of atoms bonded to a given atom of
    (zero-based) index ind and returns a string formated as a line of connectivity
    section of Gaussian input file. 
    All bond orders are set as 1.0

    Parameters
    ----------
    ind : (zero-based) index of a given atom
    lista : list of indices of atoms bonded to a given atom

    Returns
    -------
    connect_line - string variable 

    """
    sep = ' '
    b_ord = '1.0'
    connect_line = sep + str(ind + 1) + sep
    for i in range(len(lista)):
        connect_line = connect_line + str(lista[i] + 1) + sep + b_ord + sep
    connect_line = connect_line + '\n'
    return connect_line


# execute only if run as a script
if __name__ == "__main__":
    pass # add some testing
