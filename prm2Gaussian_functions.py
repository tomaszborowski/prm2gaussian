#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 07:53:26 2021

This is a collection of functions for prm2Gaussian

@author: borowski
"""
import numpy as np

# angle eq values are read in from prmtop in [rad], whereas in G16 they must be given in deg
def rad_to_deg(angle):
    """
    converts angle from radians to degrees
    returns values in the 0 ... 360 range
    """
    return np.mod((180.0 * angle)/np.pi,360.0)

def coord_to_atom_index(coord_index):
    """
    converts coordinate array index 
    to atom index
    """
    return int(coord_index/3) + 1

def remove_redundant_data(data,at_ix):
    """ 
    removes redundant entries from bonds_data, angles_data or dihedral_data
    INPUT:    
    data - list with data, atom types are grouped in a tuple
    at_ix - index (of data) for atom types tuple
    RETURNS:
    a list with nonredundant data: data_non_redund    
    """
    data_length = len(data)
    data_non_redund = []
    if data_length > 0:
        aux_non_redundant = [True for i in range(data_length)]        
        i = 0
        while i < data_length:
            raw_i = data[i]
            j = i + 1
            while j < data_length:
                raw_j = data[j]
                if (raw_i[at_ix] == raw_j[at_ix]) and (raw_i[0] == raw_j[0]):
                     aux_non_redundant[j] = False    
                elif (raw_i[at_ix] == tuple(reversed(raw_j[at_ix])) and (raw_i[at_ix][0] != raw_j[at_ix][0])):
                     aux_non_redundant[j] = False   
                j += 1
            i += 1
        
        for i in range(data_length):
            if aux_non_redundant[i]:
                data_non_redund.append(data[i])
    return data_non_redund

def remove_eq_imp(imp_data):
    """ 
    improper angles A-B-X-Y and B-A-X-Y are equivalent, Gaussian expects only one of them;
    this function reads imp_data (a list), and returns a list (imp_data_no_eq) with one such 
    if such redundancy is encountered.
    INPUT:    
    imp_data - list with data for improper angles, atom types are grouped in a tuple
    RETURNS:
    a list with nonredundant data: imp_data_no_eq    
    """
    data_length = len(imp_data)
    imp_data_no_eq = []
    if data_length > 0:
        aux_non_redundant = [True for i in range(data_length)]        
        i = 0
        while i < data_length:
            raw_i = imp_data[i]
            j = i + 1
            while j < data_length:
                raw_j = imp_data[j]
                if (raw_i[3][2:4] == raw_j[3][2:4]) and (raw_i[3][0] == raw_j[3][1])\
                    and (raw_i[3][1] == raw_j[3][0]):
                     aux_non_redundant[j] = False        
                j += 1
            i += 1
        
        for i in range(data_length):
            if aux_non_redundant[i]:
                imp_data_no_eq.append(imp_data[i])
    return imp_data_no_eq    

def remove_eq_dih(dih_data):
    """ 
    ordinary dihedral angles A-X-Y-A and A-Y-X-A are equivalent (not impropers!) and 
    Gaussian expects only one of them;
    this function reads dih_data (a list), and returns a list (dih_data_no_eq) 
    only with one such if such a redundancy is encountered.
    INPUT:    
    dih_data - list with data for normal dihedral angles, atom types are grouped in a tuple
    RETURNS:
    a list with nonredundant data: dih_data_no_eq    
    """    
    data_length = len(dih_data)
    dih_data_no_eq = []
    if data_length > 0:
        aux_non_redundant = [True for i in range(data_length)]        
        i = 0
        while i < data_length:
            raw_i = dih_data[i]
            j = i + 1
            while j < data_length:
                raw_j = dih_data[j]
                # case A-A-A-A:
                if (raw_i[3][0] == raw_i[3][1] == raw_i[3][2] == raw_i[3][3] \
                 == raw_j[3][0] == raw_j[3][1] == raw_j[3][2] == raw_j[3][3]) \
                 and (raw_i[0] != raw_j[0]):
                     pass # aux_non_redundant[j] = True
                # other cases A-X-Y-A:
                elif (raw_i[3][0] == raw_j[3][0] == raw_i[3][3] == raw_j[3][3])\
                   and (raw_i[3][1] == raw_j[3][2]) and (raw_i[3][2] == raw_j[3][1]):
                     aux_non_redundant[j] = False        
                j += 1
            i += 1
        
        for i in range(data_length):
            if aux_non_redundant[i]:
                dih_data_no_eq.append(dih_data[i])
    return dih_data_no_eq 
        
def is_3rd_atom_central(atom_list,connect_list):
    """
    INPUT:
        atom_list - a list of (4) zero-based atom indexes 
        connect_list - a dictionary key: zero based atom index; value: a list
                       (zero based) indexes of atom bonded to a given atom
    RETURNS:
        TRUE if the 3rd atom is the central one (bonded to first, second and fourth)
        FALSE otherwise
    """
    con_to_3rd = connect_list[ atom_list[2] ]
    if (atom_list[0] in con_to_3rd) and (atom_list[1] in con_to_3rd) and (atom_list[3] in con_to_3rd):
        return True
    else:
        return False

def write_xyz_file(atom_list, file_name):
    """writes an xyz file with a name file_name
    atom_list - a list of atom objects """
    n_atoms = len(atom_list)
    xyz_file = open(file_name, 'w')
    empty_l = '\n'
    xyz_file.write(str(n_atoms)+'\n')
    xyz_file.write(empty_l)
    for atom in atom_list:
        ele = atom.get_element()
        at_coord = atom.get_coords()
        line = ele + '\t' +\
        '{:06.6f}'.format(at_coord[0]) + '     ' +\
        '{:06.6f}'.format(at_coord[1]) + '     ' +\
        '{:06.6f}'.format(at_coord[2]) + '\n'
        xyz_file.write(line)
    xyz_file.close()

def write_xyz_file_MM_LA(link_at_list, file_name):
    """writes an xyz file with a name file_name
    link_at_list - a list of link_atom objects 
    coordinates taken from coords attribute of link_atom (not H_coords)"""
    n_atoms = len(link_at_list)
    xyz_file = open(file_name, 'w')
    empty_l = '\n'
    xyz_file.write(str(n_atoms)+'\n')
    xyz_file.write(empty_l)
    for atom in link_at_list:
        ele = atom.get_mm_element()
        at_coord = atom.get_MM_coords()
        line = ele + '\t' +\
        '{:06.6f}'.format(at_coord[0]) + '     ' +\
        '{:06.6f}'.format(at_coord[1]) + '     ' +\
        '{:06.6f}'.format(at_coord[2]) + '\n'
        xyz_file.write(line)
    xyz_file.close()
    
def adjust_HLA_coords(H_lk_atm, qm_atom, bl_sf=0.723886):
    """
    calculates and sets coordinates of the H link atom using the provided scaling factor
    H_lk_atm : Hydrogen link_atom
    qm_atom : qm atom to which the link atom is connected to 
    bl_sf : float, bond length scaling factor
    """
    qm_at_coords = np.array(qm_atom.get_coords())
    link_at_coords = np.array(H_lk_atm.get_MM_coords())
    H_coords = list( (1.0 - bl_sf)*qm_at_coords + bl_sf*link_at_coords )
    H_lk_atm.set_H_coords(H_coords)
    
def res2atom_lists(residues):
    """takes a list of residue objects as argument and 
    returns a list of atom objects (atoms contained in these residues)"""
    atm_list = []
    for res in residues:
        for at in res.get_atoms():
            atm_list.append(at)
    return atm_list

def not_trimmed_res2atom_lists(residues):
    """takes a list of residue objects as argument and 
    returns a list of atom objects in residues that has 
    attribute trim = False (atoms contained in these residues)"""
    atm_list = []
    for res in residues:
        if not res.get_trim():
            for at in res.get_atoms():
                atm_list.append(at)
    return atm_list    