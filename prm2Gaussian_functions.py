#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a collection of functions for prm2Gaussian

@authors: borowski, wojdyla
Last update on 18/05/2023
Last update on 7/11/2025
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
    improper angles A-B-X-Y, B-A-X-Y, Y-A-X-B, A-Y-X-B, B-Y-X-A and Y-B-X-A are all equivalent, 
    Gaussian expects only one of them;
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
            raw_i_3rd = raw_i[3][2]
            raw_i_124 = [raw_i[3][0], raw_i[3][1], raw_i[3][3]]
            raw_i_124s = set(raw_i_124)
            j = i + 1
            while j < data_length:
                raw_j = imp_data[j]
                raw_j_3rd = raw_j[3][2]
                raw_j_124 = [raw_j[3][0], raw_j[3][1], raw_j[3][3]]
                raw_j_124s = set(raw_j_124)  
                if (raw_i_3rd == raw_j_3rd) and (raw_i_124s == raw_j_124s):
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

def write_pdb_file(residue_list, file_name, write_Q=False):
    """writes a PDB file with a name file_name
    residue_list - a list of residue objects 
    only residues with trim=False are written to the file"""
    prev_resid_chain = ''
    pdb_file = open(file_name, 'w')
    residue_list_no_trim = []
    for residue in residue_list:
        if not residue.get_trim():
            residue_list_no_trim.append(residue)
    for residue in residue_list_no_trim:
        resid_name = residue.get_label()
        resid_name = resid_name.rjust(3, ' ')
        if residue.get_new_index():
            resid_number = str(residue.get_new_index() + 1)
        else:
            resid_number = str(residue.get_index() + 1)
        resid_number = resid_number.rjust(4, ' ')
        for atom in residue.get_atoms():
            ele = atom.get_element()
            ele = ele.rjust(2, ' ')
            at_coord = atom.get_coords()
            at_name = atom.get_name()
            at_name = at_name.ljust(4, ' ')
            if atom.get_new_index():
                at_number = atom.get_new_index() + 1
            else:
                at_number = atom.get_index() + 1
            at_number = str(at_number)
            at_number = at_number.rjust(5, ' ')
            at_layer = atom.get_oniom_layer()
            at_LAH = atom.get_LAH()
            if at_LAH:
               at_beta = 1.0 
            elif at_layer == 'H':
                at_beta = 2.0
            else: 
                at_beta = 0.0
            at_frozen = atom.get_frozen()    
            if at_frozen == 0:
                at_occupancy = 1.0
            else:
                at_occupancy = 0.0    
            at_charge = atom.get_at_charge()
            line = 'ATOM' + '  ' + at_number + ' ' +\
            at_name + ' ' + resid_name + '  ' + resid_number + '    ' +\
            '{:8.3f}'.format(at_coord[0]) + '{:8.3f}'.format(at_coord[1]) + '{:8.3f}'.format(at_coord[2]) +\
            '{:6.2f}'.format(at_occupancy) + '{:6.2f}'.format(at_beta) + '    ' + ele            
            if write_Q:
                line = line + ' ' + '{:5.3f}'.format(at_charge)
            line = line + '\n'
            pdb_file.write(line)
        if residue.get_new_index():
            residue_index = residue.get_new_index()
        else:
            residue_index = residue.get_index()
        if residue_index == 0:
            prev_resid_chain = residue.get_chain()
            if residue_list_no_trim[1].get_chain() != prev_resid_chain:
                pdb_file.write('TER\n')
        elif residue.get_chain() != prev_resid_chain:
            prev_resid_chain = residue.get_chain()
            pdb_file.write('TER\n')
        else:
            pass
    pdb_file.close()    
    

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


def print_help():  
    help_text = """
prm2Gaussian script: 
reads Amber prmtop and prmcrd(rst7) files and writes
Gaussian16 Amber=softonly or ONIOM(QM,Amber=softonly) input file

The script expects 3 file names as command line arguments:
            #1 prmtop file
            #2 prmcrd(rst7) file
            #3 output G16 input file
optional    #4 input file for this script  
    """
    
    print(help_text)