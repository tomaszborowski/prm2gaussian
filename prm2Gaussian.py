#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the main code for the script that is supposed
to read Amber prmtop and prmcrd(rst7) files and write
Gaussian16 Amber=softonly input file

The script expects 3 file names as command line arguments:
    #1 prmtop file
    #2 prmcrd(rst7) file
    #3 output G16 input file

Warnings:
1. bond order is uniformly 1.0 for all bonded atom pairs in the connectivity section    

2. 1-4 interactions are all the same (as given by the NonBon line); at the moment
the script does not use the information stored in the scee_scale_factor and scnb_scale_factor
(might be useful to explore this for systems where not all 1-4 interactions are treated 
in the same way, but currently in G09 and G16 Revision C.01 the flags nbdir/nbterm are
not accepted in input)

3. total spin is calculated as either 1 or 2, depending on the total charge and atomic
composition

4. dihedral term accepts periodicity from 1 to 4 (as in standard Amber and GAFF FF), 
higher values are ignored (warning is printed)

5. if atomic number read from prmtop is smaller than 1, it is corrected based on
the mass 

6. adds zero force constants for angles in triangular water molecules if explicit 
HW-HW bond data is present    

Created on Fri Oct  9 10:27:30 2020
Last update on 24/03/2021

@author: borowski, wojdyla
"""
import sys
import numpy as np
import pandas as pd
import datetime
import string

from read_prmtop import prmtop_read_pointers, prmtop_read_text_section
from read_prmtop import prmtop_read_numeric_section, crd_read_coordinates
from read_prmtop import atm_mass, atm_number, at_num_symbol
from read_prmtop import LEGIT_TEXT_FLAGS, LEGIT_NUM_FLAGS

from connectivity_list import gen_connectivity_line

# CONSTANTS
letters = string.ascii_uppercase
digits = string.digits

HW_HW_bond = ('HW', 'HW')

# Important variables (switches):
VERBOSE = False
HW_HW_present = False


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

        
### ---------------------------------------------------------------------- ###
### Seting the file names                                                  ###
prmtop_file = sys.argv[1]
prmcrd_file = sys.argv[2]
g16_inp_file = sys.argv[3]


### ---------------------------------------------------------------------- ###
### test cases
        
# prmtop_file = './pliki_do_testow/dihydroclavaminate/dihydroclavaminate.prmtop'
# prmcrd_file = './pliki_do_testow/dihydroclavaminate/dihydroclavaminate.prmcrd'
# g16_inp_file = './pliki_do_testow/dihydroclavaminate/dihydroclavaminate_22_03_beta.com'

# prmtop_file = './pliki_do_testow/H6H/h6h-oxo+succinate+water_hyo.nowat.prmtop'
# prmcrd_file = './pliki_do_testow/H6H/h6h-oxo+succinate+water_hyo.nowat.prmcrd'
# g16_inp_file = './pliki_do_testow/H6H/h6h-oxo+succinate+water_hyo.nowat_24_03.com'

# prmtop_file = './pliki_do_testow/cellulose/NAG.prmtop'
# prmcrd_file = './pliki_do_testow/cellulose/first_frame.rst7'
# g16_inp_file = './pliki_do_testow/cellulose/first_frame.com'
    
# prmtop_file = './pliki_do_testow/2_atom_systems/coulomb/AB_coulomb.prmtop'
# prmcrd_file = './pliki_do_testow/2_atom_systems/coulomb/AB_coulomb.prmcrd'
# g16_inp_file = './pliki_do_testow/2_atom_systems/coulomb/AB_coulomb.com'

# prmtop_file = './pliki_do_testow/4_atom_systems/ABCD_2.prmtop'
# prmcrd_file = './pliki_do_testow/4_atom_systems/ABCD_2.prmcrd'
# g16_inp_file = './pliki_do_testow/4_atom_systems/ABCD_2.com'

# prmtop_file = './pliki_do_testow/n_butan/n_butan_1.prmtop'
# prmcrd_file = './pliki_do_testow/n_butan/n_butan_1.prmcrd'
# g16_inp_file = './pliki_do_testow/n_butan/n_butan_1.com'

# prmtop_file = './pliki_do_testow/cl_prop/cl_prop.prmtop'
# prmcrd_file = './pliki_do_testow/cl_prop/cl_prop.prmcrd'
# g16_inp_file = './pliki_do_testow/cl_prop/cl_prop.com'

# prmtop_file = './pliki_do_testow/c2h4/c2h4.prmtop'
# prmcrd_file = './pliki_do_testow/c2h4/c2h4.prmcrd'
# g16_inp_file = './pliki_do_testow/c2h4/c2h4.com'

# prmtop_file = './pliki_do_testow/clcobr/clcobr.prmtop'
# prmcrd_file = './pliki_do_testow/clcobr/clcobr.prmcrd'
# g16_inp_file = './pliki_do_testow/clcobr/clcobr.com'

### ---------------------------------------------------------------------- ###
### Reading from prmtop file                                               ###

# open the prmtop file and read-in pointers from the prmtop file:
if VERBOSE:
    print("Program started at: ", datetime.datetime.now(), "\n")

prmtop = open(prmtop_file, 'r')
pointers = prmtop_read_pointers(prmtop)

# calculate and populate exp_length dictionary based on the pointers
NATOM = pointers['NATOM']
NTYPES = pointers['NTYPES']
NRES = pointers['NRES']   
NUMBND = pointers['NUMBND']
NUMANG = pointers['NUMANG']
NPTRA = pointers['NPTRA']
NATYP = pointers['NATYP']
NBONH = pointers['NBONH']
NBONA = pointers['NBONA']
NTHETH = pointers['NTHETH']
NTHETA = pointers['NTHETA']
NPHIH = pointers['NPHIH']
NPHIA = pointers['NPHIA']
NNB = pointers['NNB']
NPHB = pointers['NPHB']
try:
    NSPM = prmtop_read_numeric_section(prmtop, FLAG='SOLVENT_POINTERS',exp_length=3)[1]
except AssertionError:
    NSPM = None 
#
exp_length = {'ATOM_NAME' : NATOM, 'CHARGE' : NATOM, 'ATOMIC_NUMBER' : NATOM,\
              'MASS' : NATOM, 'ATOM_TYPE_INDEX' : NATOM, 'NUMBER_EXCLUDED_ATOMS' : NATOM,\
              'NONBONDED_PARM_INDEX' : NTYPES**2, 'RESIDUE_LABEL' : NRES,\
              'RESIDUE_POINTER' : NRES, 'BOND_FORCE_CONSTANT' : NUMBND,\
              'BOND_EQUIL_VALUE' : NUMBND, 'ANGLE_FORCE_CONSTANT' : NUMANG,\
              'ANGLE_EQUIL_VALUE' : NUMANG, 'DIHEDRAL_FORCE_CONSTANT' : NPTRA,\
              'DIHEDRAL_PERIODICITY' : NPTRA, 'DIHEDRAL_PHASE' : NPTRA,\
              'SCEE_SCALE_FACTOR' : NPTRA, 'SCNB_SCALE_FACTOR' : NPTRA,\
              'SOLTY' : NATYP, 'LENNARD_JONES_ACOEF' : int((NTYPES*(NTYPES+1))/2),\
              'LENNARD_JONES_BCOEF' : int((NTYPES*(NTYPES+1))/2), 'BONDS_INC_HYDROGEN' : 3*NBONH,\
              'BONDS_WITHOUT_HYDROGEN' : 3*NBONA, 'ANGLES_INC_HYDROGEN' : 4*NTHETH,\
              'ANGLES_WITHOUT_HYDROGEN' : 4*NTHETA, 'DIHEDRALS_INC_HYDROGEN' : 5*NPHIH,\
              'DIHEDRALS_WITHOUT_HYDROGEN' : 5*NPHIA, 'EXCLUDED_ATOMS_LIST' : NNB,\
              'HBOND_ACOEF' : NPHB, 'HBOND_BCOEF' : NPHB, 'HBCUT' : NPHB, 'AMBER_ATOM_TYPE' : NATOM,\
              'TREE_CHAIN_CLASSIFICATION' : NATOM, 'JOIN_ARRAY' : NATOM, 'IROTAT' : NATOM,\
              'SOLVENT_POINTERS' : 3, 'ATOMS_PER_MOLECULE' : NSPM, 'BOX_DIMENSIONS' : 4,\
              'CAP_INFO' : 1, 'CAP_INFO2' : 4, 'RADIUS_SET' : 1, 'RADII' : NATOM,\
              'IPOL' : 1, 'POLARIZABILITY' : NATOM, 'SCREEN' : NATOM}


# will read to two dictionaries that contain data read from prmtop file,
# keys are flags in lower case

# read text sections of prmtop:    
prmtop_text_sections = {}
for flag in LEGIT_TEXT_FLAGS:
     prmtop_text_sections[flag.lower()] = prmtop_read_text_section(prmtop, flag, exp_length[flag])


# modify the content of LEGIT_NUM_FLAGS based on the pointers values, i.e. 
# remove those flags (pertaining to numeric sections) that are not present 
# in the read-in prmtop file
if pointers['IFBOX'] == 0:
    for item in ['SOLVENT_POINTERS', 'ATOMS_PER_MOLECULE', 'BOX_DIMENSIONS']:
        LEGIT_NUM_FLAGS.remove(item)

if pointers['IFCAP'] == 0:
    for item in ['CAP_INFO', 'CAP_INFO2']:
        LEGIT_NUM_FLAGS.remove(item)        

if prmtop_read_numeric_section(prmtop, 'IPOL', 1)[0] == 0:
    LEGIT_NUM_FLAGS.remove('POLARIZABILITY')

# read numeric sections from prmtop:
prmtop_num_sections = {}
for flag in LEGIT_NUM_FLAGS:
     prmtop_num_sections[flag.lower()] = prmtop_read_numeric_section(prmtop, flag, exp_length[flag])

# after reading close the prmtop file: 
prmtop.close()
if VERBOSE:
    print("prmtop file has been read ", datetime.datetime.now(), "\n")

### ---------------------------------------------------------------------- ###
### Reading from prmcrd / rst7 file                                        ###

# read number of atoms and coordinates from the prmcrd (rst7) file:
prmcrd = open(prmcrd_file, 'r')
natom_crd, coordinates = crd_read_coordinates(prmcrd)
prmcrd.close()

if VERBOSE:
    print("prmcrd/rst7 file has been read ", datetime.datetime.now(), "\n")


# check if the number of atoms matches that read from prmtop file:
if natom_crd != NATOM:
    print("ATTANTION, the number of atoms read from prmtop and prmcrd(rst7) files DO NOT MATCH !!!\n")
    print("from prmtop: ", NATOM, " \n")
    print("from prmcrd: ", natom_crd, " \n")
    

### ---------------------------------------------------------------------- ###
### Processing the data                                                    ###

# converting atomic charges to atomic units:
for i in range(len(prmtop_num_sections['charge'])):
    prmtop_num_sections['charge'][i] /= 18.2223

# converting angle equilibrium values from radians to degrees:
for i in range(len(prmtop_num_sections['angle_equil_value'])):
    prmtop_num_sections['angle_equil_value'][i] = \
    rad_to_deg(prmtop_num_sections['angle_equil_value'][i])
    
# dihedral phase is read in [rad], convert to degrees:
for i in range(len(prmtop_num_sections['dihedral_phase'])):
    prmtop_num_sections['dihedral_phase'][i] = \
    rad_to_deg(prmtop_num_sections['dihedral_phase'][i])

# check if entries in atomic_number are OK, if not, infer from mass
for i in range(NATOM):
    if prmtop_num_sections['atomic_number'][i] < 1:
        print('Warning, atom #: ', i, ' has atomic number: ', prmtop_num_sections['atomic_number'][i])
        print('I try to get the correct value based on mass')
        mass_x = prmtop_num_sections['mass'][i]
        min_mass_diff = 300.0
        min_mass_diff_element = None
        for element, mass in zip(atm_mass.keys(), atm_mass.values()):
            mass_diff = abs(mass - mass_x)
            if mass_diff < min_mass_diff:
                min_mass_diff = mass_diff
                min_mass_diff_element = element
        prmtop_num_sections['atomic_number'][i] = atm_number[min_mass_diff_element]
        print('Assigned atomic number is: ' + str(atm_number[min_mass_diff_element]) +\
              ' ( ' + min_mass_diff_element + ' )')

# check if atom types are all unique and start with a letter when CAPITALIZED 
# G16 does not accepts types starting with numbers and does not distinguish small
# and capital letters, using ' ' or " " does not help (for G09 the latter worked)            

# from prmtop_text_sections['amber_atom_type'] select a unique set of atom types
unique_types = list( set(prmtop_text_sections['amber_atom_type']) )
num_unq_types =  len(unique_types)

# capitalize all letters:
unique_types_CAP = []
for atom_type in unique_types:
    unique_types_CAP.append(atom_type.upper())

# for type starting with a digit add 'A' at the front: 
unique_types_temp = []
for atom_type in unique_types_CAP:
    if atom_type[0] in digits:
        new_atom_type = 'A' + atom_type
        unique_types_temp.append(new_atom_type)
    else:
        unique_types_temp.append(atom_type)

# test for repeating type symbols and append letters from letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
unique_types_NEW = unique_types_temp.copy()       
for i in range(num_unq_types):
    typ_i = unique_types_temp[i]
    for j in range(i+1, num_unq_types):
        typ_j = unique_types_temp[j]
        if typ_i == typ_j:
            for letter in letters:
                try_new_typ_j = typ_j + letter
                if try_new_typ_j not in set(unique_types_NEW):
                    unique_types_NEW[j] = try_new_typ_j 
                    break

# test if 1:1 mapping between unique_types and unique_types_NEW:
if len(set(unique_types_NEW)) != num_unq_types:
    print('ATTENTION, the mapping between original atom types (unique_types) and new\
          atom types - to be used in Gaussian (unique_types_NEW) is not 1:1 !!!')

# create dictionary translating original types to the new ones:
unq_to_NEW_types = dict( zip(unique_types, unique_types_NEW) )          

# print the mapping between the original and new atom types:
print('Applied mapping from original atom types to a set suitable for G16: \n')
for key in unq_to_NEW_types.keys():
    print(str(key) + ' ---> ' + str(unq_to_NEW_types[key]))

if VERBOSE:
    print("unique_types created ", datetime.datetime.now(), "\n")            
    print("initial processing done ", datetime.datetime.now(), "\n")

# connectivity list section
# connect_list - a dic with (int) keys - atom numbers (indexed from 0)
# and values - lists of atoms connected to the given atom (indexed from 0)
connect_list = {}
# read data from prmtop_num_sections['bonds_inc_hydrogen']:
for i in range(NBONH):
    atom_1 = coord_to_atom_index( prmtop_num_sections['bonds_inc_hydrogen'][3*i] ) - 1
    atom_2 = coord_to_atom_index( prmtop_num_sections['bonds_inc_hydrogen'][3*i+1] ) - 1 
    if atom_1 in connect_list.keys():
        connect_list[atom_1].append(atom_2)
    else:
        connect_list[atom_1] = [atom_2]
    if atom_2 in connect_list.keys():
        connect_list[atom_2].append(atom_1)
    else:
        connect_list[atom_2] = [atom_1]
# read data from prmtop_num_sections['bonds_without_hydrogen']:        
for i in range(NBONA):
    atom_1 = coord_to_atom_index( prmtop_num_sections['bonds_without_hydrogen'][3*i] ) - 1
    atom_2 = coord_to_atom_index( prmtop_num_sections['bonds_without_hydrogen'][3*i+1] ) - 1
    if atom_1 in connect_list.keys():
        connect_list[atom_1].append(atom_2)
    else:
        connect_list[atom_1] = [atom_2]
    if atom_2 in connect_list.keys():
        connect_list[atom_2].append(atom_1)
    else:
        connect_list[atom_2] = [atom_1]
        
# add empty lists for atoms not making any bonds
for i in range(NATOM):
    if i not in connect_list.keys():
        connect_list[i] = []
        
if VERBOSE:
    print("connect_list created ", datetime.datetime.now(), "\n")
        
# sort lists of connected atoms:
for i in range(NATOM):
    connect_list[i].sort()
if VERBOSE:
    print("connect_list sorted ", datetime.datetime.now(), "\n")

# read bonds information into a Pandas DataFrame with name bonds
# first_instance - a flag marking if this is the first occurance of a given type (now it's redundant)
columns = ['ind_to_tabels', 'Atom_types',\
           'force_constant', 'equil_value', 'first_instance']    
# gather the required info in an auxiliary list bonds_data:
bonds_data = []

for i in range(NUMBND):
    force_constant = prmtop_num_sections['bond_force_constant'][i]
    equil_value = prmtop_num_sections['bond_equil_value'][i]
    first_instance = True
    temp = [None,None,None]
    temps_i = []
    for bond_type in ['bonds_inc_hydrogen', 'bonds_without_hydrogen']:
        for j in range(int(round(len(prmtop_num_sections[bond_type])/3))):
                temp = prmtop_num_sections[bond_type][3*j : 3*j+3]
                if temp[-1] == i+1:
                    temps_i.append(temp)
    raws = []
    for temp in temps_i:
        atom_numbers = [int(round(n/3)) for n in temp[0:2]]
        atom_types = tuple( [prmtop_text_sections['amber_atom_type'][k] for k in atom_numbers] )
        raw = [i+1, atom_types, force_constant, equil_value, first_instance]
        raws.append(raw)
    unique_raws_tup = set(tuple(row) for row in raws)
    unique_raws = [list(raw) for raw in unique_raws_tup]
    bonds_data = bonds_data + unique_raws

# remove redundant data, e.g. one of the pair: (A-B), (B-A)
bonds_data_non_redund = remove_redundant_data(bonds_data,1)

if VERBOSE:
    print("auxiliary list bonds_data_non_redundant done ", datetime.datetime.now(), "\n")

# sorting bonds_data wrt (atom_types):
bonds_data_non_redund.sort(key = lambda x: x[1])    

if VERBOSE:
    print("auxiliary list bonds_data_non_redund sorted ", datetime.datetime.now(), "\n")
        
# create DataFrame from the list    
bonds = pd.DataFrame(bonds_data_non_redund, columns=columns)   

if VERBOSE:
    print("bonds DataFrame created ", datetime.datetime.now(), "\n")

# check if HW-HW bond is present:
for bond in bonds.Atom_types:
    if bond == HW_HW_bond:
       HW_HW_present = True
       break

# read angles information into a Pandas DataFrame with name angles
columns = ['ind_to_tabels', 'Atom_types',\
           'force_constant', 'equil_value', 'first_instance']    
# first make auxiliary list angles_data:
angles_data = []

for i in range(NUMANG):
    force_constant = prmtop_num_sections['angle_force_constant'][i]
    equil_value = prmtop_num_sections['angle_equil_value'][i]
    first_instance = True
    temp = [None,None,None,None]
    temps_i = []
    for ang_type in ['angles_inc_hydrogen', 'angles_without_hydrogen']:
        for j in range(int(round(len(prmtop_num_sections[ang_type])/4))):
            temp = prmtop_num_sections[ang_type][4*j : 4*j+4]
            if temp[-1] == i+1:
                temps_i.append(temp)
    raws = []
    for temp in temps_i:  
        atom_numbers = [int(round(n/3)) for n in temp[0:3]]
        atom_types = tuple( [prmtop_text_sections['amber_atom_type'][k] for k in atom_numbers] )
        raw = [i+1, atom_types, force_constant, equil_value, first_instance]
        raws.append(raw)
    unique_raws_tup = set(tuple(row) for row in raws)
    unique_raws = [list(raw) for raw in unique_raws_tup]        
    angles_data = angles_data + unique_raws

# remove redundant data, e.g. one of the pair: (A-B-C), (C-B-A)
angles_data_non_redund = remove_redundant_data(angles_data,1)

# add zero force constants for angles in triangular water molecules 
# if explicit HW-HW bond data is present in the bonds_data_non_redundant
if HW_HW_present:
    angles_data_non_redund.append([0, ('OW', 'HW', 'HW'), 0.0, 0.0, True])
    angles_data_non_redund.append([0, ('HW', 'OW', 'HW'), 0.0, 0.0, True])
    print('Adding dummy angle force constants for triangulat WAT, since HW-HW bond present')

if VERBOSE:
    print("auxiliary list angles_data_non_redund done ", datetime.datetime.now(), "\n")

# sorting angles_data wrt (atom_types):
angles_data_non_redund.sort(key = lambda x: x[1])
        
# create DataFrame from the list     
angles = pd.DataFrame(angles_data_non_redund, columns=columns)
if VERBOSE:
    print("angles DataFrame created ", datetime.datetime.now(), "\n")

# read dihedral angles information into a list dihedral_data
# (for debugging also into a Pandas DataFrame with the name angles dihedrals)
columns = ['ind_to_tabels', 'if_nonbonded', 'is_improper', 'Atom_types',\
           'force_constant', 'periodicity', 'phase', 'scee_scale_factor', 'scnb_scale_factor',
           'first_instance']    
# first make auxiliary list dihedral_data:
dihedral_data = []
for i in range(NPTRA):
    force_constant = prmtop_num_sections['dihedral_force_constant'][i]
    periodicity = prmtop_num_sections['dihedral_periodicity'][i]
    phase = prmtop_num_sections['dihedral_phase'][i]
    scee_scale_factor = prmtop_num_sections['scee_scale_factor'][i]
    scnb_scale_factor = prmtop_num_sections['scnb_scale_factor'][i]
    first_instance = True
    temp = [None,None,None,None,None]
    temps_i = []
    for dih_type in ['dihedrals_inc_hydrogen', 'dihedrals_without_hydrogen']:
        for j in range(int(round(len(prmtop_num_sections[dih_type])/5))):
            temp = prmtop_num_sections[dih_type][5*j : 5*j+5]
            if temp[-1] == i+1:
                temps_i.append(temp)
    raws = []
    for temp in temps_i:
        if temp[2] < 0:
            if_nonbonded = False
        else:
            if_nonbonded = True
        if temp[3] < 0:
            is_improper = True
            atom_numbers = [int(np.abs(n)/3) for n in temp[0:4]]
#           here check if the 3rd atom of the improper is the central one
#           Gaussian requires the central atom is the 3rd in the improper quartet:
            if not is_3rd_atom_central(atom_numbers,connect_list):
                atom_numbers = list(reversed(atom_numbers))
                if not is_3rd_atom_central(atom_numbers,connect_list):
                    print("Houston, I have a problem with the improper angle:", atom_numbers)
        else:
            is_improper = False
            atom_numbers = [int(np.abs(n)/3) for n in temp[0:4]]
            
        atom_types = tuple( [prmtop_text_sections['amber_atom_type'][k] for k in atom_numbers] )  
        raw = [i+1, if_nonbonded, is_improper, atom_types,\
            force_constant, periodicity, phase, scee_scale_factor, scnb_scale_factor, first_instance]
        raws.append(raw)
        
    unique_raws_tup = set(tuple(row) for row in raws)
    unique_raws = [list(raw) for raw in unique_raws_tup]
    dihedral_data = dihedral_data + unique_raws

# split dihedral_data into proper_dihedral_data and improper_dihedral_data
# to avoid removing impropers for the same quartet of atoms as for proper dih or vice versa    

proper_dihedral_data = []
improper_dihedral_data = []
for entry in dihedral_data:
    if entry[2]:
        improper_dihedral_data.append(entry)
    else:
        proper_dihedral_data.append(entry)    

# now remove redundant entries, e.g. one of: (A-B-C-D), (D-C-B-A)
dihedral_data_non_redund = remove_eq_dih(remove_redundant_data(proper_dihedral_data,3))
improper_data_non_redund = remove_eq_imp(remove_redundant_data(improper_dihedral_data,3))

if VERBOSE:        
    print("auxiliary lists dihedral_data_non_redund and improper_data_non_redundant\
          done ", datetime.datetime.now(), "\n")

# sorting dihedral_data wrt (atom_types):
dihedral_data_non_redund.sort(key = lambda x: x[3])
improper_data_non_redund.sort(key = lambda x: x[3])    

if VERBOSE:
    print("auxiliary list dihedral_data_non_redund sorted ", datetime.datetime.now(), "\n")
        
# create DataFrames from the lists (for debuging, as these DataFrames are not used) 
# dihedrals = pd.DataFrame(dihedral_data_non_redund, columns=columns)    
# impropers = pd.DataFrame(improper_data_non_redund, columns=columns)
# if VERBOSE:            
#     print("dihedrals and impropers DataFrames created ", datetime.datetime.now(), "\n")


# extracting atomic vdW parameters:
    
# for each atom type:   
    # find atom index for atom of this type and a pointer to lennard_jones_a(b) coef for 
    # the pair: this type, this type
    # read A and B and convert into epsilon and r_m
#
# epsilon and r_min are dictionaries with amber atom types as keys and
# epsilon and r_min values, respectively
epsilon = {}
r_min = {}
for type in unique_types:
    # i - zero-based index of atom in an amber_atom_type type
    i = prmtop_text_sections['amber_atom_type'].index(type)
    # j = atom_type_index(i):
    j = prmtop_num_sections['atom_type_index'][i]
    # index - index into lennard_jones_a(b)coef for A_ii and B_ii values:
    index = prmtop_num_sections['nonbonded_parm_index'][(NTYPES*(j-1)+j)-1]
    # read A and B values:
    A = prmtop_num_sections['lennard_jones_acoef'][index-1]
    B = prmtop_num_sections['lennard_jones_bcoef'][index-1]
    # convert (A, B) into (epsilon, r_min) values:
    if A > 0.0 and B > 0.0:
        eps = B**2/(4*A)
        r_m = 0.5 * ((2*A/B)**(1/6))
    else:
        eps=0.0
        r_m=0.0
    # add the values to the dictionaries:
    epsilon[type] = eps
    r_min[type] = r_m

if VERBOSE:    
    print("epsilon and r_min created ", datetime.datetime.now(), "\n")

### ---------------------------------------------------------------------- ###
### Writing the output - G16 input file                                    ###

# open the file for writting
g16file = open(g16_inp_file, 'w')
empty_l = '\n'

# header section    
l1_g_input = '# Amber=(SoftOnly,Print) Geom=Connectivity \n'
l3_g_input = 'comment line \n'

# calculate the total charge and round it to nearest integer value:
tot_q = np.sum(prmtop_num_sections['charge'])
tot_q_int = int(round(tot_q))
print('Total charge of the system is: ', "{:.2e}".format(tot_q), ' , which is rounded to: ', tot_q_int)
if VERBOSE:
    print(datetime.datetime.now(), "\n")

# assuming the total spin is singlet or dublet (does not have any meaning for pure FF calculations)
n_electrons = np.sum(prmtop_num_sections['atomic_number']) - tot_q_int
if n_electrons % 2:
    S = 2
else:
    S = 1
    
l5_g_input = str(tot_q_int) + '    ' + str(S)  + '\n'

# header section - writting to file
g16file.write(l1_g_input)
g16file.write(empty_l)
g16file.write(l3_g_input)
g16file.write(empty_l)
g16file.write(l5_g_input)

# coordinates section
# iterate over atoms and generate appopriate line; write the line into the file    
for i in range(NATOM):    
    line = at_num_symbol[prmtop_num_sections['atomic_number'][i]] + '-' +\
        unq_to_NEW_types[ prmtop_text_sections['amber_atom_type'][i] ]  + '-' +\
        str(round(prmtop_num_sections['charge'][i], 6)) + '\t\t' +\
        '{:06.6f}'.format(coordinates[i][0]) + '     ' +\
        '{:06.6f}'.format(coordinates[i][1]) + '     ' +\
        '{:06.6f}'.format(coordinates[i][2]) + '\n'
    g16file.write(line)

    
# write separator empty line:
g16file.write(empty_l)
if VERBOSE:
    print("coordinate section written to file ", datetime.datetime.now(), "\n")
    

# write connectivity list into the gaussian input,
# a given connectivity is specified only once:
for i in range(NATOM):
    con_4_g16 = connect_list[i]
    for atom in con_4_g16.copy():
        if atom < i:
            con_4_g16.remove(atom)
    g16file.write(gen_connectivity_line(i, con_4_g16))

# write separator empty line:
g16file.write(empty_l)
if VERBOSE:
    print("connect_list written to file ", datetime.datetime.now(), "\n")

# FF specification section

# vdW parameters
sep = '     '
for type in unique_types:
    new_type = unq_to_NEW_types[type]
    vdw_line = 'VDW' + sep + new_type + sep + '{:06.6f}'.format(r_min[type]) + sep + '{:06.6f}'.format(epsilon[type]) + '\n'
    g16file.write(vdw_line)
        
if VERBOSE:
    print("vdW section written to file ", datetime.datetime.now(), "\n")

# force constant is not changed, as amber and Gaussian use the same format ( K*(r - req)**2 ) 
for index, raw in bonds.loc[bonds['first_instance']].iterrows():
    at_1 = unq_to_NEW_types[ raw['Atom_types'][0] ]
    at_2 = unq_to_NEW_types[ raw['Atom_types'][1] ]
    bond_line = 'HrmStr1' + sep + at_1 + sep + at_2+ sep +\
        '{:05.3f}'.format(round(raw['force_constant'], 3)) + sep +\
        '{:05.3f}'.format(round(raw['equil_value'], 3)) + '\n'    
    g16file.write(bond_line)
if VERBOSE:
    print("bond parameters written to file ", datetime.datetime.now(), "\n")

# angle parameters
# force constant is not changed, as amber and Gaussian use the same format ( K*(theta - thetaeq)**2 )    
for index, raw in angles.loc[angles['first_instance']].iterrows():
    at_1 = unq_to_NEW_types[ raw['Atom_types'][0] ]
    at_2 = unq_to_NEW_types[ raw['Atom_types'][1] ]
    at_3 = unq_to_NEW_types[ raw['Atom_types'][2] ]
    angle_line = 'HrmBnd1' + sep + at_1 + sep + at_2 + sep + at_3 + sep +\
        '{:05.3f}'.format(round(raw['force_constant'], 3)) + sep +\
        '{:05.3f}'.format(round(raw['equil_value'], 3)) + '\n'
    g16file.write(angle_line)
if VERBOSE:
    print("angle parameters written to file ", datetime.datetime.now(), "\n")

# proper dihedral parameters
# force constant and phase_offset are not changed, as Amber and Gaussian use the same format
sep = '   ' # 3 spaces separator for lengthy dihedral and improper lines    
fstr = '{:08.6f}'
proper_lines = []
dihedral_data_non_redund_length = len(dihedral_data_non_redund)
i = 0
while i < dihedral_data_non_redund_length:
    raw = dihedral_data_non_redund[i]           
    if raw[2]:
        i += 1
    else:
        at_1 = unq_to_NEW_types[ raw[3][0] ]
        at_2 = unq_to_NEW_types[ raw[3][1] ]
        at_3 = unq_to_NEW_types[ raw[3][2] ]   
        at_4 = unq_to_NEW_types[ raw[3][3] ]
        dih_line = 'AmbTrs' + sep + at_1 + sep + at_2 + sep +\
          at_3 + sep + at_4 + sep
        PO = [0, 0, 0, 0]
        Mag = [0.0, 0.0, 0.0, 0.0]
        di = 0
        for j in range(4):
            if (i+j) < dihedral_data_non_redund_length:
                raw_2 = dihedral_data_non_redund[i+j]
                if raw[3] == raw_2[3]:
                    di += 1
                    periodicity = round(raw_2[5]) # periodicity can be 1, 2, 3 or 4
                    # check if periodicity is one of: 1, 2, 3 or 4
                    if periodicity in (1, 2, 3, 4):
                        po = int(round((raw_2[6]))) # Gaussian expects PO as integer !!!
                        mag = round((raw_2[4]), 6) # round to 6 digits for better precision
                        PO[periodicity-1] = po
                        Mag[periodicity-1] = mag
                    elif periodicity > 4:
                        w_line = raw[3][0] + sep + raw[3][1] + sep +\
                        raw[3][2] + sep + raw[3][3]
                        print('Warning: dihedral potential for: ', w_line, ' has periodicity > 4')
                        print('which will be ignored \n')
        i = i + di
        for k in range(4):
            dih_line = dih_line + str(PO[k]) + sep
        for l in range(4):       
            dih_line = dih_line + fstr.format(Mag[l]) + sep
        dih_line = dih_line + '1.0 \n' # NPaths set to 1 (Gaussian wants it as float)
        proper_lines.append(dih_line) 
    
for line in proper_lines:
    g16file.write(line)
if VERBOSE:
    print("proper dihedral parameters written to file ", datetime.datetime.now(), "\n")

# improper parameters
# force constant and phase_offset unchanged
for raw in improper_data_non_redund:
    if raw[2]:    
        at_1 = unq_to_NEW_types[ raw[3][0] ]
        at_2 = unq_to_NEW_types[ raw[3][1] ]
        at_3 = unq_to_NEW_types[ raw[3][2] ]   
        at_4 = unq_to_NEW_types[ raw[3][3] ]
        improper_line = 'ImpTrs' + sep + at_1 + sep + at_2 + sep +\
            at_3 + sep + at_4 + sep +\
            fstr.format(round(raw[4], 3)) + sep +\
            str(round((raw[6]), 1)) + sep +\
            str(round(raw[5], 1)) + '\n'
        g16file.write(improper_line)
if VERBOSE:
    print("improper dihedral parameters written to file ", datetime.datetime.now(), "\n")

# general settings
# Non-bonded interaction master function (standard amber FF)
non_bon_line ='NonBon 3 1 0 0 0.0 0.0 0.5 0.0 0.0 -1.2 \n'
g16file.write(non_bon_line)    

# flags for units used to specify FF    
units_lines = 'StrUnit 0 \n' + 'BndUnit 0 \n' + 'TorUnit 0 \n' + 'OOPUnit 0 \n'
g16file.write(units_lines)

# write final empty line:
g16file.write(empty_l)

# close the output (gaussian input) file
g16file.close()
if VERBOSE:
    print("Happy landing at: ", datetime.datetime.now(), "\n")
