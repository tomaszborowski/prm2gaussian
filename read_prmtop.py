# -*- coding: utf-8 -*-
"""
This is a collection of functions to parse (Amber) prmtop 
and prmcrd (rst7) files.

requires fortranformat (to install: pip install fortranformat)
"""
import re
import fortranformat as ff

# atm_mass - a dictionary with element symbols as keys and molar mass [u] as values
# (up to 86 - Rn) (taken from: https://ptable.com/)
atm_mass = {\
'H':1.008,'He':4.0026,\
'Li':6.94,'Be':9.0122,'B':10.81,'C':12.011,'N':14.007,'O':15.999,'F':18.998,'Ne':20.180,\
'Na':22.990,'Mg':24.305,'Al':26.982,'Si':28.085,'P':30.974,'S':32.06,'Cl':35.45,'Ar':39.948,\
'K':39.098,'Ca':40.078,'Sc':44.956,'Ti':47.867,'V':50.942,'Cr':51.996,'Mn':54.938,'Fe':55.845,'Co':58.933,'Ni':58.693,'Cu':63.546,'Zn':65.38,'Ga':1.22,'Ge':69.723,'As':74.922,'Se':78.971,'Br':79.904,'Kr':83.798,\
'Rb':85.468,'Sr':87.62,'Y':88.906,'Zr':91.224,'Nb':92.906,'Mo':95.95,'Tc':98.,'Ru':101.07,'Rh':102.91,'Pd':106.42,'Ag':107.87,'Cd':112.41,'In':114.82,'Sn':118.71,'Sb':121.76,'Te':127.60,'I':126.90,'Xe':131.29,\
'Cs':132.91,'Ba':137.33,'Hf':178.49,'Ta':180.95,'W':183.84,'Re':186.21,'Os':190.23,'Ir':192.22,'Pt':195.08,'Au':196.97,'Hg':200.59,'Tl':204.38,'Pb':207.2,'Bi':208.98,'Po':209.,'At':210.,'Rn':222.,\
'La':138.91,'Ce':140.12,'Pr':140.91,'Nd':144.24,'Pm':145.,'Sm':150.,'Eu':151.96,'Gd':157.25,'Tb':158.93,'Dy':162.50,'Ho':164.93,'Er':167.26,'Tm':168.93,'Yb':173.05,'Lu':174.97} 

# atm_number - a dictionary with element symbols as keys and atomic number as values
# (up to 86 - Rn)  
atm_number = {\
'H':1,'He':1,\
'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,\
'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,\
'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,\
'Rb':37,'Sr':38,'Y':39,'Zr':40,'Nb':41,'Mo':42,'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,'Sb':51,'Te':52,'I':53,'Xe':54,\
'Cs':55,'Ba':56,'Hf':72,'Ta':73,'W':74,'Re':75,'Os':76,'Ir':77,'Pt':78,'Au':79,'Hg':80,'Tl':81,'Pb':82,'Bi':83,'Po':84,'At':85,'Rn':86,\
'La':57,'Ce':58,'Pr':59,'Nd':60,'Pm':61,'Sm':62,'Eu':63,'Gd':64,'Tb':65,'Dy':66,'Ho':67,'Er':68,'Tm':69,'Yb':70,'Lu':71} 

# at_num_symbol - dic mapping atomic number to element symbol (up to 86 - Rn)
at_num_symbol = \
{1:'H', 2:'He',\
3:'Li', 4:'Be', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 10:'Ne',\
11:'Na', 12:'Mg', 13:'Al', 14:'Si', 15:'P', 16:'S', 17:'Cl', 18:'Ar',\
19:'K', 20:'Ca', 21:'Sc', 22:'Ti', 23:'V', 24:'Cr', 25:'Mn', 26:'Fe', 27:'Co', 28:'Ni', 29:'Cu', 30:'Zn', 31:'Ga', 32:'Ge', 33:'As', 34:'Se', 35:'Br', 36:'Kr',\
37:'Rb', 38:'Sr', 39:'Y', 40:'Zr', 41:'Nb', 42:'Mo', 43:'Tc', 44:'Ru', 45:'Rh', 46:'Pd', 47:'Ag', 48:'Cd', 49:'In', 50:'Sn', 51:'Sb', 52:'Te', 53:'I', 54:'Xe',\
55:'Cs', 56:'Ba', 57:'La', 72:'Hf', 73:'Ta', 74:'W', 75:'Re', 76:'Os', 77:'Ir', 78:'Pt', 79:'Au', 80:'Hg', 81:'Tl', 82:'Pb', 83:'Bi', 84:'Po', 85:'At', 86:'Rn',
58:'Ce', 59:'Pr', 60:'Nd', 61:'Pm', 62:'Sm', 63:'Eu', 64:'Gd', 65:'Tb', 66:'Dy', 67:'Ho', 68:'Er', 69:'Tm', 70:'Yb', 71:'Lu'}    

LEGIT_TEXT_FLAGS = ['ATOM_NAME', 'RESIDUE_LABEL', 'AMBER_ATOM_TYPE',\
    'TREE_CHAIN_CLASSIFICATION', 'RADIUS_SET']
    
LEGIT_NUM_FLAGS = ['CHARGE', 'ATOMIC_NUMBER', 'MASS', 'ATOM_TYPE_INDEX',\
    'NUMBER_EXCLUDED_ATOMS', 'NONBONDED_PARM_INDEX', 'RESIDUE_POINTER',\
    'BOND_FORCE_CONSTANT', 'BOND_EQUIL_VALUE', 'ANGLE_FORCE_CONSTANT',\
    'ANGLE_EQUIL_VALUE', 'DIHEDRAL_FORCE_CONSTANT', 'DIHEDRAL_PERIODICITY',\
    'DIHEDRAL_PHASE', 'SCEE_SCALE_FACTOR', 'SCNB_SCALE_FACTOR', 'SOLTY',\
    'LENNARD_JONES_ACOEF', 'LENNARD_JONES_BCOEF', 'BONDS_INC_HYDROGEN',\
    'BONDS_WITHOUT_HYDROGEN', 'ANGLES_INC_HYDROGEN', 'ANGLES_WITHOUT_HYDROGEN',\
    'DIHEDRALS_INC_HYDROGEN', 'DIHEDRALS_WITHOUT_HYDROGEN', 'EXCLUDED_ATOMS_LIST',\
    'HBOND_ACOEF', 'HBOND_BCOEF', 'HBCUT', 'JOIN_ARRAY', 'IROTAT', 'RADII',\
    'SCREEN', 'IPOL', 'SOLVENT_POINTERS', 'ATOMS_PER_MOLECULE', 'BOX_DIMENSIONS',\
    'CAP_INFO', 'CAP_INFO2', 'POLARIZABILITY']

    
def prmtop_read_pointers(file):
    """
    Reads POINTERS section from the prmtop file

    Parameters
    ----------
    filename : prmtop file (file object)
        May or may not contain %COMMENT lines between: %FLAG POINTERS
        and %FORMAT

    Returns
    -------
    dictionary with keys: 
    NATOM, NTYPES, NBONH, MBONA, NTHETH, MTHETA, NPHIH, MPHIA, NHPARM, NPARM, 
    NNB, NRES, NBONA, NTHETA, NPHIA, NUMBND, NUMANG, NPTRA, NATYP, NPHB, 
    IFPERT, NBPER, NGPER, NDPER, MBPER, MGPER, MDPER, IFBOX, NMXRS, IFCAP, 
    NUMEXTRA, (if present: (NCOPY)) 
    and their integer values 

    """
    pointers_keys = ['NATOM', 'NTYPES', 'NBONH', 'MBONA', 'NTHETH', 'MTHETA',\
                     'NPHIH', 'MPHIA', 'NHPARM', 'NPARM',\
                     'NNB', 'NRES', 'NBONA', 'NTHETA', 'NPHIA', 'NUMBND',\
                     'NUMANG', 'NPTRA', 'NATYP', 'NPHB',\
                     'IFPERT', 'NBPER', 'NGPER', 'NDPER', 'MBPER', 'MGPER',\
                     'MDPER', 'IFBOX', 'NMXRS', 'IFCAP',\
                     'NUMEXTRA', 'NCOPY']
    pointers_values = []
    pointers = {}
    # w pliku file wyszukaj linii zawierajacej "%FLAG POINTERS"
    file.seek(0)
    flag_line = "%FLAG POINTERS"
    while True:
        a = file.readline()
        if not a:
            break
        match_flag=re.search(flag_line,a)
        if match_flag:            
            # wczytuj kolejne linie i sprawdzaj czy zawieraja "%FORMAT", 
            while True:
                a = file.readline()
                if not a:
                    break
                match_format=re.search("%FORMAT",a)
                if match_format:                            
                    # wczytaj format
                    format = a[8:].replace(')', '').replace(' ', '')
                    break
            # zgodnie z tym formatem wczytaj nastepne linie (fortranformat)
            fortran_line = ff.FortranRecordReader(format)
            while True:
                a = file.readline()
                if not a or a[0] == "%":
                    break
                temp = fortran_line.read(a)
                for item in temp:
                    pointers_values.append(item)                                    
    # utwórz i zwróc slownik
    for key, value in zip(pointers_keys, pointers_values):
        pointers[key] = value
    return pointers
        


def prmtop_read_text_section(file, FLAG, exp_length):
    """
    Reads sections with string values (ATOM_NAME, RESIDUE_LABEL, AMBER_ATOM_TYPE,
    TREE_CHAIN_CLASSIFICATION, RADIUS_SET) from the prmtop file
    Returns a list with string values
    ----
    file : file object
    FLAG: string, one from: 'ATOM_NAME', 'RESIDUE_LABEL', 'AMBER_ATOM_TYPE', 
    'TREE_CHAIN_CLASSIFICATION', 'RADIUS_SET'
    exp_length: integer, expected number of fields to be read (know from pointers)
    """
    assert (FLAG in LEGIT_TEXT_FLAGS), "not a valid FLAG"    
    assert (type(exp_length) == int), "exp_length must be an integer value"
    # inicjalizacja pustej listy, ktora bedzie zwracana     
    text_section = []
    # w pliku file wyszukaj linii zawierajacej "%FLAG " + FLAG
    flag_line = "%FLAG " + FLAG
    file.seek(0)
    while True:
        a = file.readline()
        if not a:
            break
        match_flag=re.search(flag_line,a)
        if match_flag:            
            # wczytuj kolejne linie i sprawdzaj czy zawieraja "%FORMAT", 
            while True:
                a = file.readline()
                if not a:
                    break
                match_format=re.search("%FORMAT",a)
                if match_format:                            
                    # wczytaj format
                    format = a[8:].replace(')', '').replace(' ', '')
                    break
            # zgodnie z tym formatem wczytaj nastepne linie (fortranformat)
            fortran_line = ff.FortranRecordReader(format)
            while True:
                a = file.readline()
                if not a or a[0] == "%":
                    break
                temp = fortran_line.read(a)
                for item in temp:
                    text_section.append(item)                                    
    # usun z listy pola zajmowane przez 4 spacje
    text_section = [item for item in text_section if item != '    ']
    # usun extra spacje z wartosci tekstowych:
    text_section = [item.replace(' ', '') for item in text_section]
    # sprawdz, czy lista ma spodziewana dlugosc i ja zwróc
    assert (len(text_section) == exp_length), "length of a text section seems \
not to match the expected length inferred from pointers"
    return text_section    



def prmtop_read_numeric_section(file, FLAG, exp_length):
    """
    file : file object
    Reads sections with numeric values from the prmtop file
    Returns a list with numeric values
    FLAG: string, one from: 'CHARGE', 'ATOMIC_NUMBER', 'MASS', 'ATOM_TYPE_INDEX',
    'NUMBER_EXCLUDED_ATOMS', 'NONBONDED_PARM_INDEX', 'RESIDUE_POINTER',
    'BOND_FORCE_CONSTANT', 'BOND_EQUIL_VALUE', 'ANGLE_FORCE_CONSTANT',
    'ANGLE_EQUIL_VALUE', 'DIHEDRAL_FORCE_CONSTANT', 'DIHEDRAL_PERIODICITY',
    'DIHEDRAL_PHASE', 'SCEE_SCALE_FACTOR', 'SCNB_SCALE_FACTOR', 'SOLTY',
    'LENNARD_JONES_ACOEF', 'LENNARD_JONES_BCOEF', 'BONDS_INC_HYDROGEN',
    'BONDS_WITHOUT_HYDROGEN', 'ANGLES_INC_HYDROGEN', 'ANGLES_WITHOUT_HYDROGEN',
    'DIHEDRALS_INC_HYDROGEN', 'DIHEDRALS_WITHOUT_HYDROGEN', 'EXCLUDED_ATOMS_LIST',
    'HBOND_ACOEF', 'HBOND_BCOEF', 'HBCUT', 'JOIN_ARRAY', 'IROTAT', 'RADII',
    'SCREEN', 'IPOL', 'SOLVENT_POINTERS', 'ATOMS_PER_MOLECULE', 'BOX_DIMENSIONS',
    'CAP_INFO', 'CAP_INFO2', 'POLARIZABILITY'
    exp_length: integer, expected number of fields to be read (know from pointers)
    """
    # sprawdzam, czy dostalem wlasciwe argumenty
    assert (FLAG in LEGIT_NUM_FLAGS), "not a valid FLAG"        
    assert (type(exp_length) == int), "exp_length must be an integer value"
    # inicjalizacja pustej listy, ktora bedzie zwracana     
    numeric_section = []
    # w pliku file wyszukaj linii zawierajacej "%FLAG " + FLAG
    flag_line = "%FLAG " + FLAG
    file.seek(0)
    while True:
        a = file.readline()
        if not a:
            break
        match_flag=re.search(flag_line,a)
        if match_flag:            
            # wczytuj kolejne linie i sprawdzaj czy zawieraja "%FORMAT", 
            while True:
                a = file.readline()
                if not a:
                    break
                match_format=re.search("%FORMAT",a)
                if match_format:                            
                    # wczytaj format
                    format = a[8:].replace(')', '').replace(' ', '')
                    break
            # zgodnie z tym formatem wczytaj nastepne linie (fortranformat)
            fortran_line = ff.FortranRecordReader(format)
            while True:
                a = file.readline()
                if not a or a[0] == "%":
                    break
                temp = fortran_line.read(a)
                for item in temp:
                    numeric_section.append(item)                                    
    # usun z listy pola zajmowane przez None
    numeric_section = [item for item in numeric_section if item != None]
    # sprawdz, czy lista ma spodziewana dlugosc i ja zwróc
    assert (len(numeric_section) == exp_length), "length of a numeric section seems \
not to match the expected length inferred from pointers"
    return numeric_section    



def crd_read_coordinates(file):
    """
    Reads atomic coordinates from amber prmcrd (rst7) file
    Parameters
    ----------
    file : prmcrd or rst7 amber file (file object)

    Returns
    -------
    (NATOM, coordinates), where:
    NATOM : int, number of atoms read from the file    
    coordinates : nested list, a list of NATOM 3-element lists, 
    each containing x,y,z for a given atom
    e.g [[x1, y1, z1], [x2, y2, z2], .... [xn, yn, zn]]
    """
    coordinates = []
    # read the first line from the file and discard it:
    file.seek(0)
    file.readline()
    # read NATOM section from the second line
    a = file.readline()
    fortran_line = ff.FortranRecordReader('I6')
    NATOM = fortran_line.read(a)[0]
    assert (type(NATOM) == int), "Problems when reading NATOM from coordinate file"
    # read the coordinate section of the file and write all numbers to temp_list
    fortran_line = ff.FortranRecordReader('6F12.7')
    counter = 0
    temp_list = []
    while counter < NATOM:
        a = file.readline()
        counter += 2
        if not a:
            break
        temp = fortran_line.read(a)
        for item in temp:
            temp_list.append(item)
    # re-write coordinates from temp_list to the final coordinates nested list
    n = 1
    while n <= NATOM:
        coordinates.append(temp_list[3*(n-1):3*n])
        n += 1
    # sprawdz, czy lista ma spodziewana dlugosc i ja zwróc wraz z NATOM
    assert (len(coordinates) == NATOM), "mismatch between NATOM and \
number or coordinates read"
    return NATOM, coordinates

    