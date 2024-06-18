








"""Contains all user-defined event types.

All user-defined events are defined here, which
must be derived from the parent class EventBase.  

See also
---------
Module: base.events for documentation about the methods possible(), get_rate(), and do_event().

"""

import numpy as np
from base.events import EventBase
#from .user_entropy import get_Zvib, get_Z_CO, get_Z_O2

#from .user_constants import mCO, mO2, Asite, modes_COads, \
#    modes_Oads, modes_TS_COOx, modes_COgas, modes_O2gas, kB, eV2J, s0CO, s0O, h

#from .user_energy import EadsCO, EadsO, get_Ea, \
#    get_repulsion, EdiffCO, EdiffO

import sys
from ase import Atoms
import ase.units as au
from ase.build import molecule
import ase.thermochemistry as tchem

import mkm

import numpy as np
from numpy import pi, exp, log
from scipy.integrate import ode
import matplotlib.pyplot as m

h_planck = au._hplanck*au.J*au.s

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def mu_H2_ideal(p,T):
    """All contributions to the chemical potential of 
    molecular hydrogen at given pressure p and T except
    of ZPE and bonding energy. The pV term cancels with a 
    contribution in TS_trans and is therefore not appearing.
    returns chem. potential in eV.
    
    p: float
    pressure in eV/Ang**3
    
    T: float
    temperature in K"""
    h_planck = h = au._hplanck*au.J*au.s
    H2 = molecule('H2')
    # translational dof:
    m_mol = H2.get_masses().sum()
    TS_trans = au.kB*T*log(au.kB*T/p/h**3 * (2*pi*m_mol*au.kB*T)**1.5)
    # rotational dof:
    inertia = max(H2.get_moments_of_inertia())
    sigma = 2
    q_rot = (8 * pi**2 * inertia * au.kB * T / sigma / h**2)
    TS_rot = au.kB * T * log(q_rot)
    # vibrational dof:
    cm_m1 = 1./8065.540106923572 # in eV
    v = 4401*cm_m1 # vibrational energy of the H2 phonon
    q_vib = 1/(1-exp(-h*v/(au.kB*T)))
    TS_vib = au.kB * T * log(q_vib)
    print('Vibrational energy of H2', TS_vib, 'ZPE', -au.kB * T * log(exp(-v/(au.kB*T*2)) ) )
    print('rotational', TS_rot)
    return -TS_trans - TS_rot - TS_vib


def mu_H2O_ideal(p,T):
    """All contributions to the chemical potential of 
    molecular hydrogen at given pressure p and T except
    of ZPE and bonding energy. The pV term cancels with a 
    contribution in TS_trans and is therefore not appearing.
    returns chem. potential in eV.
    
    p: float
    pressure in eV/Ang**3
    
    T: float
    temperature in K"""
    h_planck = h = au._hplanck*au.J*au.s
    H2O = molecule('H2O')
    # translational dof:
    m_mol = H2O.get_masses().sum()
    TS_trans = au.kB*T*log(au.kB*T/p/h**3 * (2*pi*m_mol*au.kB*T)**1.5)
    # rotational dof:
    inertia = max(H2O.get_moments_of_inertia())
    sigma = 2
    inertias = H2O.get_moments_of_inertia()
    q_rot = np.sqrt(pi * np.product(inertias)) / sigma
    q_rot *= (8.0 * pi**2 * au.kB * T / h_planck**2)**(3.0 / 2.0)
    TS_rot = au.kB * T * log(q_rot)
    # vibrational dof:
    print(h)
    cm_m1 = 1/au.kB*1./8065.540106923572 # in eV   
    v1 = 1595 * cm_m1 # vibrational energy of the H2 phonon
    v2 = 3657 * cm_m1
    v3 = 3756 * cm_m1
    q_vib = 1/(1-exp(-v1/T))  #exp(-v1/T/2)/(1-exp(-v1/T)) this seems zpe corrected
    q_vib *= 1/(1-exp(-v2/T))# exp(-v2/T/2)/(1-exp(-v2/T))
    q_vib *= 1/(1-exp(-v3/T)) #exp(-v3/T/2)/(1-exp(-v3/T))
    TS_vib = au.kB * T * log(q_vib)
    print('Vibrational energy of H2O', TS_vib, 'ZPE', - au.kB * T *log(exp(-v1/T/2)*exp(-v2/T/2)*exp(-v3/T/2)) ) 
    
    print('rotational', TS_rot)
    
    return -TS_trans - TS_rot - TS_vib # This is proportional to the species' free energy

def mu_H_ideal(p,T):
    """All contributions to the chemical potential of 
    molecular hydrogen at given pressure p and T except
    of ZPE and bonding energy. The pV term cancels with a 
    contribution in TS_trans and is therefore not appearing.
    returns chem. potential in eV.
    
    p: float
    pressure in eV/Ang**3
    
    T: float
    temperature in K"""
    h_planck = h = au._hplanck*au.J*au.s
    H = Atoms('H')
    # translational dof:
    m_mol = H.get_masses().sum()
    TS_trans = au.kB*T*log(au.kB*T/p/h**3 * (2*pi*m_mol*au.kB*T)**1.5)
    print('Vibrational energy of H 0')
    return -TS_trans



def calc_kads_lower_limit(G, E_act, T):
    """
    Lower estimate for reaction rate constant for adsorption in 1/s. Full cancellation of 
    ZPE between initial and transition state is assumed. Returns rate in units of 
    1/s.
    
    T           - Temperature in K
    G           - Gibbs free energy (without potential energy and ZPE) of gas phase
    E_act       - bare DFT activation energy without ZPE and entropic contributions
    """
    assert E_act >= 0
    h_planck = h = au._hplanck*au.J*au.s
    k_low = au.kB*T/h_planck*exp(-(E_act -np.array(G))/au.kB/T)
    print('activation energy', E_act, '$\mu$', np.array(G))   
    return k_low * au.s

def calc_kads_upper_limit(E_act, T, A, p, m_mol, ZPE=0.0):
    """
    Upper estimate for reaction rate constant for adsorption in 1/s. Full cancellation of 
    ZPE between initial and transition state is assumed if ZPE=0.0. Else
    a higher estimate is obtained if ZPE is only included for the gas phase state
    (ZPE=E_vib/2). Returns rate in units of 1/s.
    
    E_act       - bare DFT activation energy without ZPE and entropic contributions
    T           - Temperature in K
    A           - Surface area per molecule in Ang**2
    p           - Pressure in eV/Ang**3
    m_mol       - adsorbent mass in atomic units
    """
    E_act = E_act - ZPE
    assert E_act >= 0
    k_up = A * p / np.sqrt(2*pi*m_mol*au.kB*T) * exp(-E_act/au.kB/T)
    return k_up * au.s

# The adsorption rates (Hertz-Knudsen)
def calc_kads_Hertz_Knudsen(E_act, T, A, p, m_mol):
    """
    Reaction rate constant for adsorption in 1/s
    
    T           - Temperature in K
    P           - Pressure in eV/Ang**3
    A           - Surface area in Ang**2
    m           - Mass of reactant in kg
    """
    return calc_kads_upper_limit(E_act, T, A, p, m_mol, ZPE=0.0)

def calc_kads_omega_estimate(E_act, T, p, m_mol,omega):
    """
    Estimation of adsorption rates in 1/s using the omega model from 
    J. Phys. Chem. C 2008, 112, 4608-4617. omega=0 corresponds to a TS that is 
    initial state like (gas phase), omega=1 corresponds to a TS that is final 
    state like (adsorbed state).
    
    E_act       - bare DFT activation energy without ZPE and entropic contributions
    T           - Temperature in K
    p           - Pressure in eV/Ang**3
    m_mol       - adsorbent mass in atomic units
    """
    h_planck = h = au._hplanck*au.J*au.s
    k_ads = au.kB*T/h_planck * (p*h_planck**3/(au.kB*T*(2*pi*m_mol*au.kB*T)**1.5 ))**omega * exp(-(E_act)/au.kB/T)
    return k_ads * au.s

def calc_k_Eyring(T,Eact):
    """
    Calculate reaction rate constant according to Eyring equation in 1/s
    
    T       - Temperature in K
    Eact    - Activation energy in eV
    """
    h = au._hplanck*au.J*au.s
    return au.kB * T / h * np.exp(-Eact / (au.kB * T)) * au.s

def calc_kdes_lower_limit(T, Eact):
    """
    Reaction rate constant for desorption with activation barrier in 1/s
    
    T           - Temperature in K
    Eact        - Activation energy for desorption
    """
    return calc_k_Eyring(T, Eact)

def calc_kdes_upper_limit(atoms, T, A, m, sigma, Eact):
    """
    Reaction rate constant for desorption with activation barrier in 1/s
    
    atoms       - ASE atoms object of the gas molecule
    T           - Temperature in K
    A           - Surface area in Ang**2
    m           - Mass of reactant in atomic units
    sigma       - Symmetry number
    theta_rot   - Rotational temperature in K
    Eact        - Activation energy for desorption
    """
    #assert len(atoms) <= 2
    inertia = max(atoms.get_moments_of_inertia())
    m_mol = atoms.get_masses().sum()
    h = au._hplanck * au.J * au.s
    if len(atoms) == 2:
        q_rot = (8 * pi**2 * inertia * au.kB * T / sigma / h**2)
    elif len(atoms) == 3:
        inertias = atoms.get_moments_of_inertia()
        q_rot = np.sqrt(pi * np.product(inertias)) / sigma
        q_rot *= (8.0 * pi**2 * au.kB * T / h_planck**2)**(3.0 / 2.0)
    elif len(atoms) == 1:
        q_rot = 1
    q_trans_2D = A * 2 * pi * m_mol *au.kB * T / h**2
    return au.kB * T / h  * q_trans_2D * q_rot * np.exp(-Eact / (au.kB * T)) * au.s

def calc_kdes_barrierless_diatomic(atoms, T, A, m_mol, sigma, Edes):
    """
    Reaction rate constant for desorption in 1/s
    
    atoms       - ASE atoms object of the gas molecule
    T           - Temperature in K
    A           - Surface area in Ang**2
    m           - Mass of reactant in atomic units
    sigma       - Symmetry number
    Edes        - Desorption energy in eV
    """
    assert len(atoms) <= 2
    inertia = max(atoms.get_moments_of_inertia())
    h = au._hplanck * au.J * au.s
    if len(atoms) == 2:
        q_rot = (8 * pi**2 * inertia * au.kB * T / sigma / h**2)
    elif len(atoms) == 1:
        q_rot = 1
    q_trans_2D = A * 2 * pi * m_mol *au.kB * T / h**2
    return au.kB * T / h  * q_trans_2D * q_rot * np.exp(-Edes / (au.kB * T)) * au.s

def calc_kdes_barrierless(atoms, T, A, sigma, Edes, nonlinear=False):
    """
    Reaction rate constant for desorption in 1/s
    
    atoms       - ASE atoms object of the gas molecule
    T           - Temperature in K
    A           - Surface area in Ang**2
    m           - Mass of reactant in atomic units
    sigma       - Symmetry number
    Edes        - Desorption energy in eV
    """
    h_planck = au._hplanck * au.J * au.s
    m_mol = atoms.get_masses().sum()
    if len(atoms) > 1:
        if (not nonlinear):
            inertia = max(atoms.get_moments_of_inertia())
            q_rot = (8 * pi**2 * inertia * au.kB * T / sigma / h_planck**2)
        elif nonlinear:
            inertias = atoms.get_moments_of_inertia()
            q_rot = np.sqrt(pi * np.product(inertias)) / sigma
            q_rot *= (8.0 * pi**2 * au.kB * T / h_planck**2)**(3.0 / 2.0)
            # print('q_rot: ', q_rot)
    elif len(atoms) == 1:
            q_rot =1
    q_trans_2D = A * 2 * pi * m_mol *au.kB * T / h_planck**2 
    # print('q_trans_2D: ', q_trans_2D)
    return au.kB * T / h_planck  * q_trans_2D * q_rot * np.exp(-Edes / (au.kB * T)) * au.s

def calc_ads_des_ratio(atoms, T, A, p, sigma, E_rct, nonlinear=False):
    """ The ratio between adsorption and desorption rate
    to be consistent with the detailed balance condition.
    atoms       - ASE atoms object
    T           - Temperature in K
    sigma       - Symmetry number
    Erct        - E_ads - E_des in eV"""
    
    m_mol = atoms.get_masses().sum()
    h_planck = au._hplanck * au.J * au.s

    # translational dof:
    q_trans = 1/exp(1) + au.kB*T/p/h_planck**3 * (2*pi*m_mol*au.kB*T)**1.5 
    # print('q_trans: ', q_trans)
    # rotational dof:
    if len(atoms) > 1:
        if (not nonlinear):
            inertia = max(atoms.get_moments_of_inertia())
            q_rot = (8 * pi**2 * inertia * au.kB * T / sigma / h_planck**2)
        elif nonlinear:
            inertias = atoms.get_moments_of_inertia()
            q_rot = np.sqrt(pi * np.product(inertias)) / sigma
            q_rot *= (8.0 * pi**2 * au.kB * T / h_planck**2)**(3.0 / 2.0)
            print('q_rot: ', q_rot)
    elif len(atoms) == 1:
            q_rot =1
    # print('q_rot: ', q_rot)
    return q_trans * q_rot * exp(E_rct/au.kB/T)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
########CONSTANTS#####################################################
######################################################################
######################################################################














A = 6.684 * 3.005
p_H2 = 1e-5 * au.bar

p_H2O = 1e-8 * p_H2 # #20 bar, 200: 1 bar, 400: 1 Pa 265   #### changed 0405 aenm SI


p_H = 0.7/100.0 * p_H2 # 
p_Pa = p_H2/au.bar*1e5

T = 518.15

A1 = 1.5

A2 = 11.8








#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Reaction 2: 1 HTi5C + 1 HO2C <-> 1 H2 + 1 *Ti5C + 1 *O2C
mu = mu_H2_ideal(p_H2,T)
mol = molecule('H2')
m_mol = 2 * mol.get_masses()[0]
sigma = 2
E_rct_2 = 0.19+0.22 # 
E_act_des_2 = 0.41  # E_act_2 - E_rct_2
E_act_2 = E_act_des_2 + E_rct_2

k_ads_up_2 = calc_kads_upper_limit(E_act_2, T, A, p_H2, m_mol, ZPE=0.0)
k_ads_low_2 = calc_kads_lower_limit(mu, E_act_2, T)
ratio_des_ads_2 = calc_ads_des_ratio(mol, T, A, p_H2, sigma, E_rct_2)
k_des_up_2 = k_ads_up_2 * ratio_des_ads_2
k_des_low_2 = k_ads_low_2 * ratio_des_ads_2
#k_des_up_2 = calc_kdes_upper_limit(mol, T, A, m_mol, sigma, E_act_des_2)
#k_des_low_2 = calc_kdes_lower_limit(T, E_act_des_2)
print('adsorption rate limits reaction 2: ', k_ads_up_2, k_ads_low_2)
print('desorption rate limits reaction 2: ', k_des_up_2, k_des_low_2)

#T = 300 + 273 ### Temperature of atomic Hydrogen

# Reaction 3: 1 H + 1 *O3C <-> 1 HO3C


mol = Atoms('H')
m_mol = mol.get_masses().sum()
sigma = 1 
mu = mu_H_ideal(p_H*A/A1, T)
E_act_3 = 0.08 #+0.1
E_rct_3 = -1.91  + 0.23 # + 3749*0.5/8065.540106923572 # E_final - E_initial zpe: 3749*0.5/8065.540106923572 ev(hydroxyl) ~ 0.23 eV- 
#k_fwd_3 = calc_k_Eyring(T, E_act_3)
#k_bwd_3 = calc_k_Eyring(T, E_act_3-E_rct_3)
#A = 1.5
k_ads_up_3 = calc_kads_upper_limit(E_act_3, T, A1, p_H, m_mol, ZPE=0.0)
k_ads_low_3 = calc_kads_lower_limit(mu, E_act_3 , T)
ratio_des_ads_3 = calc_ads_des_ratio(mol, T, A1, p_H*A/A1, sigma, E_rct_3)
k_des_up_3 = k_ads_up_3 * ratio_des_ads_3
k_des_low_3 = k_ads_low_3 * ratio_des_ads_3
#k_des_up_2 = calc_kdes_upper_limit(mol, T, A, m_mol, sigma, E_act_des_2)
#k_des_low_2 = calc_kdes_lower_limit(T, E_act_des_2)
print('adsorption rate limits reaction 3: ', k_ads_up_3, k_ads_low_3)
print('desorption rate limits reaction 3: ', k_des_up_3, k_des_low_3)

print("Eyring fwd bwd",calc_k_Eyring(T, E_act_3), calc_k_Eyring(T, E_act_3-E_rct_3) )


# Reaction 0: 1 H + 1 *O2C <-> 1 HO2C

mol = Atoms('H')
m_mol = mol.get_masses().sum()
sigma = 1 
mu = mu_H_ideal(p_H*A/A2, T)
E_act_0 = 0.08 #+0.1
E_rct_0 = -2.93 + 0.23 # + 3749*0.5/8065.540106923572 #-1.91 # E_ads - E_des # -2.93 Calculations
#k_fwd_3 = calc_k_Eyring(T, E_act_3)
#k_bwd_3 = calc_k_Eyring(T, E_act_3-E_rct_3)

k_ads_up_0 = calc_kads_upper_limit(E_act_0, T, A2, p_H, m_mol, ZPE=0.0)
k_ads_low_0 = calc_kads_lower_limit(mu, E_act_0, T)
ratio_des_ads_0 = calc_ads_des_ratio(mol, T, A2, p_H*A/A2, sigma, E_rct_0)
k_des_up_0 = k_ads_up_0 * ratio_des_ads_0
k_des_low_0 = k_ads_low_0 * ratio_des_ads_0

print('adsorption rate limits reaction 0: ', k_ads_up_0, k_ads_low_0)
print('desorption rate limits reaction 0: ', k_des_up_0, k_des_low_0)


# Reaction 4: 1 H + 1 *HO2C <-> 1 H2O2C
E_act_4 = 0.12
E_rct_4 = 0.51  - 1.91 + 0.23 +  0.099 # + 3749*0.5/8065.540106923572 # exact value from running simulations
#k_fwd_4 = calc_k_Eyring(T, E_act_4)
#k_bwd_4 = calc_k_Eyring(T, E_act_4-E_rct_4)

k_ads_up_4 = calc_kads_upper_limit(E_act_4, T, A2, p_H, m_mol, ZPE=0.0)
k_ads_low_4 = calc_kads_lower_limit(mu, E_act_4, T)
ratio_des_ads_4 = calc_ads_des_ratio(mol, T, A2, p_H*A/A2, sigma, E_rct_4)
k_des_up_4 = k_ads_up_4 * ratio_des_ads_4
k_des_low_4 = k_ads_low_4 * ratio_des_ads_4
#k_des_up_2 = calc_kdes_upper_limit(mol, T, A, m_mol, sigma, E_act_des_2)
#k_des_low_2 = calc_kdes_lower_limit(T, E_act_des_2)
print('adsorption rate limits reaction 4: ', k_ads_up_4, k_ads_low_4)
print('desorption rate limits reaction 4: ', k_des_up_4, k_des_low_4)
print('k_deS_ads_ratio', ratio_des_ads_4, 'mu', mu)


# reaction 5: 1 H + 1 *Ti5C <-> 1 HTi5C
E_act_5 = 0.00 
E_rct_5 = 0.19-1.91 # E_final - E_initial
#k_fwd_5 = calc_k_Eyring(T, E_act_5 )
#k_bwd_5 = calc_k_Eyring(T, E_act_5 - E_rct_5)

k_ads_up_5 = calc_kads_upper_limit(E_act_5, T, A, p_H, m_mol, ZPE=0.0)
k_ads_low_5 = calc_kads_lower_limit(mu, E_act_5, T)
ratio_des_ads_5 = calc_ads_des_ratio(mol, T, A, p_H, sigma, E_rct_5)
k_des_up_5 = k_ads_up_5 * ratio_des_ads_5
k_des_low_5 = k_ads_low_5 * ratio_des_ads_5
#k_des_up_2 = calc_kdes_upper_limit(mol, T, A, m_mol, sigma, E_act_des_2)
#k_des_low_2 = calc_kdes_lower_limit(T, E_act_des_2)
print('adsorption rate limits reaction 5: ', k_ads_up_5, k_ads_low_5)
print('desorption rate limits reaction 5: ', k_des_up_5, k_des_low_5)
#T = T_ini

# Reaction 1: H2O2C <-> 1 H2O + 1 *2C
mu = mu_H2O_ideal(p_H2O*A/A2,T)
H2O = molecule('H2O')
m_H2O = H2O.get_masses().sum()
sigma_H2O = 2
E_rct_1 = +0.51-1.42 #-1.04 # E_ads - E_des # Efinal-Einitial
E_act_des_1 = 0.91 # https://pubs.rsc.org/en/content/articlepdf/2017/cy/c6cy02007k
E_act_1 = E_act_des_1 + E_rct_1

k_ads_up_1 = calc_kads_upper_limit(E_act_1, T, A1, p_H2O, m_H2O, ZPE=0.0)
k_ads_low_1 = calc_kads_lower_limit(mu, E_act_1, T)
ratio_des_ads_1 = calc_ads_des_ratio(H2O, T, A, p_H2O*A/A2, sigma_H2O, E_rct_1, nonlinear = True)
k_des_up_1 = k_ads_up_1 * ratio_des_ads_1
k_des_low_1 = k_ads_low_1 * ratio_des_ads_1
#k_des_up_1 = calc_kdes_upper_limit(mol, T, A, m_mol, sigma, E_act_des_1)
#k_des_low_1 = calc_kdes_lower_limit(T, E_act_des_1)
print('adsorption rate limits reaction 1: ', k_ads_up_1, k_ads_low_1)
print('desorption rate limits reaction 1: ', k_des_up_1, k_des_low_1)
print('activation energy both sides: ', E_act_1, E_act_des_1)


print('done!')

k_ads_mid_1 = (k_ads_low_1*k_ads_up_1)**0.5
k_ads_mid_2 = (k_ads_low_2*k_ads_up_2)**0.5
k_ads_mid_3 = (k_ads_low_3*k_ads_up_3)**0.5
k_ads_mid_4 = (k_ads_low_4*k_ads_up_4)**0.5
k_ads_mid_5 = (k_ads_low_5*k_ads_up_5)**0.5
k_ads_mid_0 = (k_ads_low_0*k_ads_up_0)**0.5



k_des_mid_1 = (k_des_low_1*k_des_up_1)**0.5
k_des_mid_2 = (k_des_low_2*k_des_up_2)**0.5
k_des_mid_3 = (k_des_low_3*k_des_up_3)**0.5
k_des_mid_4 = (k_des_low_4*k_des_up_4)**0.5
k_des_mid_5 = (k_des_low_5*k_des_up_5)**0.5
k_des_mid_0 = (k_des_low_0*k_des_up_0)**0.5

print(k_ads_mid_0, k_des_mid_0, '0')
print(k_ads_mid_1, k_des_mid_1, '1')
print(k_ads_mid_2, k_des_mid_2, '2')
print(k_ads_mid_3, k_des_mid_3, '3')
print(k_ads_mid_4, k_des_mid_4, '4')
print(k_ads_mid_5, k_des_mid_5, '5')


# reaction_7: 1 HO3C + 1 *Ti5C <-> 1 HTi5c + 1 O3C
E_act_7 = 1.72
E_rct_7 = 0.19 # E_final - E_initial
k_fwd_7 = calc_k_Eyring(T, E_act_7 )
k_bwd_7 = calc_k_Eyring(T, E_act_7 - E_rct_7 )


'''
###stype defined:
vo2c == -1
o2c == 0
o3c == 1
ti5c == 2
ti6c == 3
'''



##### O2CAds, VO2CAds, H2Ads, O3CAds, HO2CAds, Ti5CAds














class O2CAdsEvent(EventBase):
    """event_0
    The event is H + *o2c -> H*o2c.
    
    """

    def __init__(self, params):
        EventBase.__init__(self, params)

    def possible(self, system, site, other_site):
        if system.sites[site].covered == 0 and system.sites[site].stype == 0:
            return True
        else:
            return False

    def get_rate(self, system, site, other_site):
        R = k_ads_mid_0
        return self.alpha * R  # alpha important for temporal acceleration.

    def do_event(self, system, site, other_site):
        if system.sites[site].covered == 0 and system.sites[site].stype == 0:
            system.sites[site].covered = 1
        

    def get_involve_other(self):
        return False 

class VO2CAdsEvent(EventBase):
    """Event 1
    The event is H2O + *2c -> H2o*2c.
    The event is possible if the site is empty.  
    The rate comes from collision theory.  
    Performing the event adds a CO to the site.

    """

    def __init__(self, params):
        EventBase.__init__(self, params)

    def possible(self, system, site, other_site):
        if system.sites[site].covered == 0 and system.sites[site].stype == -1:
            return True
        else:
            return False

    def get_rate(self, system, site, other_site):
        R = k_ads_mid_1
        return self.alpha * R  # alpha important for temporal acceleration.

    def do_event(self, system, site, other_site):
        if system.sites[site].covered == 0 and system.sites[site].stype == -1:
        	system.sites[site].covered = 2
        	system.sites[site].stype = 0

    def get_involve_other(self):
        return False 

class H2AdsEvent(EventBase):
    """Event 2
    The event is 1 HTi5C + 1 HO2C <-> 1 H2 + 1 *Ti5C + 1 *O2C
    The event is possible if the site is empty.  
    The rate comes from collision theory.  
    Performing the event adds a CO to the site.

    """

    def __init__(self, params):
        EventBase.__init__(self, params)

    def possible(self, system, site, other_site):
        if (system.sites[site].covered == 0 and system.sites[site].stype == 0 and system.sites[other_site].covered == 0 and system.sites[other_site].stype == 2) or (system.sites[site].covered == 0 and system.sites[site].stype == 2 and system.sites[other_site].covered == 0 and system.sites[other_site].stype == 0):
            return True
        else:
            return False

    def get_rate(self, system, site, other_site):
        R = k_ads_mid_2
        return self.alpha * R  # alpha important for temporal acceleration.

    def do_event(self, system, site, other_site):

        system.sites[site].covered = 1
        system.sites[other_site].covered = 1

    def get_involve_other(self):
        return True 


class O3CAdsEvent(EventBase):
    """event 3
    The event is H + *o3c -> H*o3c.
    
    """

    def __init__(self, params):
        EventBase.__init__(self, params)

    def possible(self, system, site, other_site):
        if system.sites[site].covered == 0 and system.sites[site].stype == 1:
            return True
        else:
            return False

    def get_rate(self, system, site, other_site):
        R = k_ads_mid_3
        return self.alpha * R  # alpha important for temporal acceleration.

    def do_event(self, system, site, other_site):
        if system.sites[site].stype == 1 and system.sites[site].covered == 0: 
        	system.sites[site].covered = 1
            

    def get_involve_other(self):
        return False 


class HO2CAdsEvent(EventBase):
    """Event 4
    The event is H + *ho2c -> H2*o2c.
    
    """

    def __init__(self, params):
        EventBase.__init__(self, params)

    def possible(self, system, site, other_site):
        if system.sites[site].covered == 1 and system.sites[site].stype == 0:
            return True
        else:
            return False

    def get_rate(self, system, site, other_site):
        R = k_ads_mid_4
        return self.alpha * R  # alpha important for temporal acceleration.

    def do_event(self, system, site, other_site):
        if system.sites[site].covered == 1 and system.sites[site].stype == 0:
        	system.sites[site].covered = 2

    def get_involve_other(self):
        return False 

class Ti5CAdsEvent(EventBase):
    """Event 5
    The event is H + *Ti5c -> H*Ti5c.
    The event is possible if the site is empty.  
    The rate comes from collision theory.  
    Performing the event adds a CO to the site.

    """

    def __init__(self, params):
        EventBase.__init__(self, params)

    def possible(self, system, site, other_site):
        if system.sites[site].covered == 0 and system.sites[site].stype == 2:
            return True
        else:
            return False

    def get_rate(self, system, site, other_site):
        R = k_ads_mid_5
        return self.alpha * R  # alpha important for temporal acceleration.

    def do_event(self, system, site, other_site):
        if system.sites[site].covered == 0 and system.sites[site].stype == 2:
        	system.sites[site].covered = 1
    

    def get_involve_other(self):
        return False 














class O2CDesEvent(EventBase):
    """event_0
    The event is H + *o2c -> H*o2c.
    
    """

    def __init__(self, params):
        EventBase.__init__(self, params)

    def possible(self, system, site, other_site):
        if system.sites[site].covered == 1 and system.sites[site].stype == 0:
            return True
        else:
            return False

    def get_rate(self, system, site, other_site):
        R = k_des_mid_0
        return self.alpha * R  # alpha important for temporal acceleration.

    def do_event(self, system, site, other_site):
        system.sites[site].covered = 0

    def get_involve_other(self):
        return False 

class VO2CDesEvent(EventBase):
    """Event 1
    The event is H2O + *2c -> H2o*2c.
    The event is possible if the site is empty.  
    The rate comes from collision theory.  
    Performing the event adds a CO to the site.

    """

    def __init__(self, params):
        EventBase.__init__(self, params)

    def possible(self, system, site, other_site):
        if system.sites[site].covered == 2 and system.sites[site].stype == 0:
            return True
        else:
            return False

    def get_rate(self, system, site, other_site):
        R = k_des_mid_1
        return self.alpha * R  # alpha important for temporal acceleration.

    def do_event(self, system, site, other_site):
        system.sites[site].covered = 0
        system.sites[site].stype = -1

    def get_involve_other(self):
        return False 

class H2DesEvent(EventBase):
    """Event 2
    The event is 1 HTi5C + 1 HO2C <-> 1 H2 + 1 *Ti5C + 1 *O2C
    The event is possible if the site is empty.  
    The rate comes from collision theory.  
    Performing the event adds a CO to the site.

    """

    def __init__(self, params):
        EventBase.__init__(self, params)

    def possible(self, system, site, other_site):
        if (system.sites[site].covered == 1 and system.sites[site].stype == 0 and system.sites[other_site].covered == 1 and system.sites[other_site].stype == 2) or (system.sites[site].covered == 1 and system.sites[site].stype == 2 and system.sites[other_site].covered == 1 and system.sites[other_site].stype == 0):
            return True
        else:
            return False

    def get_rate(self, system, site, other_site):
        R = k_ads_mid_2
        return self.alpha * R  # alpha important for temporal acceleration.

    def do_event(self, system, site, other_site):
        system.sites[site].covered = 0
        system.sites[other_site].covered = 0

    def get_involve_other(self):
        return True 




class O3CDesEvent(EventBase):
    """event 3
    The event is H + *o3c -> H*o3c.
    
    """

    def __init__(self, params):
        EventBase.__init__(self, params)

    def possible(self, system, site, other_site):
        if system.sites[site].covered == 1 and system.sites[site].stype == 1:
            return True
        else:
            return False

    def get_rate(self, system, site, other_site):
        R = k_des_mid_3
        return self.alpha * R  # alpha important for temporal acceleration.

    def do_event(self, system, site, other_site):
        system.sites[site].covered = 0

    def get_involve_other(self):
        return False 


class HO2CDesEvent(EventBase):
    """Event 4
    The event is H + *ho2c -> H2*o2c.
    
    """

    def __init__(self, params):
        EventBase.__init__(self, params)

    def possible(self, system, site, other_site):
        if system.sites[site].covered == 2 and system.sites[site].stype == 0:
            return True
        else:
            return False

    def get_rate(self, system, site, other_site):
        R = k_des_mid_4
        return self.alpha * R  # alpha important for temporal acceleration.

    def do_event(self, system, site, other_site):
        system.sites[site].covered = 1

    def get_involve_other(self):
        return False 

class Ti5CDesEvent(EventBase):
    """Event 5
    The event is H + *Ti5c -> H*Ti5c.
    The event is possible if the site is empty.  
    The rate comes from collision theory.  
    Performing the event adds a CO to the site.

    """

    def __init__(self, params):
        EventBase.__init__(self, params)

    def possible(self, system, site, other_site):
        if system.sites[site].covered == 1 and system.sites[site].stype == 2:
            return True
        else:
            return False

    def get_rate(self, system, site, other_site):
        R = k_des_mid_5
        return self.alpha * R  # alpha important for temporal acceleration.

    def do_event(self, system, site, other_site):
        system.sites[site].covered = 0
    

    def get_involve_other(self):
        return False 



class O3CTi5CDiffEvent(EventBase):
    """Event 7
    The event is 1 HO3C + 1 *Ti5C <-> 1 HTi5c + 1 O3C
    """

    def __init__(self, params):
        EventBase.__init__(self, params)

    def possible(self, system, site, other_site):
        if (system.sites[site].covered == 1 and system.sites[site].stype == 1 and system.sites[other_site].covered == 0 and system.sites[other_site].stype == 2) or (system.sites[site].covered == 1 and system.sites[site].stype == 2 and system.sites[other_site].covered == 0 and system.sites[other_site].stype == 1):
            return True
        else:
            return False

    def get_rate(self, system, site, other_site):
        if system.sites[site].covered == 1 and system.sites[site].stype == 1:
            R = k_fwd_7
        else system.sites[site].covered == 1 and system.sites[site].stype == 2:
            R = k_bwd_7
        return self.alpha * R  # alpha important for temporal acceleration.

    def do_event(self, system, site, other_site):
        covered_temp = system.sites[site].covered 
        stype_temp = system.sites[site].stype
        system.sites[site].covered = system.sites[other_site].covered
        system.sites[site].stype = system.sites[other_site].stype
        system.sites[other_site].stype = stype_temp
        system.sites[site].covered = covered_temp

    def get_involve_other(self):
        return True 
