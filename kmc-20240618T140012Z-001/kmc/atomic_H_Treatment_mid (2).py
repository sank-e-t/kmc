#!/usr/bin/env python
# coding: utf-8

# In[1]:



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


# In[ ]:





# # Microkinetic Modelling
# 
# In this code there was one simulation that included all the reactions and there was another which would have all the reaction except one. 
# Hydrogenation of $TiO_2$ by $H_2$.
# 
# Relevant reactions
# 
# 1$H_2(g)+2*_{O_{2c}}\rightleftharpoons 2H*O_{2c}$  , ads/des
# 
# 2$H_2(g)+*_{O_{2c}}+*_{Ti_{5c}}\rightleftharpoons H*{O_{2c}}+H*{Ti_{5c}} $ , ads/des
# 
# 3$H_2(g)+ *_{O_{2c}}\rightleftharpoons H_2 O*{2c}$ ,  ads/des
# 
# 4$H_2O_{*_{2c}}\rightleftharpoons H_2O(g)+V_{O_{2c}}$ ,  ads/des
# 
# 5$H*O_{3c}\rightleftharpoons H*O_{3c,rot}$
# 
# 6$H_2O_{*_{2c}}+*O_{3c}\rightleftharpoons H*O_{2c}+H*O_{3c}$
# 
# 7$H*O_{2c}+*O_{3c}\rightleftharpoons *O_{2c}+H*O_{3c}$
# 
# 8$H*_{O_{3c}}+*_{Ti_{5c}}\rightleftharpoons H*_{Ti_{5c}}+*_{O_{3c}}$
# 
# 9$H*_{O_{3c,rot}}+*_{O_{sub,1}}\rightleftharpoons H*_{O_{sub,1}} + *_{O_{3c}}$
# 
# 10$H*_{O_{3c,rot}}+*_{O_{sub,2}}\rightleftharpoons H*_{O_{sub,2}} + *_{O_{3c,rot}}$
# 
# 11$H*_{O_{sub,1}}+*_{O_{sub,2}}\rightleftharpoons H*_{O_{sub,2}}+*_{O_{sub,1}}$
# 
# 12$H*_{O_{sub,1}}+*_{O_{sub,3}}\rightleftharpoons H*_{O_{sub,3}}+*_{O_{sub,1}}$
# 
# 13$H*_{O_{sub,2}}+*_{O_{sub,3}}\rightleftharpoons H*_{O_{sub,3}}+*_{O_{sub,2}}$
# 
# 14$H*_{O_{sub,3}}\rightleftharpoons H*_{O_{sub,3rot}}$
# 
# 15$H*_{O_{sub,3rot}}+*_{O_{sub,4}}\rightleftharpoons H*_{O_{sub,4}}+*_{O_{sub,3}}$
# 
# 16$H*_{O_{sub,3rot}}+*_{O_{sub,5}}\rightleftharpoons H*_{O_{sub,5}}+*_{O_{sub,3}}$
# 
# 
# 
# Relevant species
# 0. $*O_2c$,
# 
# 1. $*O_3c $, 
# 
# 2. $*Ti_{5c}$,
# 
# 3. $H*_{O_2c }$,
# 
# 4. $H*_{O_3c} $,
# 
# 5. $H*_{Ti_5c }$,
# 
# 6. $H_2O(g)$,
# 
# 7. $H_2 (g)$,
# 
# 8. $H_2 O*_{2c}$,
# 
# 9. $H*O_{sub,1,2,3,4,5}$,
# 
# 14. $*O_{sub,1,2,3,4,5}$
# 
# Relevant reactions for H treatment:
# 
# 0 $ H + *O3C \rightleftharpoons HO3C $
# 
# 1 $ H + *HO2C \rightleftharpoons H2O2C $ 
# 
# 2 $ H + *Ti5C \rightleftharpoons HTi5C $
# 
# 3 $ H2O2C \rightleftharpoons H2O + *2C $
# 
# 4 $ HTi5C + HO2C \rightleftharpoons H2 + *Ti5C + *O2C $
# 
# 5 $ HO3C \rightleftharpoons HO3Crot $
# 
# 6 $ HO3Crot + *Osub \rightleftharpoons HOsub + *O3C $
# 
# 7 $ H2O2C + *O3C \rightleftharpoons HO3C + *HO2C $
# 
# 8 $ HO3C + *Ti5C \rightleftharpoons HTi5c + O3C $
# 

# In[2]:


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


# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:



p_H2 = 1e-5 * au.bar

p_H2O = 1e-8 * p_H2 # #20 bar, 200: 1 bar, 400: 1 Pa 265   #### changed 0405 aenm SI


p_H = 0.7/100.0 * p_H2 # 
p_Pa = p_H2/au.bar*1e5

T_range = [400, 518.15, 531.15, 563.15]#[300, 323.15, 373, 400, 423, 473] #






import time
# solve the MKM:
# Solve the microkinetic initial value problem using scipy.integrate.solve_ivp
# and the implemented numerical representation of the ode and the 
# associated Jacobian.
from scipy.integrate import solve_ivp


RATES=["low", "high"]

O2C = 1.0; H_O2C = 0.0
O3C = 2.0; H_O3C = 0.0
H2_O2C = 0.0
Osub_1 = Osub_2 = Osub_4 = Osub_5 = Ti5C = 1.0
Osub_3 = 2.00
H_O3crot = H_Osub_1 = H_Osub_2 = H_Osub_3 = H_Osub_3rot = 0.0
H_Osub_4 = H_Osub_5 = H2O2C = VO2C = 0.0
H2 = 1.0; H2O = 1.0; # SI unit discrepancy maybe       
H=1.0
H_Ti5C = 0
VOsub_1 = 0
# initial values:
#involved species:'*2C' '*HO2C' '*O2C' '*O3C' '*Osub_1' '*Osub_2' '*Osub_3' '*Osub_4'
# '*Osub_5' '*Ti5C' 'H' 'H*O3Crot' 'H*Osub_1' 'H*Osub_2' 'H*Osub_3'
# 'H*Osub_3rot' 'H*Osub_4' 'H*Osub_5' 'H2' 'H2O' 'H2O2C' 'HO3C' 'HTi5C'

val0 = [VO2C, H_O2C, O2C, O3C, Osub_1, Osub_2, Osub_3,
 Osub_4, Osub_5 ,Ti5C, H ,H_O3crot ,H_Osub_1, H_Osub_2,
 H_Osub_3, H_Osub_3rot, H_Osub_4, H_Osub_5, H2, H2O, H2O2C, H_O3C, H_Ti5C, VOsub_1]

list_species = ['*2C', '*HO2C', '*O2C', '*O3C', '*Osub_1', '*Osub_2', '*Osub_3',
       '*Osub_4', '*Osub_5', '*Ti5C', 'H', 'H*O3Crot', 'H*Osub_1',
       'H*Osub_2', 'H*Osub_3', 'H*Osub_3rot', 'H*Osub_4', 'H*Osub_5',
       'H2', 'H2O', 'H2O2C', 'HO3C', 'HTi5C', 'VOsub_1']
species_dict ={}
for i, list in enumerate(list_species):
    species_dict[list] = val0[i]
print(species_dict)

#alpha = [1e-6, 1e-4, 1e-2, 1, 10]


# # First Set of Simulation below
# 
# All reactions are included

# In[4]:


#print(species_dict)
#The atomic H adsorption microkinetic model!

T_HAX = []
T_LAX = []
Y_HAX = []
Y_LAX = []

A = 6.684 * 3.005
# time span:
t0 = 0
t1 = 18000#86400*2 # 2000
#[ 0.2, 0.4, 0.8, 1, 2, 4, 8, 16, 32, 80, 160]
for T in T_range:
    print('Temperature', T)
    # Reaction 2: 1 HTi5C + 1 HO2C <-> 1 H2 + 1 *Ti5C + 1 *O2C
    
    

    mu = mu_H2_ideal(p_H2,T)
    mol = molecule('H2')
    m_mol = 2 * mol.get_masses()[0]

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
    
    A1 = 1.5
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
    A2 = 11.8
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

    # reaction 6: 1 H2O2C + 1 *O3C <-> 1 HO3C + 1 *HO2C
    E_act_6 = 0.00
    E_rct_6 = 0.00-0.51 # E_final - E_initial
    k_fwd_6 = calc_k_Eyring(T, E_act_5 )
    k_bwd_6 = calc_k_Eyring(T, E_act_5 - E_rct_5 )



    # reaction_7: 1 HO3C + 1 *Ti5C <-> 1 HTi5c + 1 O3C
    E_act_7 = 1.72
    E_rct_7 = 0.19 # E_final - E_initial
    k_fwd_7 = calc_k_Eyring(T, E_act_7 )
    k_bwd_7 = calc_k_Eyring(T, E_act_7 - E_rct_7 )


    # reaction_8: 1 HO3C <-> 1 HO3Crot
    E_act_8 = 0.87  #  try shifting this by.1
    E_rct_8 = 0.07 # E_final - E_initial
    k_fwd_8 = calc_k_Eyring(T, E_act_8 )
    k_bwd_8 = calc_k_Eyring(T, E_act_8 - E_rct_8 )

    # reaction_9:1 HO3Crot + 1 *Osub1 <-> 1 HOsub1 + 1 *O3C
    E_act_9 = 0.43
    E_rct_9 = 0.05-0.07 # E_final - E_initial
    k_fwd_9 = calc_k_Eyring(T, E_act_9 )
    k_bwd_9 = calc_k_Eyring(T, E_act_9 - E_rct_9 )

    # reaction_10: 1 H*O3crot + 1 *Osub_2 <-> 1 *O3c + 1 H*Osub_2
    E_act_10 = 0.43 # changed from catalayud 0405
    E_rct_10 = 0.05-0.07 # E_final - E_initial
    k_fwd_10 = calc_k_Eyring(T, E_act_10 )
    k_bwd_10 = calc_k_Eyring(T, E_act_10 - E_rct_10 )

    # reaction_11: 1 H*Osub_1 + 1 *Osub_2 <-> 1 *Osub_1 + 1 H*Osub_2
    E_act_11 = 0.32
    E_rct_11 = 0.0 # E_final - E_initial
    k_fwd_11 = calc_k_Eyring(T, E_act_11 )
    k_bwd_11 = calc_k_Eyring(T, E_act_11 - E_rct_11 )


    # reaction_12: 1 H*Osub_1 + 1 *Osub_3 <-> 1 *Osub_1 + 1 H*Osub_3
    E_act_12 = 0.53 # changed 0403
    E_rct_12 = 0.0 # E_final - E_initial
    k_fwd_12 = calc_k_Eyring(T, E_act_12 )
    k_bwd_12 = calc_k_Eyring(T, E_act_12 - E_rct_12 )

    # reaction_13: 1 H*Osub_2 + 1 *Osub_3 <-> 1 *Osub_2 + 1 H*Osub_3
    E_act_13 = 0.53 # changed 0403
    E_rct_13 = 0.0 # E_final - E_initial
    k_fwd_13 = calc_k_Eyring(T, E_act_13 )
    k_bwd_13 = calc_k_Eyring(T, E_act_13 - E_rct_13 )


    # reaction_14: 1 H*Osub_3 <-> 1 H*Osub_3rot 
    E_act_14 = 0.97 #8# changed 0403
    E_rct_14 = 0.00 # E_final - E_initial
    k_fwd_14 = calc_k_Eyring(T, E_act_14 )
    k_bwd_14 = calc_k_Eyring(T, E_act_14 - E_rct_14 )

    # reaction_15: 1 H*Osub_3rot + 1 *Osub_4 <-> 1 *Osub_3 + 1 H*Osub_4
    E_act_15 = 0.53 # changed 0403   50
    E_rct_15 = 0.00 # E_final - E_initial
    k_fwd_15 = calc_k_Eyring(T, E_act_15 )
    k_bwd_15 = calc_k_Eyring(T, E_act_15 - E_rct_15 )

    # reaction_16: 1 H*Osub_3rot + 1 *Osub_5 <-> 1 *Osub_3 + 1 H*Osub_5
    E_act_16 = 0.53 # # changed 0403 0.5
    E_rct_16 = 0.0 # E_final - E_initial
    k_fwd_16 = calc_k_Eyring(T, E_act_16 )
    k_bwd_16 = calc_k_Eyring(T, E_act_16 - E_rct_16 )



    # Reaction 17 : H*Ti_5c + *O_2c <-> *Ti_5c + H*O_2c  DOI: 10.1021/acs.jpcc.8b05251  !!!!! O2c to Ti5c transfer
    E_act_17 =  0.99
    E_rct_17 = -1.19
    k_fwd_17 = calc_k_Eyring(T, E_act_17)
    k_bwd_17 = calc_k_Eyring(T, E_act_17- E_rct_17)

    
    # Reaction 25 : VO2c + Osub1 <-> O2c + VOsub1 !!!! Vacancy migration included https://doi.org/10.1021/acs.jpclett.2c03827
    E_act_25 = 0.98 # 
    E_rct_25 = 0.73 # 
    k_fwd_25 = calc_k_Eyring(T, E_act_25)
    k_bwd_25 = calc_k_Eyring(T, E_act_25-E_rct_25)

    
    y0 = []
    
    
    # Reaction 1: H2O2C <-> 1 H2O + 1 *2C
    mu = mu_H2O_ideal(p_H2O*A/A2,T)
    H2O = molecule('H2O')
    m_H2O = H2O.get_masses().sum()
    sigma_H2O = 2
    E_rct_1 = +0.51-1.42 #-1.04 # E_ads - E_des # Efinal-Einitial
    E_act_des_1 = 1.49 # https://pubs.rsc.org/en/content/articlepdf/2017/cy/c6cy02007k
    '''
    if iter != 0 and  VO2C > 0.135:
        
        E_act_des_1 = 0.91 + 0.8 # Eact- Einitial
    else:
        E_act_des_1 = 0.91
    '''
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
    
    

    rctsys = mkm.ReactionSystem()
    
    
    rctsys.add_reaction(' H2O2C <-> H2O + *2C ', fwd_rate=k_des_mid_1, bwd_rate=k_ads_mid_1)
    rctsys.add_reaction('  HTi5C + *HO2C <-> H2 + *Ti5C + *O2C  ', fwd_rate=k_des_mid_2, bwd_rate=k_ads_mid_2)
    rctsys.add_reaction(' H + *O3C <-> HO3C ', fwd_rate=k_ads_mid_3, bwd_rate=k_des_mid_3)
    rctsys.add_reaction(' H + *HO2C <-> H2O2C ', fwd_rate=k_ads_mid_4, bwd_rate=k_des_mid_4)
    rctsys.add_reaction(' H + *O2C <-> *HO2C ', fwd_rate=k_ads_mid_0, bwd_rate=k_des_mid_0)
    #rctsys.add_reaction(' H + *Ti5C <-> HTi5C ', fwd_rate=k_ads_mid_5, bwd_rate=k_des_mid_5)
    
    rctsys.add_reaction(' H2O2C + *O3C <-> HO3C + *HO2C ', fwd_rate=k_fwd_6, bwd_rate=k_bwd_6)
    rctsys.add_reaction('  HO3C + *Ti5C <-> HTi5C + *O3C  ', fwd_rate=k_fwd_7, bwd_rate=k_bwd_7)
    rctsys.add_reaction(' HO3C <-> H*O3Crot ', fwd_rate=k_fwd_8, bwd_rate=k_bwd_8)
    
    rctsys.add_reaction('  H*O3Crot + *Osub_1 <-> H*Osub_1 + *O3C ', fwd_rate=k_fwd_9, bwd_rate=k_bwd_9)
    rctsys.add_reaction('H*O3Crot + *Osub_2 <-> *O3C + H*Osub_2', fwd_rate=k_fwd_10, bwd_rate=k_bwd_10)
    rctsys.add_reaction('H*Osub_1 + *Osub_2 <-> *Osub_1 + H*Osub_2', fwd_rate=k_fwd_11, bwd_rate=k_bwd_11)
    rctsys.add_reaction('H*Osub_1 + *Osub_3 <-> *Osub_1 + H*Osub_3', fwd_rate=k_fwd_12, bwd_rate=k_bwd_12)
    rctsys.add_reaction('H*Osub_2 + *Osub_3 <-> *Osub_2 + H*Osub_3', fwd_rate=k_fwd_13, bwd_rate=k_bwd_13)
    rctsys.add_reaction('H*Osub_3 <-> H*Osub_3rot', fwd_rate=k_fwd_14, bwd_rate=k_bwd_14)
    rctsys.add_reaction('H*Osub_3rot + *Osub_4 <-> *Osub_3 + H*Osub_4', fwd_rate=k_fwd_15, bwd_rate=k_bwd_15)
    rctsys.add_reaction('H*Osub_3rot + *Osub_5 <-> *Osub_3 + H*Osub_5', fwd_rate=k_fwd_16, bwd_rate=k_bwd_16)
    #rctsys.add_reaction(' HTi5C + *O2C <-> *HO2C + *Ti5C', fwd_rate=k_fwd_17, bwd_rate=k_bwd_17)
    rctsys.add_reaction(' *2C + *Osub_1 <-> *O2C + VOsub_1', fwd_rate=k_fwd_25, bwd_rate=k_bwd_25)


    
    for element in rctsys.species:    
        y0.append(species_dict[element])
                                                
    rctsys.constant_species = [ 'H', 'H2O', 'H2']
    print('rctsys', rctsys, rctsys.species)

    runtime = []

        # Solve the IVP:
        # Let the solver choose the time steps ...

    tic = time.time()
    print(t0, t1, y0)
    my_solution = solve_ivp(rctsys.dy_dt_num, t_span=[t0,t1], y0=y0, 
                            method='LSODA', t_eval=None, dense_output=False,
                            events=None, vectorized=False, args=None, jac=rctsys.J_num,
                            rtol=1.e-8, atol=1.e-12)


    toc = time.time()
    print('run time: ', toc - tic)
    runtime.append(toc-tic)
    print(my_solution.message)
    print(my_solution.nfev)
    print(my_solution.njev)
    N=len(rctsys.species)
    

    y_total = my_solution.y
    t_total = my_solution.t
    
    #timeax=my_solution.t

    Y_HAX.append(y_total)#y_high = my_solution.y
    T_HAX.append(t_total)#timeax_high=my_solution.t


# In[7]:





RATE = 'Mean'
for ind, T in enumerate(T_range):
    
    
    plot_species = [ 'H*O3Crot', 'HO3C', 'H*Osub_1', 'H*Osub_2', 'H*Osub_3', 'H*Osub_3rot' , 'H*Osub_4', 'H*Osub_5', '*HO2C', 'H*Ti5C', 'H2O2C']
    #plot_species = [ '*HO2C''H2O2C''H*O3Crot' 'H*Osub_1' 'H*Osub_2' 'H*Osub_3', 'H*Osub_3rot' 'H*Osub_4' 'H*Osub_5' 'H2' 'H2O' 'H2O2C' 'HO3C' 'HTi5C'
    y_high = Y_HAX[ind]
    timeax_high = T_HAX[ind]
    
    
    #y_low = Y_LAX[ind]
    #timeax_low = T_LAX[ind]
    
    adsorbed_high=np.zeros(len(timeax_high))
    #adsorbed_low=np.zeros(len(timeax_low))
    for nl,l in enumerate(rctsys.species):
            if l in plot_species:
                #m.plot(my_solution.t,my_solution.y[nl,:], '-', label=l+' auto times')
                adsorbed_high+= y_high[nl,:]
                #adsorbed_low+= y_low[nl,:]
    
    m.plot(timeax_high, adsorbed_high, label= str(T) + 'K')
    #m.plot(timeax_low, adsorbed_low,label= str(T) + 'K')
m.xscale("log")
m.xlabel('Time (s)')
m.ylabel('Coverage')
m.legend()
m.grid()
#m.title('total hydrogen (%s limit)'%(RATE))
m.savefig('atom_H_totalhydrogen_%dK_%.2EbarH2_time_%d_rate_%s_o2c_1__mean.pdf' %(T,p_H2/au.bar, t1-t0, RATE ), bbox_inches='tight', dpi = 400)
m.show()     

for ind, T in enumerate(T_range):
    
    
    plot_species = [  'HO3C', '*HO2C', 'H*Ti5C', 'H2O2C',  'H*O3Crot']
    y_high = Y_HAX[ind]
    timeax_high = T_HAX[ind]
    
    
    #y_low = Y_LAX[ind]
    #timeax_low = T_LAX[ind]
    
    adsorbed_high=np.zeros(len(timeax_high))
    #adsorbed_low=np.zeros(len(timeax_low))

    for nl,l in enumerate(rctsys.species):
            if l in plot_species:
                #m.plot(my_solution.t,my_solution.y[nl,:], '-', label=l+' auto times')
                adsorbed_high+= y_high[nl,:]
                #adsorbed_low+= y_low[nl,:]
    m.plot(timeax_high, adsorbed_high, label= str(T) + 'K')
    #m.plot(timeax_low, adsorbed_low,label= str(T) + 'K')

m.xscale("log")
m.xlabel('Time (s)')
m.ylabel('Coverage')
m.legend()
#m.title('Surface hydrogen (%s limit)'%(RATE))
m.grid()
m.savefig('atom_H_surfacehydrogen_%dK_%.2EbarH2_time_%d_rate_%s_o2c_1__mean.pdf' %(T,p_H2/au.bar, t1-t0, RATE ), bbox_inches='tight', dpi = 400)
m.show()


for ind, T in enumerate(T_range):
    
    
    plot_species = [ 'H*Osub_1', 'H*Osub_2', 'H*Osub_3', 'H*Osub_3rot' , 'H*Osub_4', 'H*Osub_5']

    y_high = Y_HAX[ind]
    timeax_high = T_HAX[ind]
    
    
    #y_low = Y_LAX[ind]
    #timeax_low = T_LAX[ind]
    
    adsorbed_high=np.zeros(len(timeax_high))
    #adsorbed_low=np.zeros(len(timeax_low))

    for nl,l in enumerate(rctsys.species):
            if l in plot_species:
                #m.plot(my_solution.t,my_solution.y[nl,:], '-', label=l+' auto times')
                adsorbed_high+= y_high[nl,:]
                #adsorbed_low+= y_low[nl,:]
    
    m.plot(timeax_high, adsorbed_high, label= str(T) + 'K')
    #m.plot(timeax_low, adsorbed_low,label= str(T) + 'K')

m.xscale("log")
m.xlabel('Time (s)')
m.ylabel('Coverage')
m.legend()
#m.title('subsurface hydrogen (%s limit)'%(RATE))
m.grid()
m.savefig('atom_H_subsurfacehydrogen_%dK_%.2EbarH2_time_%d_rate_%s_o2c_1__mean.pdf' %(T,p_H2/au.bar, t1-t0, RATE ), bbox_inches='tight', dpi = 400)
m.show()


for ind, T in enumerate(T_range):
    
    
    plot_species = [ '*2C']

    y_high = Y_HAX[ind]
    timeax_high = T_HAX[ind]
    
    
    #y_low = Y_LAX[ind]
    #timeax_low = T_LAX[ind]
    
    adsorbed_high=np.zeros(len(timeax_high))
    #adsorbed_low=np.zeros(len(timeax_low))

    for nl,l in enumerate(rctsys.species):
            if l in plot_species:
                #m.plot(my_solution.t,my_solution.y[nl,:], '-', label=l+' auto times')
                adsorbed_high+= y_high[nl,:]
                #adsorbed_low+= y_low[nl,:]
    m.plot(timeax_high, adsorbed_high, label= str(T) + 'K')
    #m.plot(timeax_low, adsorbed_low,label= str(T) + 'K')

m.xscale("log")
m.xlabel('Time (s)')
m.ylabel('Coverage')
m.legend()
#m.title('oxygen vacancy (%s limit)'%(RATE))
m.grid()
m.savefig('atom_H_oxygenvacancy_%dK_%.2EbarH2_time_%d_rate_%s_o2c_1__mean.pdf' %(T,p_H2/au.bar, t1-t0, RATE ), bbox_inches='tight', dpi = 400)
m.show()
'''

for ind, T in enumerate(T_range):
    
    
    plot_species = [ 'H*O3Crot', 'HO3C', 'H*Osub_1', 'H*Osub_2', 'H*Osub_3', 'H*Osub_3rot' , 'H*Osub_4', 'H*Osub_5', '*HO2C', 'H*Ti5C', 'H2O2C']
    #plot_species = [ '*HO2C''H2O2C''H*O3Crot' 'H*Osub_1' 'H*Osub_2' 'H*Osub_3', 'H*Osub_3rot' 'H*Osub_4' 'H*Osub_5' 'H2' 'H2O' 'H2O2C' 'HO3C' 'HTi5C'
    y_high = Y_HAX[ind]
    timeax_high = T_HAX[ind]
    
    
    #y_low = Y_LAX[ind]
    #timeax_low = T_LAX[ind]
    
    adsorbed_high=np.zeros(len(timeax_high))
    #adsorbed_low=np.zeros(len(timeax_low))
    for nl,l in enumerate(rctsys.species):
            if l in plot_species:
                #m.plot(my_solution.t,my_solution.y[nl,:], '-', label=l+' auto times')
                adsorbed_high+= y_high[nl,:]
                #adsorbed_low+= y_low[nl,:]
    #m.xscale("log")
    m.plot(timeax_high, adsorbed_high, label= str(T) + 'K')
    #m.plot(timeax_low, adsorbed_low,label= str(T) + 'K')
m.xlabel('Time (s)')
m.ylabel('Coverage')
m.xlim([0,50])
m.legend()
m.grid()
#m.title('total hydrogen (%s limit)'%(RATE))
m.savefig('nolog_atom_H_totalhydrogen_%dK_%.2EbarH2_time_%d_rate_%s_o2c_1__mean.pdf' %(T,p_H2/au.bar, t1-t0, RATE ), bbox_inches='tight', dpi = 400)
m.show()     


for ind, T in enumerate(T_range):
    
    
    plot_species = [  'HO3C', '*HO2C', 'H*Ti5C', 'H2O2C']
    y_high = Y_HAX[ind]
    timeax_high = T_HAX[ind]
    
    
    #y_low = Y_LAX[ind]
    #timeax_low = T_LAX[ind]
    
    adsorbed_high=np.zeros(len(timeax_high))
   # adsorbed_low=np.zeros(len(timeax_low))

    for nl,l in enumerate(rctsys.species):
            if l in plot_species:
                #m.plot(my_solution.t,my_solution.y[nl,:], '-', label=l+' auto times')
                adsorbed_high+= y_high[nl,:]
                #adsorbed_low+= y_low[nl,:]
    #m.xscale("log")
    m.plot(timeax_high, adsorbed_high, label= str(T) + 'K')
    #m.plot(timeax_low, adsorbed_low,label= str(T) + 'K')
m.xlabel('Time (s)')
m.ylabel('Coverage')
m.xlim([0,50])
m.legend()
#m.title('Surface hydrogen (%s limit)'%(RATE))
m.grid()
m.savefig('nolog_atom_H_surfacehydrogen_%dK_%.2EbarH2_time_%d_rate_%s_o2c_1__mean.pdf' %(T,p_H2/au.bar, t1-t0, RATE ), bbox_inches='tight', dpi = 400)
m.show()


for ind, T in enumerate(T_range):
    
    
    plot_species = [ 'H*Osub_1', 'H*Osub_2', 'H*Osub_3', 'H*Osub_3rot' , 'H*Osub_4', 'H*Osub_5']
    y_high = Y_HAX[ind]
    timeax_high = T_HAX[ind]
    
    
    #y_low = Y_LAX[ind]
    #timeax_low = T_LAX[ind]
    

    adsorbed_high=np.zeros(len(timeax_high))
    #adsorbed_low=np.zeros(len(timeax_low))

    for nl,l in enumerate(rctsys.species):
            if l in plot_species:
                #m.plot(my_solution.t,my_solution.y[nl,:], '-', label=l+' auto times')
                adsorbed_high+= y_high[nl,:]
                #adsorbed_low+= y_low[nl,:]
    #m.xscale("log")
    m.plot(timeax_high, adsorbed_high, label= str(T) + 'K')
    #m.plot(timeax_low, adsorbed_low,label= str(T) + 'K')
m.xlabel('Time (s)')
m.ylabel('Coverage')

m.legend()
#m.title('subsurface hydrogen (%s limit)'%(RATE))
m.grid()
m.savefig('nolog_atom_H_subsurfacehydrogen_%dK_%.2EbarH2_time_%d_rate_%s_o2c_1__mean.pdf' %(T,p_H2/au.bar, t1-t0, RATE ), bbox_inches='tight', dpi = 400)
m.show()

for ind, T in enumerate(T_range):
    
    
        
    plot_species = [ '*2C']

    y_high = Y_HAX[ind]
    timeax_high = T_HAX[ind]
    
    
    #y_low = Y_LAX[ind]
    #timeax_low = T_LAX[ind]
    
    adsorbed_high=np.zeros(len(timeax_high))
    #adsorbed_low=np.zeros(len(timeax_low))

    for nl,l in enumerate(rctsys.species):
            if l in plot_species:
                #m.plot(my_solution.t,my_solution.y[nl,:], '-', label=l+' auto times')
                adsorbed_high+= y_high[nl,:]
                #adsorbed_low+= y_low[nl,:]
    #m.xscale("log")
    m.plot(timeax_high, adsorbed_high, label= str(T) + 'K')
    #m.plot(timeax_low, adsorbed_low,label= str(T) + 'K')

m.xlabel('Time (s)')
m.ylabel('Coverage')
m.legend()
#m.title('oxygen vacancy (%s limit)'%(RATE))
m.grid()
m.savefig('nolog_atom_H_oxygenvacancy_%dK_%.2EbarH2_time_%d_rate_%s_o2c_1__mean.pdf' %(T,p_H2/au.bar, t1-t0, RATE ), bbox_inches='tight', dpi = 400)
m.show()

'''




for nl,l in enumerate(rctsys.species):
    for ind, T in enumerate(T_range):
        

        y_high = Y_HAX[ind]
        timeax_high = T_HAX[ind]
        
    
        m.plot(timeax_high, y_high[nl, :], label= str(T) + 'K') #"high")
        #m.plot(timeax_low, y_high[nl, :],label= str(T) + 'K')
    if nl == 22:
        m.yscale('log')
    m.xscale("log")
    m.xlabel('Time (s)')
    m.ylabel('Coverage')
    m.legend()
    #m.title('%s (%s limit)'%(l, RATE))
    m.grid()
    m.savefig('atomicH_%d_%dK_%.2EbarH2_time_%d_rate_%s_o2c_1__mean.pdf' %(nl,T,p_H2/au.bar, t1-t0, RATE ), bbox_inches='tight', dpi = 400)
    m.show()


# In[29]:





RATE = 'Mean'

for ind, T in enumerate(T_range):
    
    
    plot_species = [ '*2C']

    y_high = Y_HAX[ind]
    timeax_high = T_HAX[ind]
    
    
    #y_low = Y_LAX[ind]
    #timeax_low = T_LAX[ind]
    
    adsorbed_high=np.zeros(len(timeax_high))
    #adsorbed_low=np.zeros(len(timeax_low))

    for nl,l in enumerate(rctsys.species):
            if l in plot_species:
                #m.plot(my_solution.t,my_solution.y[nl,:], '-', label=l+' auto times')
                adsorbed_high+= y_high[nl,:]
                #adsorbed_low+= y_low[nl,:]
    m.plot(timeax_high, adsorbed_high, label= str(T) + 'K')
    #m.plot(timeax_low, adsorbed_low,label= str(T) + 'K')

#m.xscale("log")
m.xlim([0, 200])
m.xlabel('Time (s)')
m.ylabel('Coverage')
m.legend()
#mtitle('oxygen vacancy (%s limit)'%(RATE))
m.grid()
#m.savefig('atom_H_oxygenvacancy_%dK_%.2EbarH2_time_%d_rate_%s_o2c_1_ti5c_const.pdf' %(T,p_H2/au.bar, t1-t0, RATE ), bbox_inches='tight', dpi = 400)
m.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[9]:





# In[ ]:





# In[ ]:




