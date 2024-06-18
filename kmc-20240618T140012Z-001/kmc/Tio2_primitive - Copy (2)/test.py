"""Script that runs the tutorial for A adsorption and desorption.
 
"""
import numpy as np
#from ase.build import fcc100
from .user_sites import Site
from .user_system import System
from .user_kmc import NeighborKMC
from .user_events import (O2CAdsEvent, VO2CAdsEvent, H2AdsEvent, O3CAdsEvent, HO2CAdsEvent, Ti5CAdsEvent,O2CDesEvent, VO2CDesEvent, H2DesEvent, O3CDesEvent, HO2CDesEvent, Ti5CDesEvent )
from ase import *
from ase.io import *
#from ase.visualize import view
    
def run_test():
    """Runs the test of A adsorption and desorption over a surface.

    First, constants are defined and old output files cleared.
    Next, the sites, events, system and simulation objects
    are loaded, and the simulation is performed.

    Last, the results are read in from the generated.txt files,
    and plotted using matplotlib.

    """
    tend = 60 # seconds
    np.savetxt("time.txt", [])
    np.savetxt("coverages.txt", [])

    p = System(atoms=atoms, sites=sites)



    sys2 = read('CONTCAR_243')
    #view(sys2)
    sys = sys2
    #print(sys.positions[0][3])
    del sys[[atom.index for atom in sys if sys.positions[atom.index][2] < 24]]

    Ti5c_list = [32, 35, 24, 30, 37, 34, 26, 28]
    Ti6c_list = [31, 36, 38, 29, 39, 33, 27, 25]


    a_list = sys.get_chemical_symbols()
    Ti_list = [n for n in range(len(a_list)) if a_list[n]=='Ti' ]
    O_list = [n for n in range(len(a_list)) if a_list[n]=='O' ]

    O2c_list = []
    O3c_list = []
    for ind in O_list:
        if sys.positions[ind][2]> 24.8:
            O2c_list.append(ind)
        elif sys.positions[ind][2]< 24.8:
            O3c_list.append(ind)


    print(len(sys.positions), O2c_list, O3c_list, sys.get_distances(0, 25), sys.get_distances(0, 27), sys.get_distances(2, 26))

    for i in range(len(a_list)):
        sx = 0
        for j in range(len(a_list)):
            if sys.get_distances(i,j) < 2.1 and i!=j:
                sx= sx +1
        print(i,':',sx)
    sites = []
    for i in range(len(a_list)):
        if i in O2c_list:
            s = 0
        elif i in O3c_list:
            s = 1
        elif i in Ti5c_list:
            s = 2
        else:
            s = 3
        sites.append(Site(stype = s, covered = 0, ind = i))
    p = System(atoms= sys, sites = sites)
    p.set_neighbors(Ncutoff = 2, pbc = True)
    #view(sys)








    # Set the global neighborlist based on distances:

    events = [O2CAdsEvent, VO2CAdsEvent, H2AdsEvent, O3CAdsEvent, HO2CAdsEvent, Ti5CAdsEvent,O2CDesEvent, VO2CDesEvent, H2DesEvent, O3CDesEvent, HO2CDesEvent, Ti5CDesEvent]

#    parameters = {"pH2": pCO, "pO2": pO2, "T": T,
    parameters = { "Name": "Primitive Simulation"}

    # Instantiate simulator object.
    sim = NeighborKMC(system=p, tend=tend,
                      parameters=parameters,
                      events=events)

    # Run the simulation.
    sim.run_kmc()
    print("Simulation end time reached ! ! !")

if __name__ == '__main__':
    run_test()
