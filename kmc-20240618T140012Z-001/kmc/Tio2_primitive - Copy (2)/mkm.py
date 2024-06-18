import numpy as np
import sympy as sp


def get_coefficients(species_with_coeff):
    """ Extract the stoichiometric coefficients from a list of species with
    stoichiometric coefficients such as ['O2', '2*'].
    Returns the coefficients (here [1, 2]) and the species without
    coefficients (here ['O2', '*']).

    species_with_coeff: list of strings
    """
    coefficients = []
    species_no_coeff = []
    for specie in species_with_coeff:
        coeff = 1
        specie_new = specie
        for n_specie in range(1, len(specie)):
            if specie[:n_specie].isnumeric():
                coeff = int(specie[:n_specie])
                specie_new = specie[n_specie:].strip()
        coefficients.append(coeff)
        species_no_coeff.append(specie_new)
    return coefficients, species_no_coeff


def string_to_reaction(reaction_str):
    step_0 = reaction_str.split('<->')
    assert len(step_0) == 2
    educts = step_0[0].strip()
    educts = educts.split('+')
    educts = [educt.strip() for educt in educts]
#   print('Educts: ', educts)
    coeff_educts, educts = get_coefficients(educts)
#   print('Educts: ', educts)
    products = step_0[1].strip()
    products = products.split('+')
    products = [product.strip() for product in products]
    coeff_products, products = get_coefficients(products)
#   print('Products: ', products)
    species = set(educts).union(set(products))
    return educts, products, coeff_educts, coeff_products, species


def reaction_to_string(reaction_dict):
    """Generate string of reaction from educts, products and coefficients.
    e.g., reaction_dict = {'educts': [O2, *], 'coeff_educts': [1, 2],
                           'products': [O*], 'coeff_products': [2]}"""
    string = ''
    for neduct, educt in enumerate(reaction_dict['educts']):
        if neduct > 0:
            string += ' + '
        string += '%d %s' % (reaction_dict['coeff_educts'][neduct], educt)
    string += ' <-> '
    for nproduct, product in enumerate(reaction_dict['products']):
        if nproduct > 0:
            string += ' + '
        string += '%d %s' % (reaction_dict['coeff_products'][nproduct],
                             product)
    return string


class ReactionSystem:
    """Class to set up a system of coupled chemical reactions
    and to derive the corresponding microkinetic model.
    Reactions can be added via the corresponding chemical formula
    and forward and backward rates via the method add_reaction.

    The rates can be changes by the method set_rates.

    Finally, the temporal derivative of the species concentrations
    according to the law of mass action using the method dydt.

    The system of ordinary differential equations can then be solved
    e.g. with ode from scipy.integrate"""
    def __init__(self):
        self.reactions = []
        self.species = []
        self.constant_species = []

    def add_reaction(self, reaction_str, fwd_rate, bwd_rate):
        """Specify reaction in the form of a string: e.g.
        reaction_str = 'O2 + 2 * <-> O2*'. The forward and
        backward rates are given as floating numbers.

        reaction_str: str
        Specify the chemical reaction in the form 'aA + bB + cC <-> dD + eE'.

        fwd_rate: float
        The forward rate of the reaction.

        bwd_rate: float
        The backward rate of the reaction.
        """
        educts, products, coeff_educts, coeff_products,\
            species = string_to_reaction(reaction_str)
        reaction = {'educts': educts, 'coeff_educts': coeff_educts,
                    'products': products, 'coeff_products': coeff_products,
                    'fwd_rate': fwd_rate, 'bwd_rate': bwd_rate}
        self.reactions.append(reaction)
#       print(list(species))
#       print(self.species)
#       list(species) + list(self.species)
        self.species = np.sort(np.unique(list(self.species) + list(species)))

    def __str__(self):
        str = ''
        for rct in self.reactions:
            str += (
                reaction_to_string(rct)
                + ' fwd_rate: %f bwd_rate:%f\n' % (rct['fwd_rate'],
                                                   rct['bwd_rate']))
        return str

    @property
    def dy_dt_num(self):
        print('Set up ode in symbolic form.')
        self.calc_dy_dt_sym()
        print('Set up ode in numerical form.')
        self.calc_dy_dt_num()
        return self._dy_dt_num

    @property
    def dy_dt_sym(self):
        print('Set up ode in symbolic form.')
        self.calc_dy_dt_sym()
        return self._dy_dt_sym

    @property
    def J_sym(self):
        print('Set up ode in symbolic form.')
        self.calc_dy_dt_sym()
        print('Set up Jacobian in symbolic form.')
        self.calc_jac_dy_dt_sym()
        return self._J_sym

    @property
    def J_num(self):
        print('Set up ode in symbolic form.')
        self.calc_dy_dt_sym()
        print('Set up Jacobian in symbolic form.')
        self.calc_jac_dy_dt_sym()
        print('Set up Jacobian in numeric form.')
        self.calc_jac_dy_dt_num()
        return self._J_num

    def set_rates(self, fwd_rates, bwd_rates):
        """Set forward and backward rates for the different reactions.

        fwd_rates: list of floats
        Sets forward rate for each reaction

        bwd_rates: list of floats
        Sets backward rate for each reaction """
        assert len(fwd_rates) == len(self.reactions)
        assert len(bwd_rates) == len(self.reactions)
        for nrct, rct in enumerate(self.reactions):
            rct['fwd_rate'] = fwd_rates[nrct]
            rct['bwd_rate'] = bwd_rates[nrct]

    def calc_dy_dt_sym(self):
        """
        Here the system of ordinary differential equations is set up
        which corresponds to the reactions defined in self.reactions.
        Here the functional form f_i(t,y) of the time derivatives
        dy_i/dt = f_i(t,y) is determined. Here, y_i is the specie with
        index i, i.e. self.species[i]

        constant species are taken into account if constant species are
        set. Constant species do not vary over time (e.g. gas at
        constant pressure).

        The functions f_i(t,y) are saved in symbolic form (SymPy)
        in the variable self._dy_dt_sym.
        """
        var_species = [specie for specie in self.species if
                       specie not in self.constant_species]
        n_of_species = len(self.species)
        # the list of functions f_i describing the time derivatives
        # dy_i/dt
        fct_list = [0] * n_of_species
        for n_rct, rct in enumerate(self.reactions):
            educts = rct['educts']
            products = rct['products']
            # list of indices of the educts and products in the reaction:
            # list_var_educt_indices = []
            list_educt_indices = []
            list_product_indices = []
            for educt in educts:
                assert educt in self.species
                nofspecie = np.where(self.species == educt)[0]
                list_educt_indices.append(nofspecie[0])
            for product in products:
                assert product in self.species
                nofspecie = np.where(self.species == product)[0]
                list_product_indices.append(nofspecie[0])
                # print('indices of educts: ', list_educt_indices)
                # print('indices of products: ', list_product_indices)
            # the raw forward rate of the reaction
            # rct raw_fwd_rate = k_fwd * [A]**nu_A * [B]**nu_B * ...
            raw_fwd_rate = sp.Symbol('k_fwd_%03d' % n_rct)
            for n_educt, educt in enumerate(educts):
                educt_ind = list_educt_indices[n_educt]
                coeff_educt = rct['coeff_educts'][n_educt]
                raw_fwd_rate *= sp.Symbol('y_%03d' % educt_ind)**coeff_educt
                # print('educt mass law: ', educt)
                # print('y_educt: ', y[educt_ind])
            # print('raw_fwd_rate: ', raw_fwd_rate)
            # the raw backward rate of the reaction rct
            # raw_bwd_rate = k_bwd * [C]**nu_C * [D]**nu_D * ...
            raw_bwd_rate = sp.Symbol('k_bwd_%03d' % n_rct)
            for n_product, product in enumerate(products):
                product_ind = list_product_indices[n_product]
                coeff_product = rct['coeff_products'][n_product]
                raw_bwd_rate *= \
                    sp.Symbol('y_%03d' % product_ind)**coeff_product
            # print('raw_bwd_rate: ', raw_bwd_rate)
            # apply raw forward and backward rates
            # to all variable educts and products:
            for n_educt, educt in enumerate(educts):
                if (educt in var_species):
                    educt_ind = list_educt_indices[n_educt]
                    coeff_educt = rct['coeff_educts'][n_educt]
                    fct_list[educt_ind] -= coeff_educt * raw_fwd_rate
                    fct_list[educt_ind] += coeff_educt * raw_bwd_rate
                    # dydt[educt_ind] -= coeff_educt * raw_fwd_rate
                    # dydt[educt_ind] += coeff_educt * raw_bwd_rate
                    # print(educt, 'fwd_rate: ', -coeff_educt * raw_fwd_rate)
                    # print(educt, 'bwd_rate: ', +coeff_educt * raw_bwd_rate)
            for n_product, product in enumerate(products):
                if (product in var_species):
                    # print(product)
                    product_ind = list_product_indices[n_product]
                    coeff_product = rct['coeff_products'][n_product]
                    fct_list[product_ind] += coeff_product * raw_fwd_rate
                    fct_list[product_ind] -= coeff_product * raw_bwd_rate
                    # dydt[product_ind] += coeff_product * raw_fwd_rate
                    # dydt[product_ind] -= coeff_product * raw_bwd_rate
                    # print(product, 'fwd_rate: ', +coeff_educt * raw_fwd_rate)
                    # print(product, 'bwd_rate: ', -coeff_educt * raw_bwd_rate)
        self._dy_dt_sym = sp.Matrix(fct_list)
        # update Jacobian:
        self.calc_jac_dy_dt_sym()

    def calc_jac_dy_dt_sym(self):
        """Determine the Jacobian matrix from the set of differential equations
        self._dy_dt_sym in symbolic form. self._dy_dt_sym[ind] = f_ind(y,t).
        If dy_i/dt = f_i then the Jacobian J_ij = J[i][i]
        is given by J_ij = df_i/dy_j where y_j is the concentration of
        specie self.species[j].
        """
        # Create the y and t symbols:
        # t = sp.symbols('t')
        y_str = ''
        nofspecies = len(self.species)
        for n in range(nofspecies):
            y_str += 'y_%03d ' % (n)
        y = sp.symbols(y_str)
        # Determine the Jacobian using SymPy:
        self._J_sym = self._dy_dt_sym.jacobian(y)

    def calc_jac_dy_dt_num(self, k=None):
        """Determine the Jacobian matrix from the symbolic form of the
        Jacobian self._J_sym in numeric form as python function.
        """
        # Create the k symbols:
        k_str = ''
        nofrcts = len(self.reactions)
        for n in range(nofrcts):
            k_str += 'k_fwd_%03d k_bwd_%03d ' % (n, n)
        k_sym = sp.symbols(k_str)
        # Make a python function that allows to
        # explicitly set the rates k_fwd_%03d, k_bwd_%03d:
        J_num_k = sp.lambdify((k_sym,), self._J_sym)
        # the numerical values for the rates k:
        if k is None:
            k = []
            for rct in self.reactions:
                k += [rct['fwd_rate'], rct['bwd_rate']]
        assert len(k) == len(self.reactions) * 2
        J_num = sp.Matrix(J_num_k(k))
        # Now determine the numerical python function with parameters
        # t and y. t is the time variable and y a list with the
        # species concentrations.
        # Create the y and t symbols:
        t = sp.symbols('t')
        y_str = ''
        nofspecies = len(self.species)
        for n in range(nofspecies):
            y_str += 'y_%03d ' % (n)
        y = sp.symbols(y_str)
        self._J_num = sp.lambdify((t, y,), J_num)

    def calc_dy_dt_num(self, k=None):
        """Determines the python function that returns numerical values
        for the functions f_i(t,y) of the system of differential equations
        defined in self._dy_dt_sym.

        k: None or list or array of floats
        The values of the forward and backward reaction rates
        k = [k_fwd_000, k_bwd_000, k_fwd_001, k_bwd_001, ...].

        If k is None, the rates from the ReactionSystem object
        will be used, see self.reactions and self.set_rates

        The functions f_i(t,y) are saved in numeric form (python functions)
        in the variable self._dy_dt_num.
        """
        # define the variables for the forward and backward rates:
        k_str = ''
        nofrcts = len(self.reactions)
        for n in range(nofrcts):
            k_str += 'k_fwd_%03d k_bwd_%03d ' % (n, n)
        k_sym = sp.symbols(k_str)
        # print(k_sym)
        # Make a python function that allows to
        # explicitly set the rates k_fwd_%03d, k_bwd_%03d:
        dy_dt_num_k = sp.lambdify((k_sym,), self._dy_dt_sym)

        # the numerical values for the rates k:
        if k is None:
            k = []
            for rct in self.reactions:
                k += [rct['fwd_rate'], rct['bwd_rate']]
        assert len(k) == len(self.reactions) * 2

        # The SymPy function featuring the explicit values for
        # the rates k:
        # print(k)
        dy_dt_num = sp.Matrix(dy_dt_num_k(k))
        # return dy_dt_num
        # Now determine the numerical python function with parameters
        # t and y. t is the time variable and y a list with the
        # species concentrations.
        # Create the y and t symbols:
        t = sp.symbols('t')
        y_str = ''
        nofspecies = len(self.species)
        for n in range(nofspecies):
            y_str += 'y_%03d ' % (n)
        y = sp.symbols(y_str)
        print('Now self._dy_dt_num is available')
        self._dy_dt_num = sp.lambdify((t, y,), dy_dt_num)

    # time derivatives of the different chemical species according to the
    # system of chemical reactions and the law of mass action
    def dydt(self, t, y):
        """
        Here the system of ordinary differential equations is set up
        which corresponds to the reactions defined in self.reactions.
        dydt is in a form that can be used by the ode solver of
        scipy.integrate.

        t: float
        The time coordinate

        y: list or array of floats
        The concentrations of the various species

        constant species are taken into account if constant species are
        set. Constant species do not vary over time (e.g. gas at
        constant pressure).

        The original version to obtain the numerical version of the functions
        f_i(t,y) in the ode set dy_i/dt = f_i(t,y). Use self.dy_dt_num instead!
        """
        var_species = [specie for specie in self.species if
                       specie not in self.constant_species]
        n_of_species = len(self.species)
        # n_of_var_species = len(var_species)
        dydt = np.zeros(n_of_species)
        # print('variable species: ', var_species)
        # print('all species: ', self.species)
        for rct in self.reactions:
            # print(reaction_to_string(rct))
            educts = rct['educts']
            products = rct['products']
            # list of indices of the educts and products in the reaction:
            # list_var_educt_indices = []
            list_educt_indices = []
            list_product_indices = []
            for educt in educts:
                assert educt in self.species
                nofspecie = np.where(self.species == educt)[0]
                list_educt_indices.append(nofspecie[0])
            for product in products:
                assert product in self.species
                nofspecie = np.where(self.species == product)[0]
                list_product_indices.append(nofspecie[0])
                # print('indices of educts: ', list_educt_indices)
                # print('indices of products: ', list_product_indices)
            # the raw forward rate of the reaction
            # rct raw_fwd_rate = k_fwd * [A]**nu_A * [B]**nu_B * ...
            raw_fwd_rate = rct['fwd_rate']
            for n_educt, educt in enumerate(educts):
                educt_ind = list_educt_indices[n_educt]
                coeff_educt = rct['coeff_educts'][n_educt]
                raw_fwd_rate *= y[educt_ind]**coeff_educt
                # print('educt mass law: ', educt)
                # print('y_educt: ', y[educt_ind])
            # print('raw_fwd_rate: ', raw_fwd_rate)
            # the raw backward rate of the reaction rct
            # raw_bwd_rate = k_bwd * [C]**nu_C * [D]**nu_D * ...
            raw_bwd_rate = rct['bwd_rate']
            for n_product, product in enumerate(products):
                product_ind = list_product_indices[n_product]
                coeff_product = rct['coeff_products'][n_product]
                raw_bwd_rate *= y[product_ind]**coeff_product
            # print('raw_bwd_rate: ', raw_bwd_rate)
            # apply raw forward and backward rates
            # to all variable educts and products:
            for n_educt, educt in enumerate(educts):
                if (educt in var_species):
                    educt_ind = list_educt_indices[n_educt]
                    coeff_educt = rct['coeff_educts'][n_educt]
                    dydt[educt_ind] -= coeff_educt * raw_fwd_rate
                    dydt[educt_ind] += coeff_educt * raw_bwd_rate
                    # print(educt, 'fwd_rate: ', -coeff_educt * raw_fwd_rate)
                    # print(educt, 'bwd_rate: ', +coeff_educt * raw_bwd_rate)
            for n_product, product in enumerate(products):
                if (product in var_species):
                    # print(product)
                    product_ind = list_product_indices[n_product]
                    coeff_product = rct['coeff_products'][n_product]
                    dydt[product_ind] += coeff_product * raw_fwd_rate
                    dydt[product_ind] -= coeff_product * raw_bwd_rate
                    # print(product, 'fwd_rate: ', +coeff_educt * raw_fwd_rate)
                    # print(product, 'bwd_rate: ', -coeff_educt * raw_bwd_rate)
        return dydt
