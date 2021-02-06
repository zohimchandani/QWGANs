#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'jupyternotify')
from pytket import Circuit, Qubit, Bit, OpType
from pytket.utils.operators import QubitPauliOperator
from sympy import symbols
from openfermion import QubitOperator
from random import sample
import numpy as np
# from pytket.backends.projectq import ProjectQBackend
from pytket.backends.ibm import AerStateBackend, AerBackend, AerUnitaryBackend, IBMQBackend
from scipy.linalg import expm, sinm, cosm
from sympy.physics.quantum.dagger import Dagger
import functools
import operator
import itertools    
from openfermion import get_sparse_operator
from scipy.optimize import minimize, LinearConstraint, Bounds
import matplotlib.pyplot as plt


def fidelity(rsv, gsv): 
    overlap = np.vdot(rsv, gsv)
    return abs(overlap)**2

#constants 
n = 1 #number of qubits 
lamb = np.float(2)
s = np.exp(-1 / (2 * lamb)) - 1
cst1 = (s / 2 + 1) ** 2
cst2 = (s / 2) * (s / 2 + 1)
cst3 = (s / 2) ** 2


# In[ ]:


def real(n, weights):
    
    real_circ = Circuit()
    qubits = real_circ.add_q_register('q', n)

    real_circ.Rx(weights[0], qubits[0])
        
    real_circ.Ry(weights[1], qubits[0])
        
    real_circ.Rz(weights[2], qubits[0])
        

    # backend.compile_circuit(real_circ)
    # statevector = backend.get_state(real_circ)
    
    # return real_circ, statevector
    return real_circ


# In[ ]:


def generator(n, weights):
    
    generator_circ = Circuit()
    qubits = generator_circ.add_q_register('q', n)

    generator_circ.Rx(weights[0], qubits[0])

    generator_circ.Ry(weights[1], qubits[0])

    generator_circ.Rz(weights[2], qubits[0])
        
    # backend.compile_circuit(generator_circ)
    # statevector = backend.get_state(generator_circ)
    
    # return generator_circ, statevector
    return generator_circ



def generator_symbolic(n):
    
    generator_circ = Circuit()
    qubits = generator_circ.add_q_register('q', n)
    weight_symbols = symbols([f'theta_{i}' for i in range(3)])

    generator_circ.Rx(weight_symbols[0], qubits[0])

    generator_circ.Ry(weight_symbols[1], qubits[0])

    generator_circ.Rz(weight_symbols[2], qubits[0])
        
    # backend.compile_circuit(generator_circ)
    # statevector = backend.get_state(generator_circ)
    
    # return generator_circ, statevector
    return generator_circ,weight_symbols


# In[ ]:


def operator_inner(left, operator_matrix, right):
    return np.vdot(left, operator_matrix.dot(right))


# In[ ]:


class Discriminator:
    def __init__(self, init_weights):
        self.set_weights(init_weights)
    def set_weights(self, _init_weights):
        halfway = len(_init_weights)//2
        self.psi_weights = _init_weights[:halfway]
        self.phi_weights = _init_weights[halfway:]
        assert(len(self.phi_weights) == len(self.psi_weights))
        self.n_qubits = len(self.phi_weights)//4

    def make_operator(self, weights_list):
        iden =  weights_list[-1] * QubitOperator(" ")
        tuple_list = [(weight, tup[0], tup[1]) for weight, tup in zip(weights_list[:-1], itertools.product(['X', 'Y', 'Z'], range(self.n_qubits)))]
        measurements = functools.reduce(operator.add, (weight * QubitOperator(f'{a}{n}') for weight, a, n in tuple_list))

        return iden + measurements


    def calculate_loss(self, real_sv, gen_sv):
        # construct operators
        psi = self.make_operator(self.psi_weights)
        phi = self.make_operator(self.phi_weights)
        #convert phi and psi operators to matrix 
        psi_matrix = np.array(get_sparse_operator(psi).todense())
        phi_matrix = np.array(get_sparse_operator(phi).todense())
        
        #calculate expectation values 
        
        psi_exp = operator_inner(real_sv, psi_matrix, real_sv)
        phi_exp = operator_inner(gen_sv, phi_matrix, gen_sv)

        #calculate values for A and B which go into the calculation for the regterm
        A = expm(np.float(-1 / lamb) * phi_matrix)
        B = expm(np.float(1 / lamb) * psi_matrix)

        term1 = operator_inner(gen_sv, A, gen_sv)
        term2 = operator_inner(real_sv, B, real_sv)
        term3 = operator_inner(gen_sv, B, real_sv)
        term4 = operator_inner(real_sv, A, gen_sv)
        term5 = operator_inner(gen_sv, A, real_sv)
        term6 = operator_inner(real_sv, B, gen_sv)
        term7 = operator_inner(gen_sv, B, gen_sv)
        term8 = operator_inner(real_sv, A, real_sv)

        regterm = (lamb / np.e * (cst1 * term1 * term2 - cst2 * term3 * term4 - cst2 * term5 * term6 + cst3 * term7 * term8)).item()

        return np.real(psi_exp - phi_exp - regterm)


# In[ ]:



def make_disc_loss(real_state, gen_state):
    def disc_loss(disc_weights):
        disc = Discriminator(disc_weights)

        return -disc.calculate_loss(real_state, gen_state)
    return disc_loss

def make_gen_loss(base_circuit, symb_weights, real_state, backend, discriminator):
    
    def gen_loss(gen_weights):
        gen_circ = base_circuit.copy()
        gen_circ.symbol_substitution(dict(zip(symb_weights, gen_weights)))
        gen_state = backend.get_state(gen_circ)
        return discriminator.calculate_loss(real_state, gen_state)


    return gen_loss


# In[ ]:


get_ipython().run_cell_magic('time', '', '%%notify\n\nbackend = AerStateBackend()\ngen_circ_base, symbolic_weights = generator_symbolic(n)\nbackend.compile_circuit(gen_circ_base)\nnp.random.seed(3)\n\n\nfor r in range(1):\n    \n    #generate a random real state and a initial fake state and the disc params \n    real_weights = np.random.uniform(0,2,3) \n    init_gen_weights = np.random.uniform(0,2,3)\n    init_disc_weights = np.random.uniform(0,2,8)\n    \n    real_circ = real(n, real_weights)\n    backend.compile_circuit(real_circ)\n    real_sv = backend.get_state(real_circ)\n\n    gen_circ = gen_circ_base.copy()\n    gen_circ.symbol_substitution(dict(zip(symbolic_weights, init_gen_weights)))\n\n    curr_gen_sv = backend.get_state(gen_circ)\n    init_fid = fidelity(real_sv, curr_gen_sv)\n\n    fid = init_fid\n\n\n    # dloss = []\n    gloss = []\n    f = []\n    \n    # f.append(init_fid)\n    # dloss.append(disc_loss(init_disc_weights))\n    # gloss.append(gen_loss(init_gen_weights))\n\n    curr_gen_weights = init_gen_weights\n    curr_disc_weights = init_disc_weights\n    niter = 0 \n\n\n    while fid <0.99:\n\n        #calculate the updated gen circ and statevector \n        curr_gen_circ = gen_circ_base.copy()\n        curr_gen_circ.symbol_substitution(dict(zip(symbolic_weights, curr_gen_weights)))\n        curr_gen_sv = backend.get_state(curr_gen_circ)\n        fid = fidelity(real_sv, curr_gen_sv)\n\n        f.append(fid)\n\n        #maximise loss for disc\n        disc_loss = make_disc_loss(real_sv, curr_gen_sv)\n        gloss.append(-disc_loss(curr_disc_weights))\n\n        def disc_callback(x):\n            print("D train", -disc_loss(x))\n            return False\n        disc_result = minimize(disc_loss, curr_disc_weights, method=\'Powell\',bounds=Bounds(0,1),  options={ \'maxiter\': 10, \'ftol\':1e-5}, callback=disc_callback)\n        curr_disc_weights = disc_result.x\n        print(curr_disc_weights)\n        # dloss.append(float(disc_result.fun))\n\n        #minimise loss for gen\n        curr_disc = Discriminator(curr_disc_weights)\n        gen_loss = make_gen_loss(gen_circ_base, symbolic_weights, real_sv, backend, curr_disc)\n        def gen_callback(x):\n            print(x)\n            print("G train", gen_loss(x))\n            return False\n        gloss.append(gen_loss(curr_gen_weights))\n        print("start", curr_gen_weights)\n        print(disc_loss(curr_disc_weights), gen_loss(curr_gen_weights))\n        gen_result = minimize(gen_loss, curr_gen_weights, method=\'Powell\', bounds=Bounds(0,2.0), options={ \'maxiter\': 10, \'ftol\':1e-10}, callback=gen_callback )\n        curr_gen_weights = gen_result.x\n        \n        # gloss.append(float(gen_result.fun))\n        \n\n        niter += 1\n\n        if niter == 5:\n            break \n\n            \n    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))\n\n    y = np.real(f)\n    x =list(range(0, len(y)))\n\n    y1 = -np.array(gloss)\n    x1 =list(range(0, len(y1)))\n    \n    y2 = gloss\n    x2 =list(range(0, len(y2)))\n\n\n    ax1.plot(x,y)\n    ax2.plot(x1, y1, label = "disc loss")\n    ax2.plot(x2, y2,  label = "gen loss")\n    ax2.legend()\n\n\n    ax1.set_xlabel(\'While loop iterations\')\n    ax1.set_ylabel(\'Fidelity\')\n\n    ax2.set_xlabel(\'While loop iterations\')\n    ax2.set_ylabel(\'Wasserstein  Loss\')')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




