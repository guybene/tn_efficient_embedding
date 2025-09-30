from matplotlib.pyplot import legend

from embeddings.sym_gaussian_embedding import SymmetricMpoEmbedding
from tensor_network.sym_tensor_network import SymmetricalTensorNetwork

from tensornetwork import Node
import tensornetwork as tn

import time

import numpy as np
import string
import matplotlib.pyplot as plt

def calc_trace(W):
    nodes = []
    for data in W:
        v = Node(data)
        v[0] ^ v[-1]
        v = tn.contract_between(v,v)
        nodes.append(v)
    for i in range(len(W)-1):
        u = nodes[i]
        v = nodes[i+1]
        u[-1] ^ v[0]
        nodes[i+1 ] = tn.contract_between(u,v)
    ans = nodes[len(W) - 1].tensor
    return ans


def mps_to_symmetric_mpo(A, kind: str = "hermitian"):
    """
    A[i]: (d, Dl, Dr)  ->  W[i]: (d, Dl, Dr, d)
    - 'hermitian':  W[s,a,b,t] = A[s,a,b] * conj(A[t,a,b])   (Hermitian on physical legs)
    - 'transpose':  0.5*(W + W^T_phys) to enforce matrix-transpose symmetry on (s,t)
    """
    W = []
    for Ai in A:
        d, Dl, Dr = Ai.shape
        Wi = np.einsum('sab,tab->sabt', Ai, Ai.conj(), optimize=True)  # (s,a,b,t)
        if kind == "transpose":
            Wi = 0.5 * (Wi + Wi.transpose(3,1,2,0))                    # sym under s↔t
        elif kind != "hermitian":
            raise ValueError("kind must be 'hermitian' or 'transpose'")
        W.append(Wi)
    return W

def calc_noa_trace(W):
    nodes = [Node(W[i], name=names[i]) for i in range(len(W))]
    for i in range(len(W)-1):
        u = nodes[i]
        v = nodes[i+1]
        if i != 0:
            u[2] ^ v[1]
        else:
            u[1] ^ v[1]
    start = time.time()
    for i in range(len(nodes)):
        u = nodes[i]
        d = u.shape[0]
        data =  np.random.normal(0,1,d)
        random_vec = Node(data)
        random_vec_prime = Node(data)
        u[0] ^ random_vec[0]
        u[-1] ^ random_vec_prime[0]
        u = tn.contract_between(u, random_vec)
        u = tn.contract_between(u, random_vec_prime)
        nodes[i] = u
    for i in range(len(nodes)-1):
        u = nodes[i]
        v = nodes[i+1]
        nodes[i+1 ] = tn.contract_between(u,v)
    ans = nodes[-1].tensor
    time_cost = time.time() - start
    return ans, time_cost



if __name__ == "__main__":
    d = 50
    D = 20 # will have bond D^2
    M = 100
    m = 30
    N = 5

    rng = np.random.default_rng(10)
    names = string.ascii_uppercase

    A = [rng.standard_normal((d, D, D)) for _ in range(N)] # Create tensors

    W = mps_to_symmetric_mpo(A, kind="hermitian")  # each Wi: (d, Dl, Dr, d)
    W[0] = W[0][:,0,:,:]
    W[-1] = W[-1][:,0,:,:]

    actual_trace = calc_trace(W)

    edges =  [((i, 2), (i + 1, 1)) for i in range(1, N - 1)]
    edges = [((0, 1), (1, 1))] + edges


    symmetry = [((i,0), (i,3)) for i in range(1,N-1)]
    symmetry = [((0, 0), (0, 2))] + symmetry + [((N-1, 0), (N-1,2))]

    contraction_path = [(i, i+1) for i in range(N-1)]

    gaussian_error = []
    gaussian_times = []
    gaussian_ans = 0
    gaussian_time = 0

    noa_error = []
    noa_times = []
    noa_ans = 0
    noa_time = 0

    for i in range(1, M):
        if i % 10 == 0:
            print(i)
        noa_ans_curr, noa_curr_time = calc_noa_trace(W)

        noa_ans += noa_ans_curr
        noa_time += noa_curr_time

        noa_error.append(np.abs(noa_ans/i - actual_trace)/ actual_trace)
        noa_times.append(noa_time)

        nodes = [Node(W[i],name=names[i]) for i in range(len(W))]
        network = SymmetricalTensorNetwork(nodes, edges, symmetry)
        embedding = SymmetricMpoEmbedding(eps=0.1, delta=0.1, m_scalar=1)

        start = time.time()
        gaussian_embeding = embedding.embed_mpo(network, contraction_path, m_override=m)

        gaussian_curr_ans = np.trace(gaussian_embeding[0].tensor)
        gaussian_curr_time = time.time() - start

        gaussian_ans += gaussian_curr_ans
        gaussian_time += gaussian_curr_time

        gaussian_error.append(np.abs(gaussian_ans/i - actual_trace)/actual_trace)
        gaussian_times.append(gaussian_time)

    fig, ax = plt.subplots(2)
    ax[0].plot(np.arange(M-1),gaussian_error, label="Gaussian")
    ax[0].plot(np.arange(M-1),noa_error, label="Noa")
    ax[0].set_title(f"N_{N}_M_{M}_m_{m}_d_{d}_D_{D}\nIteration / Error")

    ax[1].plot(gaussian_times,gaussian_error, label="Gaussian")
    ax[1].plot(noa_times,noa_error, label="Noa")
    ax[1].set_title("Time / Error")

    fig.legend()
    plt.savefig(f"results_N_{N}_M_{M}_m_{m}_d_{d}_D_{D}.jpg")






