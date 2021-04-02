
import numpy as np
import networkx as nx
import pickle
from scipy.spatial.distance import *



def _load_network(filename, mtrx='adj'):
    print ("### Loading [%s]..." % (filename))
    if mtrx == 'adj':
        G = nx.read_edgelist(filename, edgetype=float, data=(('weight', float),))
        A = nx.adjacency_matrix(G)
        A = A.todense()
        A = np.squeeze(np.asarray(A))
        if A.min() < 0:
            print ("### Negative entries in the matrix are not allowed!")
            A[A < 0] = 0
            print ("### Matrix converted to nonnegative matrix.")
        if (A.T == A).all():
            pass
        else:
            print ("### Matrix not symmetric!")
            A = A + A.T
            print ("### Matrix converted to symmetric.")
    else:
        print ("### Wrong mtrx type. Possible: {'adj', 'inc'}.")
    A = A - np.diag(np.diag(A))
    A = A + np.diag(A.sum(axis=1) == 0)

    return A


def load_networks(filenames, mtrx='adj'):
    Nets = []
    for filename in filenames:
        Nets.append(_load_network(filename, mtrx))

    return Nets


def _net_normalize(X):
    """
    Normalizing networks according to node degrees.
    """
    if X.min() < 0:
        print ("### Negative entries in the matrix are not allowed!")
        X[X < 0] = 0
        print ("### Matrix converted to nonnegative matrix.")
    if (X.T == X).all():
        pass
    else:
        print ("### Matrix not symmetric.")
        X = X + X.T - np.diag(np.diag(X))
        print ("### Matrix converted to symmetric.")

    # normalizing the matrix
    deg = X.sum(axis=1).flatten()
    deg = np.divide(1., np.sqrt(deg))
    deg[np.isinf(deg)] = 0
    D = np.diag(deg)
    X = D.dot(X.dot(D))

    return X


def net_normalize(Net):
    """
    Normalize Nets or list of Nets.
    """
    if isinstance(Net, list):
        for i in range(len(Net)):
            Net[i] = _net_normalize(Net[i])
        print (Net.shape)
    else:
        Net = _net_normalize(Net)

    return Net


def _scaleSimMat(A):
    """Scale rows of similarity matrix"""
    #A = A - np.diag(np.diag(A))
    A = A + np.diag(A.sum(axis=0) == 0)
    col = A.sum(axis=0)
    A = A.astype(np.float)/col[:, None]

    return A

def PPMI_matrix(M):
    """ Compute Positive Pointwise Mutual Information Matrix"""
    M = _scaleSimMat(M)
    n = M.shape[0]
    col = np.asarray(M.sum(axis=0), dtype=float)
    col = col.reshape((1, n))
    row = np.asarray(M.sum(axis=1), dtype=float)
    row = row.reshape((n, 1))
    D = np.sum(col)
    np.seterr(all='ignore')
    PPMI = np.log(np.divide(D*M, np.dot(row, col)))
    PPMI[np.isnan(PPMI)] = 0
    PPMI[PPMI < 0] = 0
    return PPMI

def svds(nnode,Q):
    #Q = Q.detach().numpy()
    alpha = 1/nnode
    Q = np.log(Q+alpha) - np.log(alpha)
    Q = np.matmul(Q,Q.T)
    #Q[np.isnan(Q) | np.isinf(Q)] = 0.0
    U,S,V = np.linalg.svd(Q)
    S = np.diag(S)
    X = np.matmul(U,np.sqrt(np.sqrt(S)))
    return X

def compute_similarity(net):
    d2 = 1 - pdist(net, 'jaccard')
    sim = squareform(d2)
    sim = sim + np.eye(net.shape[0])
    sim[np.isnan(sim) | np.isinf(sim)] = 0
    return sim

def RWR(A, K=20, alpha=0.4):
    """Random Walk on graph"""
    # Random surfing
    n = A.shape[0]
    P0 = np.eye(n, dtype=float)
    P = P0.copy()
    M = np.zeros((n, n), dtype=float)
    for i in range(0, K):
        P = alpha*np.dot(P, A) + (1 - alpha)*P0
        M = M + P

    return P
def get_rwr(type):
    drug=['mat_drug_drug', 'mat_drug_disease', 'mat_drug_se', 'Similarity_Matrix_Drugs']
    #drug = ['mat_drug_drug']
    protein=['mat_protein_protein', 'mat_protein_disease', 'Similarity_Matrix_Proteins']
    #protein = ['mat_protein_protein', 'mat_protein_disease']
    result =[]
    if type == 'drug':
        for i in range(len(drug)):
            f = np.loadtxt('data/'+drug[i]+'.txt')
            f = compute_similarity(f)
            rwr = RWR(f)
            #r = PPMI_matrix(rwr)
            r = svds(708,rwr)
            if result == []:
                result = np.around(rwr,5)
            else:
                result = np.concatenate([result, np.around(rwr,5)], axis=1)
            #result.append(r)
    elif type== 'protein':
        for i in range(len(protein)):
            f = np.loadtxt('data/'+protein[i]+'.txt')
            f = compute_similarity(f)
            rwr = RWR(f)
            #r = PPMI_matrix(rwr)
            r = svds(1512,rwr)
            if result == []:
                result = np.around(rwr,5)
            else:
                result = np.concatenate([result, np.around(rwr,5)], axis=1)
            #result.append(r)
    return result


if __name__ == "__main__":
    drug= get_rwr('drug')
    np.savetxt('../feature/drug_200.txt',drug,fmt='%0.5f')
    protein = get_rwr('protein')
    protein = svds(1512,protein,400)
    np.savetxt('../feature/protein_400.txt',protein,fmt='%0.5f')
    print(drug)
