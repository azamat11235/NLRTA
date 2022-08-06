import numpy as np
import scipy as sp


def TestMatrix(m, n, distribution='rademacher', p=None):
    if distribution == 'normal':
        res = np.random.normal(size=(m, n))
    elif distribution == 'rademacher' and p is None:
        res = np.random.choice([-1,1], size=(m,n))
    elif distribution == 'rademacher':
        res = np.random.choice([0,1,-1], size=(m,n), p=[1-p,p/2,p/2])
    else:
        raise TypeError('Invalid arguments')
    return res

def svdr(a, r):
    u, s, vh = sp.linalg.svd(a, full_matrices=False)
    return u[:, :r], s[:r], vh[:r, :]

def SVDr(X, r):
    Ur, Sr, Vhr = svdr(X, r)
    return Ur, Sr, Vhr

def HMT(X, rank, p, k, distr='rademacher', rho=None):
    n = X.shape[1]

    Psi = TestMatrix(n, k, distr, rho)
    Z1 = X @ Psi
    Q, _ = np.linalg.qr(Z1)
    for _ in range(p):
        Z2 = Q.T @ X
        Q, _ = np.linalg.qr(Z2.T)
        Z1 = X @ Q
        Q, _ = np.linalg.qr(Z1)
    Z2 = Q.T @ X
    Ur, Sr, Vhr = svdr(Z2, rank)
    Ur = Q @ Ur
    
    return Ur, Sr, Vhr

def Tropp(X, rank, k, l, distr='rademacher', rho=None):
    m, n = X.shape
        
    Psi = TestMatrix(n, k, distr, rho)
    Phi = TestMatrix(l, m, distr, rho)
    Z = X @ Psi
    Q, R = np.linalg.qr(Z)
    W = Phi @ Q
    P, T = np.linalg.qr(W)
    G = np.linalg.inv(T) @ P.T @ Phi @ X
    Ur, Sr, Vhr = svdr(G, rank)
    Ur = Q @ Ur
    
    return Ur, Sr, Vhr

def GN(X, rank, l, distr='rademacher', rho=None):
    m, n = X.shape

    Psi = TestMatrix(n, rank, distr, rho)
    Phi = TestMatrix(l, m, distr, rho)
    Z = X @ Psi
    W = Phi @ Z
    Q, R = np.linalg.qr(W)
    V = (Phi@X).T @ Q
    U = Z @ np.linalg.inv(R)
        
    return U, V.T


def getAlgName(alg, **params):
    if alg == 'Tangent':
        return alg
    if alg == 'SVDr':
        return 'SVD$_r$'
    distr = params.get('distr')
    d = 'N' if distr=='normal' else 'Rad'
    if 'rho' in params.keys():
        rho = params.pop('rho')
    else:
        rho = None
    if alg == 'GN':
        p = list(params.values())[0]
        alg_name = f'{alg}({p}), {d}'
    else:
        if alg == 'HMT':
                p0 = params['p']
                p1 = params['k']
        elif alg == 'Tropp':
            p0 = params['k']
            p1 = params['l']
        alg_name = f'{alg}({p0},{p1}), {d}'
    if d == 'N':
        alg_name += '(0,1)'
    elif rho is not None:
        alg_name += f'({rho})'
    return alg_name

class TruncatedSvd:
    def __init__(self, alg, **params):
        alg = alg.lower()
        if alg in {'svd', 'svdr'}:
            self._truncatedSvd = SVDr
        elif alg == 'hmt':
            self._truncatedSvd = HMT
        elif alg == 'tropp':
            self._truncatedSvd = Tropp
        elif alg == 'gn':
            self._truncatedSvd = GN
        else:
            raise
        self._params = params

    def __call__(self, a, ranks):
        if self._params:
            if self._truncatedSvd.__name__ == 'GN':
                Ur, Vhr = self._truncatedSvd(a, ranks, **self._params)
            else:
                Ur, Sr, Vhr = self._truncatedSvd(a, ranks, **self._params)
                Vhr = (Sr * Vhr.T).T
            return Ur, Vhr 
        else:
            Ur, Sr, Vhr = self._truncatedSvd(a, ranks)
            return Ur, (Sr * Vhr.T).T
    
    def setParams(self, params):
        self._params = params

    def getParams(self):
        return self._params
    
    def getName(self):
        return getAlgName(self._truncatedSvd.__name__, **self._params)
