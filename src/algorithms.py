import numpy as np
import sketching

def unfold(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

def fold(unfolded_tensor, mode, shape):
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(np.reshape(unfolded_tensor, full_shape), 0, mode)

def mode_product(tensor, matrix, mode):
    if tensor.shape[mode] != matrix.shape[0]:
        raise RuntimeError(f'tensor.shape[{mode}] != matrix.shape[0]')
    new_shape = list(tensor.shape)
    new_shape[mode] = matrix.shape[1]
    tensor_mode = unfold(tensor, mode)
    return fold(matrix.T @ tensor_mode, mode, new_shape)


def HOSVD(tensor, ranks):
    S = tensor.copy()
    U_list = []
    for k in range(len(tensor.shape)):
        ak = unfold(S, k)
        u, s, vh = sketching.svdr(ak, ranks[k])
        vh = (s * vh.T).T
        shape = list(S.shape)
        shape[k] = ranks[k]
        S = fold(vh, k, shape)
        U_list.append(u)
    return S, U_list

def TTSVD(tensor, r):
    n = np.array(tensor.shape)
    G_list = []

    G = tensor.copy()
    G0 = unfold(G, 0)
    u, s, vh = sketching.svdr(G0, r[0])
    vh = (s * vh.T).T
    G_list.append(u)
    for k in range(1, len(tensor.shape)-1):
        vh = vh.reshape(r[k-1]*n[k], np.prod(n[k+1:]))
        u, s, vh = sketching.svdr(vh, r[k])
        vh = (s * vh.T).T
        r_cur = min(r[k], vh.shape[0])
        u = u.reshape(r[k-1], n[k], r_cur)
        G_list.append(u)
    G_list.append(vh)
    
    return G_list


def myHOSVD(tensor, ranks, svdr, iters_num=100, left=0, right=None, info=None): # alternating projections
    if info is not None:
        info.init('HOSVD', svdr.getName(), l=left, r=right)
    for i in range(iters_num):
        S = tensor.copy()
        U_list = []
        S[S < left] = left
        if right is not None:
            S[S > right] = right
        for k in range(len(tensor.shape)):
            Sk = unfold(S, k)
            Ur, Vhr = svdr(Sk, ranks[k])
            shape = list(S.shape)
            shape[k] = ranks[k]
            S = fold(Vhr, k, shape)
            U_list.append(Ur)
        tensor = restoreTensor_hosvd(S, U_list)
        if info is not None:
            info.update(tensor)
    return S, U_list

def myTTSVD(tensor, ranks, svdr, iters_num=100, left=0, right=None, info=None): # alternating projections
    if info is not None:
        info.init('TT-SVD', svdr.getName(), l=left, r=right)
    n = np.array(tensor.shape)
    for i in range(iters_num):
        G_list = []
        G = tensor.copy()
        G[G < left] = left
        if right is not None:
            G[G > right] = right
        G0 = unfold(G, 0)
        Ur, Vhr = svdr(G0, ranks[0])
        G_list.append(Ur)
        for k in range(1, len(tensor.shape)-1):
            Vhr = Vhr.reshape(ranks[k-1]*n[k], np.prod(n[k+1:]))
            Ur, Vhr = svdr(Vhr, ranks[k])
            r_cur = min(ranks[k], Vhr.shape[0])
            Ur = Ur.reshape(ranks[k-1], n[k], r_cur)
            G_list.append(Ur)
        G_list.append(Vhr)
        tensor = restoreTensor_ttsvd(G_list)
        if info is not None:
            info.update(tensor)
    return G_list   


def restoreTensor_hosvd(s, u_list):
    res = mode_product(s, u_list[0].T, 0)
    for k, u in enumerate(u_list[1:], 1):
        res = mode_product(res, u.T, k)
    return res

def restoreTensor_ttsvd(g_list):
    res = mode_product(g_list[1], g_list[0].T, 0)
    for k in range(2, len(g_list)-1):
        g = g_list[k]
        new_shape = list(res.shape[:-1]) + list(g.shape[1:])
        g = unfold(g, 0)
        res = mode_product(res, g, k)
        res = res.reshape(new_shape)
    res = mode_product(res, g_list[-1], len(res.shape)-1)
    return res