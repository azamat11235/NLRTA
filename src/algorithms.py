import numpy as np
from  truncatedSvd import TruncatedSvd


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


def STHOSVD(tensor, ranks, truncatedSvd=TruncatedSvd('SVDr')):
    S = tensor.copy()
    U_list = []
    for k in range(len(tensor.shape)):
        ak = unfold(S, k)
        u, vh = truncatedSvd(ak, ranks[k])
        shape = list(S.shape)
        shape[k] = ranks[k]
        S = fold(vh, k, shape)
        U_list.append(u)
    return S, U_list

def TTSVD(tensor, r, truncatedSvd=TruncatedSvd('SVDr')):
    n = np.array(tensor.shape)
    G_list = []

    G = tensor.copy()
    G0 = unfold(G, 0)
    u, vh = truncatedSvd(G0, r[0])
    G_list.append(u)
    for k in range(1, len(tensor.shape)-1):
        vh = vh.reshape(r[k-1]*n[k], np.prod(n[k+1:]))
        u, vh = truncatedSvd(vh, r[k])
        r_cur = min(r[k], vh.shape[0])
        u = u.reshape(r[k-1], n[k], r_cur)
        G_list.append(u)
    G_list.append(vh)
    
    return G_list


def NLRT(tensor, ranks, truncatedSvd, itersNum, info=None):
    if info is not None:
        info.init('NLRT', truncatedSvd.getName())
    tensor = tensor.copy()
    # for _ in range(itersNum):
    #     tensor[tensor < 0] = 0
    #     S, U_list = STHOSVD(tensor, ranks, truncatedSvd)
    #     tensor = restoreTensor_hosvd(S, U_list)
    #     if info is not None:
    #         info.update(tensor)
    return# S, U_list

def NSTHOSVD(tensor, ranks, truncatedSvd, itersNum, info=None): # alternating projections
    if info is not None:
        info.init('NSTHOSVD', truncatedSvd.getName())
    tensor = tensor.copy()
    for _ in range(itersNum):
        tensor[tensor < 0] = 0
        S, U_list = STHOSVD(tensor, ranks, truncatedSvd)
        tensor = restoreTensorTucker(S, U_list)
        if info is not None:
            info.update(tensor)
    return S, U_list

def NTTSVD(tensor, ranks, truncatedSvd, itersNum, info=None): # alternating projections
    if info is not None:
        info.init('NTTSVD', truncatedSvd.getName())
    tensor = tensor.copy()
    for _ in range(itersNum):
        tensor[tensor < 0] = 0
        G_list = TTSVD(tensor, ranks, truncatedSvd)
        tensor = restoreTensorTT(G_list)
        if info is not None:
            info.update(tensor)
    return G_list


def restoreTensorTucker(s, u_list):
    res = mode_product(s, u_list[0].T, 0)
    for k, u in enumerate(u_list[1:], 1):
        res = mode_product(res, u.T, k)
    return res

def restoreTensorTT(g_list):
    res = mode_product(g_list[1], g_list[0].T, 0)
    for k in range(2, len(g_list)-1):
        g = g_list[k]
        new_shape = list(res.shape[:-1]) + list(g.shape[1:])
        g = unfold(g, 0)
        res = mode_product(res, g, k)
        res = res.reshape(new_shape)
    res = mode_product(res, g_list[-1], len(res.shape)-1)
    return res
