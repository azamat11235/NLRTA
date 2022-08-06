import numpy as np


class Info:
    def __init__(self):
        self._info = None
        
    def init(self, decomp, _truncatedSvd, l=0, r=None):
        self._decomp = decomp
        self._truncatedSvd = _truncatedSvd
        self._info = {'frobenius': [], 'chebyshev': [], 'density': []}
  
    def clear(self):
        self._info = {'frobenius': [], 'chebyshev': [], 'density': []}

    def update(self, a):
        self._info['frobenius'].append(np.linalg.norm(a[a < 0]))
        neg_count = (a < 0).sum()
        self._info['density'].append(neg_count/(np.prod(a.shape)))
        self._info['chebyshev'].append(np.max(abs(a[a < 0]), initial=0))
        
    def getConvergenceInfo(self):
        return self._info
    
    def getTruncatedSvdName(self):
        return self._truncatedSvd
    
    def getDecompName(self):
        return self._decomp
    
    def getAlgName(self):
        return ', '.join([self._decomp, self._truncatedSvd])