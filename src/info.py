import numpy as np
import algorithms


class Info:
    def __init__(self):
        self._info = None
        
    def init(self, algName: str, _truncatedSvd: str, tuckerRank=None, ttRank=None):
        self._algName = algName
        self._truncatedSvd = _truncatedSvd
        self._tuckerRank = tuckerRank
        self._ttRank = ttRank
        self._info = {'frobenius': [], 'chebyshev': [], 'density': []}
        if algName == 'NLRT':
            self._X_sthosvd = None
            self._info = {}
  
    def clear(self):
        self._info = {'frobenius': [], 'chebyshev': [], 'density': []}
        if self._algName == 'NLRT':
            self._info = {}

    def update(self, a):
        if self._algName == 'NLRT': # a = {'X': [...], 'Yi': Yi}
            if not self._info:
                self._info['X'] = [{'frobenius': [], 'chebyshev': [], 'density': []} for _ in range(len(a['X']))]
                self._info['X_sthosvd'] = {'frobenius': [], 'chebyshev': [], 'density': []}
            for i, Zi in enumerate(a['X']):
                info = self._computeConvergenceInfo(Zi)
                self._info['X'][i]['frobenius'].append(info['frobenius'])
                self._info['X'][i]['chebyshev'].append(info['chebyshev'])
                self._info['X'][i]['density'].append(info['density'])
            g, u_list = algorithms.STHOSVD(a['Yi'], self._tuckerRank)
            X_sthosvd = algorithms.restoreTensorTucker(g, u_list)
            self._X_sthosvd = X_sthosvd 
            info = self._computeConvergenceInfo(X_sthosvd)
            self._info['X_sthosvd']['frobenius'].append(info['frobenius'])
            self._info['X_sthosvd']['chebyshev'].append(info['chebyshev'])
            self._info['X_sthosvd']['density'].append(info['density'])
       
        else:
            info = self._computeConvergenceInfo(a)
            self._info['frobenius'].append(info['frobenius'])
            self._info['chebyshev'].append(info['chebyshev'])
            self._info['density'].append(info['density'])
    
    def _computeConvergenceInfo(self, a):
        info = {}
        info['frobenius'] = np.linalg.norm(a[a < 0])
        info['chebyshev'] = np.max(abs(a[a < 0]), initial=0)
        info['density'] = (a < 0).sum() /(np.prod(a.shape))
        return info
    
    def getXsthosvd(self):
        if self._algName == 'NLRT':
            return self._X_sthosvd
        
    def getConvergenceInfo(self):
        return self._info
    
    def getTruncatedSvdName(self, testMatrixName=False):
        if not testMatrixName:
            return self._truncatedSvd.split(', ')[0]
        return self._truncatedSvd
    
    def getAlgName(self):
        return self._algName
    
    def getFullAlgName(self, testMatrixName=False):
        return ', '.join([self._algName, self.getTruncatedSvdName(testMatrixName=testMatrixName)])