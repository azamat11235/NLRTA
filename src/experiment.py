from re import L
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from time import time
from math import ceil

import algorithms

class Experiment:
    def __init__(self, tensor, ttSvdrList=[], hoSvdrList=[], hoRanks=[], ttRanks=[]):
        self._tensor = tensor
        self._itersNum = None

        self._initialTTSVD = None
        self._ttApproximations = []
        self._ttInfo = []
        self._ttTimes = []
        self._ttRanks = ttRanks
        self._ttSvdrList = ttSvdrList

        self._initialHOSVD = None
        self._hoApproximations = []
        self._hoInfo = []
        self._hoTimes = []
        self._hoRanks = hoRanks
        self._hoSvdrList = hoSvdrList

    def run(self, ttsvd=True, hosvd=True, itersNum=100, convergenceInfo=True, saveRuntimes=False, verbose=True):
        if self._ttSvdrList is None and self._hoSvdrList is None:
            print('svdrList is None. Use obj.setSvdrList(svdrList).')
            return
        self._itersNum = itersNum
        line = '-' * 36

        algNames = []
        infoList = []
        times = []
        svdrList = []
        if ttsvd and self._ttRanks is not None:
            algNames.append('TT-SVD')
            self._ttApproximations = []
            self._ttInfo = []
            infoList.append(self._ttInfo)
            if saveRuntimes: self._ttTimes = []
            times.append(self._ttTimes)
            svdrList.append(self._ttSvdrList)
        if hosvd and self._hoRanks is not None:
            algNames.append('HOSVD')
            self._hoApproximations = []
            self._hoInfo = []
            infoList.append(self._hoInfo)
            if saveRuntimes: self._hoTimes = []
            times.append(self._hoTimes)
            svdrList.append(self._hoSvdrList)
        
        for k in range(len(algNames)):
            algName = algNames[k]
            if verbose:
                print(algName)
                print(line)
            for i in range(len(svdrList[k])):
                info = Info() if convergenceInfo else None
                t0 = time()
                if algName == 'TT-SVD':
                    g_list = algorithms.myTTSVD(self._tensor, self._ttRanks, svdrList[k][i], iters_num=itersNum, info=info)
                    t1 = time()
                    self._ttApproximations.append(algorithms.restoreTensor_ttsvd(g_list))
                else: # HOSVD
                    g, u_list = algorithms.myHOSVD(self._tensor, self._hoRanks, svdrList[k][i], iters_num=itersNum, info=info)
                    t1 = time()
                    self._hoApproximations.append(algorithms.restoreTensor_hosvd(g, u_list))
                infoList[k].append(info)
                if saveRuntimes: times[k].append(t1 - t0)
                if verbose:
                    print('%-24s | %6.2f s.' % (svdrList[k][i].getName(), t1-t0))
            if verbose: print(line)
    
    def timeit(self, ttsvd=True, hosvd=True, itersNum=100, verbose=True):
        if self._ttSvdrList is None and self._hoSvdrList is None:
            print('svdrList is None. Use obj.setSvdrList(svdrList).')
            return
        self._itersNum = itersNum
        line = '-' * 36

        algNames = []
        times = []
        algs = []
        ranksList = []
        svdrList = []
        if ttsvd and self._ttRanks is not None:
            algNames.append('TT-SVD')
            self._ttTimes = []
            times.append(self._ttTimes)
            algs.append(algorithms.myTTSVD)
            ranksList.append(self._ttRanks)
            svdrList.append(self._ttSvdrList)
        if hosvd and self._hoRanks is not None:
            algNames.append('HOSVD')
            self._hoTimes = []
            times.append(self._hoTimes)
            algs.append(algorithms.myTTSVD)
            ranksList.append(self._hoRanks)
            svdrList.append(self._hoSvdrList)
        
        for k in range(len(algNames)):
            algName = algNames[k]
            alg = algs[k]
            ranks = ranksList[k]
            if verbose:
                print(algName)
                print(line)
            for i in range(len(svdrList[k])):
                t0 = time()
                alg(self._tensor, ranks, svdrList[k][i], iters_num=itersNum)
                t1 = time()
                times[k].append(t1 - t0)
                if verbose:
                    print('%-23s | %6.2f s.' % (svdrList[k][i].getName(), t1-t0))
            if verbose: print(line)

    def runInitialSvd(self, ttsvd=True, hosvd=True, verbose=True):
        algNames = []
        ranks = []
        approximations = []
        times = []
        if ttsvd and self._ttRanks is not None:
            algNames.append('Initial TT-SVD')
            t0 = time()
            g_list = algorithms.TTSVD(self._tensor, self._ttRanks)
            times.append(time() - t0)
            self._initialTTSVD = algorithms.restoreTensor_ttsvd(g_list)
            approximations.append(self._initialTTSVD)
            ranks.append(self._ttRanks)
        if hosvd and self._hoRanks is not None:
            algNames.append('Initial HOSVD')
            t0 = time()
            g, u_list = algorithms.HOSVD(self._tensor, self._hoRanks)
            times.append(time() - t0)
            self._initialHOSVD = algorithms.restoreTensor_hosvd(g, u_list)
            approximations.append(self._initialHOSVD)
            ranks.append(self._hoRanks)
        if verbose:
            line = '-' * 40
            for i in range(len(algNames)):
                algName = algNames[i]
                a = self._tensor
                ar = approximations[i]
                if verbose:
                    print(algName)
                    print(line)
                print('%-28s | %9.5f' % ('time (s.)', times[i]))
                print('%-28s | %9.5f' % ('negative elements (fro)', np.linalg.norm(ar[ar < 0])))
                print('%-28s | %9.5f' % ('negative elements (che)', np.max(abs(ar[ar < 0]), initial=0)))
                neg_count = (ar < 0).sum()
                print('%-28s | %9.5f' % ('negative elements (density)', neg_count/(np.prod(ar.shape))))
                print('%-28s | %9.5f' % ('relative error (fro)', np.linalg.norm(a - ar)/np.linalg.norm(a)))
                print('%-28s | %9.5f' % ('relative error (che)', np.max(abs(a - ar)/ np.max(abs(a)))))
                print('%-28s | %9.5f' % ('r2_score', r2_score(a.flatten(), ar.flatten())))
                print('%-28s | %9.2f' % ('compression', compute_compression(a.shape, ranks[i])))
                print(line)
    
    def printErrors(self, ttsvd=True, hosvd=True):
        algNames = []
        approximationsList = []
        svdrList = []
        if ttsvd and self._ttApproximations:
            algNames.append('TT-SVD')
            approximationsList.append(self._ttApproximations)
            svdrList.append(self._ttSvdrList)
        if hosvd and self._hoApproximations:
            algNames.append('HOSVD')
            approximationsList.append(self._hoApproximations)
            svdrList.append(self._hoSvdrList)

        line = '-' * 84
        tensorFro = np.linalg.norm(self._tensor)
        tensorChe = np.max(abs(self._tensor))

        for k in range(len(algNames)):
            print('| %-24s | %s (fro) | %s (che) | %s |' % (algNames[k], 'relative error', 'relative error', 'r2_score'))
            print(line)
            for i in range(len(approximationsList[k])):
                approximation = approximationsList[k][i]
                fro = np.linalg.norm(self._tensor - approximation) / tensorFro
                che = np.max(abs(self._tensor - approximation)) / tensorChe
                r2 = r2_score(self._tensor.flatten(), approximation.flatten())
                print('| %-24s | %20.5f | %20.5f | %8.6f |' % (svdrList[k][i].getName(), fro, che, r2))
            print(line)

    def plotConvergence(self, ttsvd=True, hosvd=True, figsize=(12, 32), yticks=None, wspace=0.2, hspace=0.4):
        titles = ['Distance to nonnegative tensors', 'Percent of negative elements']
        norms  = ['Chebyshev', 'Frobenius', 'Density']
        colors = ['C0', 'C1', 'C2']
        if yticks is None:
            yticks = [10**(-x) for x in range(-1, 6, 1)]

        infoList = []
        if ttsvd and self._ttInfo:
            infoList.append(self._ttInfo)
        if hosvd and self._hoInfo:
            infoList.append(self._hoInfo)

        if len(infoList) > 0:
            nrows = len(infoList[0]) * len(infoList)
            fig, ax = plt.subplots(nrows, 2, figsize=figsize)
            if nrows == 1: ax = [ax] #
            for k in range(len(infoList)):
                for kk in range(len(infoList[k])):
                    info = infoList[k][kk]
                    algName = info.getAlgName()
                    convInfo = info.getConvergenceInfo()
                    i = k*len(infoList[0]) + kk
                    percentOfNegElems = [100*x for x in convInfo['density']]
                    ax[i][0].plot(range(1, self._itersNum+1), convInfo['chebyshev'], colors[0], label=norms[0])
                    ax[i][0].plot(range(1, self._itersNum+1), convInfo['frobenius'], colors[1], label=norms[1])
                    ax[i][1].plot(range(1, self._itersNum+1), percentOfNegElems, colors[2], label=norms[2])
                    ax[i][0].set_yscale('log')
                    ax[i][0].set_yticks(yticks)
                    ax[i][0].legend()
                    for j in range(2):
                        ax[i][j].set_title('$\\bf{%s}$\n%s' % (algName, titles[j]))
                        ax[i][j].grid()
            if nrows == 1: ax = ax[0] #
            plt.subplots_adjust(wspace=wspace, hspace=hspace)
            return fig, ax
    
    def plotRuntimes(self, ttsvd=True, hosvd=True, figsize=(6,6), wspace=0.01, rotation=45):
        algNames = []
        times = []
        svdrList = []
        if hosvd and self._hoTimes:
            algNames.append('HOSVD')
            times.append(self._hoTimes)
            svdrList.append(self._hoSvdrList)
        if ttsvd and self._ttTimes:
            algNames.append('TT-SVD')
            times.append(self._ttTimes)
            svdrList.append(self._ttSvdrList)
        ncols = len(algNames)
        if ncols == 0: return
        fig, ax = plt.subplots(1, ncols, figsize=figsize)
        if ncols == 1: ax = [ax] #
        for i in range(len(algNames)):
            xticks_labels = ['\n'.join(sketching.getName().split(', ')) for sketching in svdrList[i]]
            ax[i].set_title(algNames[i])
            ax[i].set_ylabel('Running time (Second)')
            ax[i].tick_params(axis='both', direction='in', right=True, top=True)
            ax[i].tick_params(axis='x', rotation=rotation)
            ax[i].set_xticks(ticks=range(len(svdrList[i])))
            ax[i].set_xticklabels(labels=xticks_labels, ha='right')
            ax[i].bar(range(0, len(svdrList[i]), 1), times[i], edgecolor='black')
        if ncols == 1: ax = ax[0] #
        plt.subplots_adjust(wspace=wspace)
        return fig, ax
    
    def showApproximations(self, ttsvd=True, hosvd=True, band=2, ncols=3, figsize=(15,15), wspace=0.01, hspace=0.13, cmap=None): # for HSIs
        approximationsList = [self._tensor]
        titles = ['Original']
        if ttsvd:
            if self._initialTTSVD is not None:
                titles.append('Initial TT-SVD')
                approximationsList.append(self._initialTTSVD)
            if self._ttApproximations:
                titles += [svdr.getName() for svdr in self._ttSvdrList]
                approximationsList.extend(self._ttApproximations)
        if hosvd:
            if self._initialHOSVD is not None:
                titles.append('Initial HOSVD')
                approximationsList.append(self._initialHOSVD)
            if self._hoApproximations:
                titles += [svdr.getName() for svdr in self._hoSvdrList]
                approximationsList.extend(self._hoApproximations)

        if len(titles) > 1:
            m = ceil(len(titles) / ncols)
            n = min(ncols, len(titles))
            fig, ax = plt.subplots(m, n, figsize=(15, 15))
            if m == 1: ax = [ax] #
            for k in range(len(titles)):
                title = titles[k]
                imgr = approximationsList[k][band]
                i = k // ncols
                j = k % ncols
                ax[i][j].set_title('%s' % title)
                ax[i][j].imshow(imgr, cmap=cmap)
                ax[i][j].axis('off')
            plt.subplots_adjust(wspace=wspace, hspace=hspace)
            if m == 1: ax = ax[0] #
            return fig, ax

    def set_ttRanks(self, ttRanks):
        self._ttRanks = ttRanks
    
    def set_hoRanks(self, hoRanks):
        self._hoRanks = hoRanks
    
    def setRanks(self, ttRanks, hoRanks):
        self._ttRanks = ttRanks
        self._hoRanks = hoRanks

    def setTensor(self, tensor):
        self._tensor = tensor
    
    def setSvdrList(self, svdrList):
        self._ttSvdrList = svdrList
        self._hoSvdrList = svdrList
    
    def set_ttSvdrList(self, svdrList):
        self._ttSvdrList = svdrList

    def set_hoSvdrList(self, svdrList):
        self._hoSvdrList = svdrList
    
    def getRanks(self):
        return {'ttRankd': self._ttRanks, 'hoRanks': self._hoRanks}
      

class Info:
    def __init__(self):
        self._info_l = None
        self._info_r = None
        
    def init(self, decomp, sketching, l=0, r=None):
        self._decomp = decomp
        self._sketching = sketching
        self._info_l = {'frobenius': [], 'chebyshev': [], 'density': []}
        if r is not None:
            self._info_r = {'frobenius': [], 'chebyshev': [], 'density': []}
        self._l = l
        self._r = r
  
    def clear(self):
        self._info_l = {'frobenius': [], 'chebyshev': [], 'density': []}
        if self._r is not None:
            self._info_r = {'frobenius': [], 'chebyshev': [], 'density': []}

    def update(self, a):
        a = a - self._l
        self._info_l['frobenius'].append(np.linalg.norm(a[a < 0]))
        neg_count = (a < 0).sum()
        self._info_l['density'].append(neg_count/(np.prod(a.shape)))
        self._info_l['chebyshev'].append(np.max(abs(a[a < 0]), initial=0))
        if self._r is not None:
            a = -(a - self._r)
            self._info_r['frobenius'].append(np.linalg.norm(a[a < 0]))
            neg_count = (a < 0).sum()
            self._info_r['density'].append(neg_count/(np.prod(a.shape)))
            self._info_r['chebyshev'].append(np.max(abs(a[a < 0]), initial=0))
        
    def getConvergenceInfo(self):
        if self._r is not None:
            return self._info_l, self._info_r
        else:
            return self._info_l
    
    def getAlgName(self):
        return ','.join([self._decomp, self._sketching])


def compute_compression(shape, ranks):
    shape = np.array(shape)
    orig_size = np.prod(shape)
    if (len(shape) == len(ranks)): # HOSVD
        ranks = np.array(ranks)
        return orig_size / (np.prod(ranks) + (shape * ranks).sum())
    compr = shape * np.array([1] + list(ranks))
    compr *= np.array(list(ranks) + [1])
    return orig_size / compr.sum() # TT-SVD