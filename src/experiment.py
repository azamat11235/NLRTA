import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from time import time, sleep
from math import ceil

import algorithms
from info import Info


class Experiment:
    def __init__(self,
                 tensor,
                 nttsvdTruncatedSvdList=None,
                 nsthosvdTruncatedSvdList=None,
                 nlrtTruncatedSvdList=None,
                 ttRank=None,
                 tuckerRank=None):
        self._tensor = tensor

        self._ttsvdApproximation = None
        self._nttsvdApproximations = None
        self._nttsvdInfo = []
        self._nttsvdTimes = []
        self._ttRank = ttRank
        self._nttsvdTruncatedSvdList = nttsvdTruncatedSvdList

        self._sthosvdApproximation = None
        self._nsthosvdApproximations = None
        self._nsthosvdInfo = []
        self._nsthosvdTimes = []
        self._tuckerRank = tuckerRank
        self._nsthosvdTruncatedSvdList = nsthosvdTruncatedSvdList

        self._nlrtApproximations = None # [{'X': [], 'X_sthosvd': X_sthosvd}, ... ]
        self._nlrtInfo = [] # [{'X': [info, ...], 'X_sthosvd': {...}} ...]
        self._nlrtTimes = []
        self._nlrtTruncatedSvdList = nlrtTruncatedSvdList

    def runInitialSvd(self, ttsvd=True, sthosvd=True, verbose=True):
        algNames = []
        ranks = []
        approximations = []
        times = []
        if ttsvd and self._ttRank:
            algNames.append('TTSVD')
            t0 = time()
            g_list = algorithms.TTSVD(self._tensor, self._ttRank)
            times.append(time() - t0)
            self._ttsvdApproximation = algorithms.restoreTensorTT(g_list)
            approximations.append(self._ttsvdApproximation)
            ranks.append(self._ttRank)
        if sthosvd and self._tuckerRank:
            algNames.append('STHOSVD')
            t0 = time()
            g, u_list = algorithms.STHOSVD(self._tensor, self._tuckerRank)
            times.append(time() - t0)
            self._sthosvdApproximation = algorithms.restoreTensorTucker(g, u_list)
            approximations.append(self._sthosvdApproximation)
            ranks.append(self._tuckerRank)
        if verbose:
            line = '-' * 43
            for i in range(len(algNames)):
                algName = algNames[i]
                a = self._tensor
                ar = approximations[i]
                neg_count = (ar < 0).sum()
                if verbose:
                    print(algName)
                    print(line)
                print('%-28s | %12.5f'  % ('time (s.)', times[i]))
                print('%-28s | %12f'    % ('negative elements (fro)', np.linalg.norm(ar[ar < 0])))
                print('%-28s | %12f'    % ('negative elements (che)', np.max(abs(ar[ar < 0]), initial=0)))
                print('%-28s | %12.8f'  % ('negative elements (%)', 100*neg_count/(np.prod(ar.shape))))
                print('%-28s | %12.10f' % ('relative error (fro)', np.linalg.norm(a - ar)/np.linalg.norm(a)))
                print('%-28s | %12.10f' % ('relative error (che)', np.max(abs(a - ar)/ np.max(abs(a)))))
                print('%-28s | %12.10f' % ('r2_score', r2_score(a.flatten(), ar.flatten())))
                print('%-28s | %12.2f'  % ('compression', computeCompression(a.shape, ranks[i])))
                print(line)

    def run(self, itersNum,
                  nttsvd=True,
                  nsthosvd=True,
                  nlrt=True,
                  convergenceInfo=True,
                  saveRuntimes=False,
                  verbose=True):
        line = '-' * 36
        algNames = []
        infoList = []
        times = []
        truncatedSvdList = []
        if nttsvd and self._ttRank and self._nttsvdTruncatedSvdList:
            algNames.append('NTTSVD')
            self._nttsvdApproximations = []
            self._nttsvdInfo = []
            infoList.append(self._nttsvdInfo)
            if saveRuntimes: self._nttsvdTimes = []
            times.append(self._nttsvdTimes)
            truncatedSvdList.append(self._nttsvdTruncatedSvdList)
        if nsthosvd and self._tuckerRank and self._nsthosvdTruncatedSvdList:
            algNames.append('NSTHOSVD')
            self._nsthosvdApproximations = []
            self._nsthosvdInfo = []
            infoList.append(self._nsthosvdInfo)
            if saveRuntimes: self._nsthosvdTimes = []
            times.append(self._nsthosvdTimes)
            truncatedSvdList.append(self._nsthosvdTruncatedSvdList)
        if nlrt and self._tuckerRank and self._nlrtTruncatedSvdList:
            algNames.append('NLRT')
            self._nlrtApproximations = []
            self._nlrtInfo = []
            infoList.append(self._nlrtInfo)
            if saveRuntimes: self._nlrtTimes = []
            times.append(self._nlrtTimes)
            truncatedSvdList.append(self._nlrtTruncatedSvdList)
        
        for k in range(len(algNames)):
            algName = algNames[k]
            if verbose:
                print(algName)
                print(line)
            for i in range(len(truncatedSvdList[k])):
                info = Info() if convergenceInfo else None
                t0 = time()
                if algName == 'NTTSVD':
                    g_list = algorithms.NTTSVD(self._tensor, self._ttRank, truncatedSvdList[k][i], itersNum=itersNum, info=info)
                    t1 = time()
                    self._nttsvdApproximations.append(algorithms.restoreTensorTT(g_list))
                elif algName == 'NSTHOSVD':
                    g, u_list = algorithms.NSTHOSVD(self._tensor, self._tuckerRank, truncatedSvdList[k][i], itersNum=itersNum, info=info)
                    t1 = time()
                    self._nsthosvdApproximations.append(algorithms.restoreTensorTucker(g, u_list))
                else: # NLRT
                    X = algorithms.NLRT(self._tensor, self._tuckerRank, truncatedSvdList[k][i], itersNum=itersNum, info=info)
                    t1 = time()
                    self._nlrtApproximations.append({'X': X, 'X_sthosvd': info.getXsthosvd()})
                infoList[k].append(info)
                if saveRuntimes: times[k].append(t1 - t0)
                if verbose:
                    print('%-24s | %6.2f s.' % (truncatedSvdList[k][i].getName(), t1-t0))
            if verbose: print(line)
    
    def time(self, itersNum,
                   nttsvd=True,
                   nsthosvd=True,
                   nlrt=True,
                   verbose=True,
                   _sleep=10):

        line = '-' * 36
        algNames = []
        times = []
        algs = []
        ranksList = []
        truncatedSvdList = []
        if nttsvd and self._ttRank and self._nttsvdTruncatedSvdList:
            algNames.append('NTTSVD')
            self._nttsvdTimes = []
            times.append(self._nttsvdTimes)
            algs.append(algorithms.NTTSVD)
            ranksList.append(self._ttRank)
            truncatedSvdList.append(self._nttsvdTruncatedSvdList)
        if nsthosvd and self._tuckerRank and self._nsthosvdTruncatedSvdList:
            algNames.append('NSTHOSVD')
            self._nsthosvdTimes = []
            times.append(self._nsthosvdTimes)
            algs.append(algorithms.NSTHOSVD)
            ranksList.append(self._tuckerRank)
            truncatedSvdList.append(self._nsthosvdTruncatedSvdList)
        if nlrt and self._tuckerRank and self._nlrtTruncatedSvdList:
            algNames.append('NLRT')
            self._nlrtTimes = []
            times.append(self._nlrtTimes)
            algs.append(algorithms.NLRT)
            ranksList.append(self._tuckerRank)
            truncatedSvdList.append(self._nlrtTruncatedSvdList)
        
        for k in range(len(algNames)):
            algName = algNames[k]
            alg = algs[k]
            ranks = ranksList[k]
            if verbose:
                print(algName)
                print(line)
            for i in range(len(truncatedSvdList[k])):
                t0 = time()
                alg(self._tensor, ranks, truncatedSvdList[k][i], itersNum=itersNum)
                t1 = time()
                times[k].append(t1 - t0)
                if verbose:
                    print('%-23s | %6.2f s.' % (truncatedSvdList[k][i].getName(), t1-t0))
                if _sleep and not (k == len(algNames) - 1 and i == len(truncatedSvdList[k]) - 1):
                    sleep(_sleep)
            if verbose: print(line)
    
    def printErrors(self, nttsvd=True, nsthosvd=True, nlrt=True):
        algNames = []
        approximationsList = []
        truncatedSvdList = []
        if nttsvd and self._nttsvdApproximations:
            algNames.append('NTTSVD')
            approximationsList.append(self._nttsvdApproximations)
            truncatedSvdList.append(self._nttsvdTruncatedSvdList)
        if nsthosvd and self._nsthosvdApproximations:
            algNames.append('NSTHOSVD')
            approximationsList.append(self._nsthosvdApproximations)
            truncatedSvdList.append(self._nsthosvdTruncatedSvdList)
        if nlrt and self._nlrtApproximations:
            algNames.append('NLRT')

        line = '-' * 85
        tensorFro = np.linalg.norm(self._tensor)
        tensorChe = np.max(abs(self._tensor))

        for k, algName in enumerate(algNames):
            if algName == 'NLRT':
                line = '-' * 97
                if k: print('\n', line, sep='')
                print('| %-24s | relative error (fro) | relative error (che) | r2_score | %9s |' % (algName, 'X_i'))
                print(line)
                for j in range(len(self._nlrtApproximations)):
                    X = self._nlrtApproximations[j]['X']
                    X_sthosvd = self._nlrtApproximations[j]['X_sthosvd']
                    for i, x in enumerate(X + [X_sthosvd]):
                        fro = np.linalg.norm(self._tensor - x) / tensorFro
                        che = np.max(abs(self._tensor - x)) / tensorChe
                        r2 = r2_score(self._tensor.flatten(), x.flatten())
                        truncatedSvdName = ' ' * 24
                        if i == 0:
                            truncatedSvdName = self._nlrtTruncatedSvdList[j].getName()
                        tensorName = 'X_%d' % i
                        if i == len(X):
                            tensorName = 'X_sthosvd'
                        print('| %-24s | %2.18f | %20.18f | %8f | %9s |' % (truncatedSvdName, fro, che, r2, tensorName))
                    print(line)
            else:
                print('| %-24s | relative error (fro) | relative error (che) | r2_score |' % algName)
                print(line)
                for i in range(len(approximationsList[k])):
                    approximation = approximationsList[k][i]
                    fro = np.linalg.norm(self._tensor - approximation) / tensorFro
                    che = np.max(abs(self._tensor - approximation)) / tensorChe
                    r2 = r2_score(self._tensor.flatten(), approximation.flatten())
                    print('| %-24s | %2.18f | %20.18f | %8f |' % (truncatedSvdList[k][i].getName(), fro, che, r2))
                print(line)
    
    def printNegativePart(self, nttsvd=True, nsthosvd=True, nlrt=True):
        algNames = []
        infoList = []
        if nttsvd and self._nttsvdInfo:
            algNames.append('NTTSVD')
            infoList.append(self._nttsvdInfo)
        if nsthosvd and self._nsthosvdInfo:
            algNames.append('NSTHOSVD')
            infoList.append(self._nsthosvdInfo)
        if nlrt and self._nlrtInfo:
            algNames.append('NLRT')
            infoList.append(self._nlrtInfo)

        line = '-' * 52

        for k, algName in enumerate(algNames):
            if algName == 'NLRT':
                line = '-' * 64
                if k: print('\n', line, sep='')
                print('| %-24s | negative elements (%%) | %9s |' % (algName, 'X_i'))
            else:
                print('| %-24s | negative elements (%%) |' % algName)
            print(line)
            for i in range(len(infoList[k])):
                truncatedSvdName = infoList[k][i].getTruncatedSvdName()
                if algName == 'NLRT':
                    convInfo = infoList[k][i].getConvergenceInfo()
                    convInfo = convInfo['X'] + [convInfo['X_sthosvd']]
                    for i, info in enumerate(convInfo):
                        percentOfNegative = info['density'][-1] * 100
                        tensorName = 'X_%d' % i
                        if i == len(convInfo) - 1:
                            tensorName = 'X_sthosvd'
                        print('| %-24s | %21.7f | %9s |' % (truncatedSvdName, percentOfNegative, tensorName))
                        if i == 0:
                            truncatedSvdName = ' ' * 24
                    print(line)
                else:
                    percentOfNegative = infoList[k][i].getConvergenceInfo()['density'][-1] * 100
                    print('| %-24s | %21.7f |' % (truncatedSvdName, percentOfNegative))
            if not nlrt: print(line)
    
    def plotConvergence(self,
                        nttsvd=True,
                        nsthosvd=True,
                        nlrt=True,
                        figsize=None,
                        yticks=None,
                        wspace=None, hspace=None,
                        titlesize=None,
                        ticksize=None,
                        legendsize=None,
                        legendloc=None,
                        testMatrixName=False,):
        title = 'Distance to nonnegative tensors'
        norms  = ['Chebyshev', 'Frobenius', 'Density']
        colors = ['C0', 'C1', 'C2']
        infoList = []
        step = 0
        if nsthosvd and self._nsthosvdInfo:
            infoList.extend(self._nsthosvdInfo)
            step += 1
        if nttsvd and self._nttsvdInfo:
            infoList.extend(self._nttsvdInfo)
            step += 1
        if nlrt and self._nlrtInfo:
            infoList.extend(self._nlrtInfo)

        if len(infoList) > 0:
            nInfo = len(infoList)
            nrows = nInfo // 2 + (nInfo % 2 != 0)
            ncols = 2 if nInfo > 1 else 1
            fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
            if nInfo <= 2: ax = [ax] #
            if nInfo <= 1: ax = [ax] #
            if len(self._nttsvdInfo) != len(self._nsthosvdInfo) or nlrt:
                step = 1
            if step == 2:
                nInfo //= 2
            for k in range(nInfo):
                for kk in range(step):
                    i = k * step // 2
                    j = (k * step + kk)  % 2
                    info = infoList[k + kk * nInfo]
                    algName = info.getFullAlgName() if testMatrixName else ',\\ '.join(info.getFullAlgName().split(', ')[:2])
                    info = info.getConvergenceInfo()
                    if 'NLRT' in algName:
                        for m, xInfo in enumerate(info['X']):
                            itersNum = len(xInfo['frobenius'])
                            ax[i][j].plot(range(1, itersNum+1), xInfo['frobenius'], f'C{m}', label='$\\bf{X}$$^{(i)}_%d$' % m)
                        itersNum = len(info['X_sthosvd']['frobenius'])
                        ax[i][j].plot(range(1, itersNum+1), info['X_sthosvd']['frobenius'], f'C{m+1}', label='$\\bf{X}$$^{(i)}_{NLRT+STHOSVD}$')
                    else:
                        itersNum = len(info['frobenius'])
                        ax[i][j].plot(range(1, itersNum+1), info['chebyshev'], colors[0], label=norms[0])
                        ax[i][j].plot(range(1, itersNum+1), info['frobenius'], colors[1], label=norms[1])
                        ax[i][j].plot(range(1, itersNum+1), info['density'],   colors[2], label=norms[2])
                    ax[i][j].set_yscale('log')
                    if yticks is not None:
                        ax[i][j].set_yticks(yticks)
                    ax[i][j].legend(loc=legendloc, fontsize=legendsize)
                    ax[i][j].tick_params(axis='both', labelsize=ticksize)
                    ax[i][j].grid()
                    if 'SVD$_r$' in algName:
                        title_ = '$\\bf{%s,\\ SVD_r}$\n%s' % (algName.split(',')[0], title)
                    else:
                        title_ = '$\\bf{%s}$\n%s' % (algName, title)
                    if 'NLRT' in algName:
                        title_ = '$\\bf{NLRT}$\n%s' % title
                    ax[i][j].set_title(title_, fontsize=titlesize)
            if nrows == 1: ax = ax[0] #
            if ncols == 1: ax = ax[0] #
            plt.subplots_adjust(wspace=wspace, hspace=hspace)
            return fig, ax
    
    def plotConvergence2(self, nttsvd=True, nsthosvd=True, testMatrixName=False,
                        figsize=None,
                        yticks=None,
                        wspace=None, hspace=None,
                        titlesize=None,
                        ticksize=None,
                        legendsize=None,
                        legendloc=None):
        titles = ['Distance to nonnegative tensors', 'Percent of negative elements']
        norms  = ['Chebyshev', 'Frobenius', 'Density']
        colors = ['C0', 'C1', 'C2']

        infoList = []
        if nttsvd and self._nttsvdInfo:
            infoList.append(self._nttsvdInfo)
        if nsthosvd and self._nsthosvdInfo:
            infoList.append(self._nsthosvdInfo)

        if len(infoList) > 0:
            nrows = len(infoList[0])
            if len(infoList) > 1:
                nrows += len(infoList[1])
            fig, ax = plt.subplots(nrows, 2, figsize=figsize)
            if nrows == 1: ax = [ax] #
            for k in range(len(infoList)):
                for kk in range(len(infoList[k])):
                    info = infoList[k][kk]
                    algName = info.getFullAlgName() if testMatrixName else ',\\ '.join(info.getFullAlgName().split(', ')[:2])
                    convInfo = info.getConvergenceInfo()
                    i = k*len(infoList[0]) + kk
                    percentOfNegElems = [100*x for x in convInfo['density']]
                    ax[i][0].plot(range(1, self._itersNum+1), convInfo['chebyshev'], colors[0], label=norms[0])
                    ax[i][0].plot(range(1, self._itersNum+1), convInfo['frobenius'], colors[1], label=norms[1])
                    ax[i][1].plot(range(1, self._itersNum+1), percentOfNegElems, colors[2], label=norms[2])
                    ax[i][0].set_yscale('log')
                    if yticks is not None:
                        ax[i][0].set_yticks(yticks)
                    ax[i][0].legend(loc=legendloc, fontsize=legendsize)
                    for j in range(2):
                        if 'SVD$_r$' in algName:
                            title = '$\\bf{%s,\\ SVD_r}$\n%s' % (algName.split(',')[0], titles[j])
                        else:
                            title = '$\\bf{%s}$\n%s' % (algName, titles[j])
                        ax[i][j].set_title(title, fontsize=titlesize)
                        ax[i][j].tick_params(axis='both', labelsize=ticksize)
                        ax[i][j].grid()
            if nrows == 1: ax = ax[0] #
            plt.subplots_adjust(wspace=wspace, hspace=hspace)
            return fig, ax
    
    def plotRuntimes(self, nttsvd=True,
                           nsthosvd=True,
                           nlrt=False,
                           testMatrixName=False,
                           figsize=None,
                           wspace=None,
                           rotation=None,
                           ha='center',
                           ylabelsize=None,
                           titlesize=None,
                           xticksize=None, yticksize=None,
                           paramsNewLine=False,
                           hideTitle=False,
                           exclude=[]):
        algNames = []
        times = []
        truncatedSvdList = []
        if nsthosvd and self._nsthosvdTimes:
            algNames.append('NSTHOSVD')
            times.append([self._nsthosvdTimes[i] for i in range(len(self._nsthosvdTimes)) if i not in exclude])
            truncatedSvdList.append([self._nsthosvdTruncatedSvdList[i] for i in range(len(self._nsthosvdTruncatedSvdList)) if i not in exclude])
        if nlrt and self._nlrtTimes:
            algNames.append('NLRT')
            times.append([self._nlrtTimes[i] for i in range(len(self._nlrtTimes)) if i not in exclude])
            truncatedSvdList.append([self._nlrtTruncatedSvdList[i] for i in range(len(self._nlrtTruncatedSvdList)) if i not in exclude])
        if nttsvd and self._nttsvdTimes:
            algNames.append('NTTSVD')
            times.append([self._nttsvdTimes[i] for i in range(len(self._nttsvdTimes)) if i not in exclude])
            truncatedSvdList.append([self._nttsvdTruncatedSvdList[i] for i in range(len(self._nttsvdTruncatedSvdList)) if i not in exclude])
        ncols = len(algNames)
        if ncols == 0: return
        fig, ax = plt.subplots(1, ncols, figsize=figsize)
        if ncols == 1: ax = [ax] #
        for i in range(len(algNames)):
            if testMatrixName:
                xticks_labels = ['\n'.join(truncatedSvd.getName().split(', ')) for truncatedSvd in truncatedSvdList[i]]
            else:
                xticks_labels = [truncatedSvd.getName().split(', ')[0] for truncatedSvd in truncatedSvdList[i]]
                if paramsNewLine:
                    xticks_labels = ['\n('.join(label.split('(')) for label in xticks_labels]
            if not hideTitle:
                ax[i].set_title(algNames[i], fontsize=titlesize)
            ax[i].set_ylabel('Running time (Second)', size=ylabelsize)
            ax[i].tick_params(axis='both', direction='in', right=True, top=True)
            ax[i].tick_params(axis='x', rotation=rotation)
            ax[i].set_xticks(ticks=range(len(truncatedSvdList[i])))
            ax[i].tick_params(axis='y', labelsize=yticksize)
            ax[i].set_xticklabels(labels=xticks_labels, size=xticksize, ha=ha)
            ax[i].bar(range(0, len(truncatedSvdList[i]), 1), times[i], edgecolor='black')
        if ncols == 1: ax = ax[0] #
        plt.subplots_adjust(wspace=wspace)
        return fig, ax
    
    def showApproximations(self, nttsvd=True, nsthosvd=True, band=0, testMatrixName=False,
                           figsize=None,
                           ncols=3,
                           titlesize=None,
                           wspace=None, hspace=None,
                           cmap=None): # for HSIs
        approximationsList = [self._tensor]
        titles = ['Original']
        if nttsvd:
            if self._ttsvdApproximation:
                titles.append('TTSVD')
                approximationsList.append(self._ttsvdApproximation)
            if self._nttsvdApproximations:
                if testMatrixName:
                    titles += ['NTTSVD, ' + truncatedSvd.getName() for truncatedSvd in self._nttsvdTruncatedSvdList]
                else:
                    titles += ['NTTSVD, ' + truncatedSvd.getName().split(', ')[0] for truncatedSvd in self._nttsvdTruncatedSvdList]
                approximationsList.extend(self._nttsvdApproximations)
        if nsthosvd:
            if self._sthosvdApproximation:
                titles.append('STHOSVD')
                approximationsList.append(self._sthosvdApproximation)
            if self._nsthosvdApproximations:
                if testMatrixName:
                    titles += ['NSTHOSVD, ' + truncatedSvd.getName() for truncatedSvd in self._nsthosvdTruncatedSvdList]
                else:
                    titles += ['NSTHOSVD, ' + truncatedSvd.getName().split(', ')[0] for truncatedSvd in self._nsthosvdTruncatedSvdList]
                approximationsList.extend(self._nsthosvdApproximations)

        if len(titles) > 1:
            m = ceil(len(titles) / ncols)
            n = min(ncols, len(titles))
            fig, ax = plt.subplots(m, n, figsize=figsize)
            if m == 1: ax = [ax] #
            for k in range(len(titles)):
                title = titles[k]
                imgr = approximationsList[k][band]
                i = k // ncols
                j = k % ncols
                ax[i][j].set_title('%s' % title, fontsize=titlesize)
                ax[i][j].imshow(imgr, cmap=cmap)
                ax[i][j].axis('off')
            plt.subplots_adjust(wspace=wspace, hspace=hspace)
            if m == 1: ax = ax[0] #
            return fig, ax

    def set_ttRank(self, ttRank):
        self._ttRank = ttRank
    
    def set_tuckerRank(self, tuckerRank):
        self._tuckerRank = tuckerRank
    
    def setRanks(self, ttRank, tuckerRank):
        self._ttRank = ttRank
        self._tuckerRank = tuckerRank

    def setTensor(self, tensor):
        self._tensor = tensor
    
    def setTruncatedSvdList(self, truncatedSvdList):
        self._nttsvdTruncatedSvdList = truncatedSvdList
        self._nsthosvdTruncatedSvdList = truncatedSvdList
    
    def set_nttsvdTruncatedSvdList(self, truncatedSvdList):
        self._nttsvdTruncatedSvdList = truncatedSvdList

    def set_nsthosvdTruncatedSvdList(self, truncatedSvdList):
        self._nsthosvdTruncatedSvdList = truncatedSvdList
    
    def getRanks(self):
        return {'ttRank': self._ttRank, 'tuckerRank': self._tuckerRank}


def computeCompression(shape, ranks):
    shape = np.array(shape)
    orig_size = np.prod(shape)
    if (len(shape) == len(ranks)): # Tucker format
        ranks = np.array(ranks)
        return orig_size / (np.prod(ranks) + (shape * ranks).sum())
    compr = shape * np.array([1] + list(ranks))
    compr *= np.array(list(ranks) + [1])
    return orig_size / compr.sum() # TT format
