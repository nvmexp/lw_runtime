import sys

import scipy
import scipy.cluster.hierarchy as sch
import pandas as pd

import sys
import itertools
import copy
import random
import math
import argparse

import glob
import scipy.cluster.vq as sc
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random

from lwtensorUtil import *
from candidateLwtlass import *

def getRegAndLmem(kernel):
    numRegs = int(findAndSplit("reg:", kernel, []))
    lmem = int(findAndSplit("lmem:", kernel, []))
    return numRegs, lmem

def parse(filename, flops, kernels, problemSizes, kernelIsFastest, ta, tb):
    f = open(filename,"r")
    #
    # Parse perf for each test case and kernel
    #
    for l in f:
        if l.find("lwTENSOR:") == -1:
            continue
        extent = findAndSplit("-extent", l, [',','='])
        if( len(extent) == 0 ):
            print("WARNING: not processed: ", l)
        modeA = findAndSplit("-modeA", l, [','])
        modeB = findAndSplit("-modeB", l, [','])
        modeC = findAndSplit("-modeC", l, [','])
        modeM = intersect(modeA, modeC)
        modeN = intersect(modeB, modeC)
        modeK = intersect(modeA, modeB)
        modeL = intersect(modeK, modeC)
        for mode in modeL:
            modeK.remove(mode);
            modeN.remove(mode);
            modeM.remove(mode);
        m = product(modeM, extent)
        n = product(modeN, extent)
        k = product(modeK, extent)
        L = product(modeL, extent)
        transA = modeA[0] in modeK
        transB = not modeB[0] in modeK
        if (l.find("contraction") != -1 and (transA != ta or transB != tb)):
            continue # skip since we separate the contraction benchmarks into its transpose variants
        flop = findAndSplit("lwTENSOR:", l, [])
        algo = int(findAndSplit("-algo", l, []))
        if( algo == 1000 or algo < 0 ):
            continue # only analyze lwtlass

        pos = l.find("kernel:")
        if( pos == -1 ):
            #print("WARNING", l)
            continue # TODO warning
        pos += len("kernel:")
        endPos = l.find("lmem:")
        endPos = l.find(";", endPos) #comma after lmem

        kernel = l[pos:endPos]
        #kernel = kernel[0: kernel.find("t:")] + kernel[kernel.find("t:")+6:] #remove transpose from kernelname => treat all transpose variation identical
        if (not(kernel in kernels)):
            kernels.append(kernel)

        if (l.find("key:") == -1):
            continue
        key = l.split("|")[0]
        key = key.replace("-algo+%d"%algo, "")
        problemSizes[key] = (m,n,k,L)
        if( key in flops ):
            flops[key].append((kernel, flop))
        else:
            flops[key] = [ (kernel, flop) ]

    # find fastest kernels
    for tc in flops:
        if(len(flops[tc])<3):
            continue
        flops[tc]
        flops[tc] = sorted(flops[tc], key=lambda t:t[1], reverse=True) #sort contractions w.r.t. their total time
        kernel = flops[tc][0][0]
        if (kernel in kernelIsFastest):
            kernelIsFastest[kernel] += 1
        else:
            kernelIsFastest[kernel] = 1


class KernelSubset:
    def __init__(self, kernels, evaluation):
        self.kernels = kernels
        self.evaluation = evaluation
        self.histogram = [len([1 for i in evaluation.values() if (j * 0.1) <= i <= ((j + 1) * 0.1)]) for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        self.wavg =  sum([(10 - i ) * self.histogram[i] for i in range(10)])
        filtered = [val for key, val in evaluation.items() if val != 0]
        if filtered:
            self.avg = sum(filtered) / len(filtered)
            self.min = min(filtered)
        else:
            self.avg = 0
            self.min = 0
        self.skip = len(evaluation) - len(filtered)
        self.best = len([1 for key, val in evaluation.items() if val == 1])
    @staticmethod
    def from_set(kernels, flops):
        evaluation = {}
        for key in flops:
            best = 0
            for k in kernels:
                best = max(best, flops[key].get(k, 0))
            evaluation[key] = best
        return KernelSubset(frozenset(kernels), evaluation)
    def add(self, kernel, flops):
        assert kernel not in self.kernels
        new_evaluation = dict(self.evaluation)
        for key in self.evaluation:
            new_evaluation[key] = max(new_evaluation[key], self.flops[key].get(kernel, 0))
        return KernelSubset(self.kernels | frozenset(kernel), new_evaluation)
    def merge(self, other):
        new_evaluation = dict(self.evaluation)
        for key in self.evaluation:
            new_evaluation[key] = max(new_evaluation[key], other.evaluation[key])
        return KernelSubset(self.kernels | other.kernels, new_evaluation)
    def dominates(self, other):
        better = False
        for key in self.evaluation:
            if other.evaluation[key] > self.evaluation[key]: return False
            if other.evaluation[key] < self.evaluation[key]: better = True
        return better
    def __repr__(self):
        return "<KernelSubset " + repr(self.kernels) + " " + repr(self.avg) + " " + repr(self.min) + " " + repr(self.skip) + " " + repr(self.best) + " " + repr(len(self.kernels)) + ">"
    def __eq__(self, other):
        return self.kernels == other.kernels
    def __hash__(self):
        return hash(self.kernels)


def kernelAnalysis(filenames):
    flops = {}
    kernels = []
    problemSizes = {}
    kernelIsFastest = {}

    for filename in filenames:
        parse(filename, flops, kernels, problemSizes, kernelIsFastest)

    flops = {key: {kernel: flops / elem[0][1] for kernel, flops in elem} for key, elem in flops.items()}
    kernels = [KernelSubset.from_set(set([kernel]), flops) for kernel in kernels]
    
    problems = list(kernels[0].evaluation.keys())
    
    mat = np.zeros((len(problems), len(kernels)))
    for i, p in enumerate(problems):
        for j, k in enumerate(kernels):
            mat[i,j] = k.evaluation[p]
    print(mat.shape)
    return flops, kernels, mat

def cluster_corr(corr_array, inplace=False):

    pairwise_distances = sch.distance.pdist(corr_array)
    # pairwise_distances = 1 - np.abs(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    # sch.dendrogram(linkage)
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]

def cluster_mat(mat):
    corr_array = np.corrcoef(mat)
    pairwise_distances = sch.distance.pdist(corr_array)
    # pairwise_distances = 1 - np.abs(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    # sch.dendrogram(linkage)
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    return mat[idx, :]

def cluster_corr_n(kernels, corr_array, N, show=True, value=lambda a: a.avg + a.min):
    pairwise_distances = sch.distance.pdist(corr_array)
    # pairwise_distances = 1 - np.abs(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    # sch.dendrogram(linkage)
    idx_to_cluster_array = sch.fcluster(linkage, N, 
                                        criterion='maxclust')
    result = []
    for i in range(N):
        print(i, '====================')
        num = 0
        best = None
        for idx in range(corr_array.shape[0]):
            if idx_to_cluster_array[idx] == i + 1:
                kernel = kernels[idx]
                best = kernel if best is None or value(best) < value(kernel) else best
                if show:
                    print(i, kernels[idx])
                num += 1
        print(i, 'num', num)
        print(i, 'best', best)
        result.append(best)
    return result

class Summary:
    
    def __init__(self, values):
        self.avg = np.average(values[values > 0])
        self.min = np.min(values[values > 0], initial = 100)
        if self.min == 100: self.min = 0
        self.skip = np.sum(values == 0)
        self.best = np.sum(values == 1)

class Optimize:
    
    def __init__(self, mat, target=lambda a: np.min(a), kernels=None):
        self.mat = mat
        self.target = target
        self.num_kernels = mat.shape[1]
        self.kernels = kernels
        
    def apply(self, a):
        return np.max(self.mat[:, list(a)], axis=1, initial=0)
    
    def evaluate(self, a):
        return self.target(self.apply(a))
    
    def optimize_greedy(self, max_kernels, hysteresis=2):
        result = set()
        result_target = 0
        def generate(max_kernels, max_its, hysteresis):
            for i in range(max_kernels):
                yield True
            for i in range((max_its - max_kernels) // (2 * hysteresis)):
                for j in range(hysteresis):
                    yield False
                for j in range(hysteresis):
                    yield True
        for it in generate(max_kernels, 200, hysteresis):
            if it:
                best = -1
                best_target = result_target
                for i in range(self.num_kernels):
                    i_target = self.evaluate(result | set([i]))
                    if i_target > best_target:
                        best_target = i_target
                        best = i
                result.add(best)
                result_target = best_target
            else:
                best = -1
                best_target = 0
                for i in result:
                    i_target = self.evaluate(result - set([i]))
                    if i_target > best_target:
                        best_target = i_target
                        best = i
                result.remove(i)
                result_target = best_target
        return result_target, result
    
    def optimize_random(self, max_kernels, num_explore=1000, fixed=set()):
        result = set()
        result_target = 0
        pop = list(set(range(self.num_kernels)) - fixed)
        for it in range(num_explore):
            i = set(random.sample(pop, max_kernels - len(fixed))) | fixed
            i_target = self.evaluate(i)
            if i_target > result_target:
                result_target = i_target
                result = i
        return result_target, result
    
    def optimize_random_improve(self, max_kernels, num_rounds=100, num_look=2, num_explore=30000, fixed=set()):
        for r in range(num_rounds):
            looks = []
            for l in range(num_look):
                looks.append(self.optimize_random(max_kernels, num_explore, fixed=fixed)[1])
            intersection = looks[0]
            for l in looks[1:]:
                intersection &= l
            fixed = intersection
            # print(fixed)
            if len(fixed) == max_kernels:
                break
        return self.optimize_random(max_kernels, num_explore, fixed=fixed)
    
    def optimize_random_improve2(self, max_kernels, num_rounds=100, num_look=5, num_explore=30000, fixed=set()):
        for r in range(num_rounds):
            looks = []
            for l in range(num_look):
                looks.append(self.optimize_random(max_kernels, num_explore, fixed=fixed)[1])
                for other in looks[:-1]:
                    if other & looks[-1]:
                        fixed = other & looks[-1]
                        break
            #intersection = looks[0]
            #for l in looks[1:]:
            #    intersection &= l
            #fixed = intersection
            print(fixed)
            if len(fixed) == max_kernels:
                break
        return self.optimize_random(max_kernels, num_explore, fixed=fixed)
    
    def move(self, heur_p, current, kernels, remove=False):
        def reduce_skip(fn):
            def result(current, test): # return true if b is better
                if test.skip < current.skip: return True
                if test.skip == current.skip: return fn(current, test)
                return False
            return result
        #improve_thr_most = lambda current, test: test.thresh > current.thresh
        improve_avg_most = reduce_skip(lambda current, test: test.avg > current.avg)
        improve_min_most = reduce_skip(lambda current, test: test.min > current.min)
        improve_best_most = reduce_skip(lambda current, test: test.best > current.best)
        #improve_wavg_most = reduce_skip(lambda current, test: test.wavg > current.wavg)
        #def improve_hist_most_a(current, test):
        #    for c, t in zip(current.histogram[:-1], test.histogram[:-1]):
        #        if t < c: return True
        #        if t > c: return False
        #    return False
        #improve_hist_most = reduce_skip(improve_hist_most_a)
        improve_mix_most = reduce_skip(lambda current, test: test.avg + test.min > current.avg + current.min)
        improve_mix2_most = reduce_skip(lambda current, test: test.avg * test.min > current.avg * current.min)
        improve = {
            'a': improve_avg_most, 
            'm': improve_min_most, 
            'b': improve_best_most, 
            'x': improve_mix_most, 
            '2': improve_mix2_most, 
            #'h': improve_hist_most, 
            #'w': improve_wavg_most, 
            #'t': improve_thr_most
        }
        if remove:
            if heur_p == 'r':
                best_idx = random.randrange(len(current))
                best = current[best_idx]
            else:
                heur = improve[heur_p]
                best = None
                best_idx = -1
                for idx, kernel in enumerate(current):
                    test = current - frozenset([kernel])
                    if best is None or heur(Summary(self.apply(best)), Summary(self.apply(test))):
                        best = test
                        best_idx = idx
            new_kernels = list(kernels)
            if best_idx != -1:
                new_kernels.append(best_idx)
        else:
            if heur_p == 'r':
                best_idx = random.randrange(len(kernels))
                best = kernels[best_idx]
            else:
                heur = improve[heur_p]
                best = current
                best_idx = -1
                for idx, kernel in enumerate(kernels):
                    test = current | frozenset([kernel])
                    if heur(Summary(self.apply(best)), Summary(self.apply(test))):
                        best = test
                        best_idx = idx
            new_kernels = list(kernels)
            if best_idx != -1:
                del new_kernels[best_idx]
        return best, new_kernels
    
    def move2(self, heur_p, current, kernels):
        current, kernels = self.move(heur_p, current, kernels)
        if len(current) > 7:
            current, kernels = self.move(heur_p, current, kernels, remove=True)
        
    
    def prune(self, current, kernels):
        # exclude a kernel that is worse in every case than the best kernel in current
        kernels = list(kernels)
        prune = []
        for idx, kernel in enumerate(kernels):
            if np.all(self.mat[:, kernel] <= self.apply(current)):
                prune.append(idx)
        for idx in reversed(prune):
            del kernels[idx]
        return kernels
    
    def optimize_relwrsive(self, current = frozenset(), kernels = None, path = '', seen = None):
        if seen is None: seen = set()
        if kernels == None: kernels = list(range(self.num_kernels))
        if current in seen: return
        seen.add(current)
        lwrrent_sum = Summary(self.apply(current))
        if (lwrrent_sum.avg > 0.8) and (lwrrent_sum.min > 0.55):
            print("%.2f %.2f %d %s" % (lwrrent_sum.avg, lwrrent_sum.min, len(current), path))
        if len(current) == 7: return
        if len(kernels) == 0: return
        kernels = self.prune(current, kernels)
        best_seen = [current]
        for heur_p in 'ma2':
            best, new_kernels = self.move(heur_p, current, kernels)
            if best not in best_seen:
                best_seen.append(best)
                self.optimize_relwrsive(best, new_kernels, path + heur_p, seen)

    def optimize_anneal(self, current = set(), factor = 0.2):
        def cost_fn(c):
            return - 8 * np.min(self.apply(c)) + max(0, len(c) - 7)
        def is_sym(kernel):
            if self.kernels is None: return True
            kernel = list(map(int, list(self.kernels[kernel].kernels)[0].split('b:')[1].split(';')[0].split(',')))
            return (kernel[1] == 1) or (kernel[0] == kernel[1])
        kernels = [i for i in range(self.num_kernels)] # if is_sym(kernel)]
        cost = cost_fn(current)
        best_cost = cost
        best = None
        temp = 90
        while temp > 0.2:
            if current and ((len(current) > 10) or random.choice([True, False])):
                new_lwrrent, new_kernels = self.move('r', current, kernels, remove=True)
                #remove = random.choice(list(current))
                #new_lwrrent = current - set([remove])
                #new_kernels = list(kernels)
                #new_kernels.append(remove)
            else:
                #new_lwrrent, new_kernels = heuristic_apply(random.choice('aammxb2hhhhhwwwr'), current, kernels)
                random.shuffle(kernels)
                new_lwrrent, new_kernels = self.move2(random.choice('ma2'), current, kernels)
            # if size to big: there is a chance a random kernel gets deleted
            # iterate that
            # apply a random move
            new_cost = cost_fn(new_lwrrent)
            delta = cost - new_cost
            if delta > 0:
                cost = new_cost
                current = new_lwrrent
                kernels = new_kernels
            elif random.uniform(0, 1) < math.exp(delta / temp):
                cost = new_cost
                current = new_lwrrent
                kernels = new_kernels
            if cost < best_cost:
                print('%.2f %.2f %.2f %d' % (cost, np.min(self.apply(current)), np.average(self.apply(current)), len(current)), sorted(current))
                best_cost = cost
                best = current
            temp -= 0.01
        return best

def get_optimized_kernels_helper(N, files, ta, tb):
    flops = {}
    kernels = []
    problemSizes = {}
    kernelIsFastest = {}
    for filename in files:
        parse(filename, flops, kernels, problemSizes, kernelIsFastest, ta, tb)

    flops = {key: {kernel: flops / elem[0][1] for kernel, flops in elem} for key, elem in flops.items()}
    kernels = [KernelSubset.from_set(set([kernel]), flops) for kernel in kernels]
    
    problems = list(kernels[0].evaluation.keys())
    
    mat = np.zeros((len(problems), len(kernels)))
    for i, p in enumerate(problems):
        for j, k in enumerate(kernels):
            mat[i,j] = k.evaluation[p]
    

    #o = Optimize(mat, target=lambda a: 2 * np.min(a) + np.average(a))
    o = Optimize(mat)
    score, r = o.optimize_random_improve2(N, 100, 5, 100000)
    #score, r = o.optimize_random(N, 10000)
    print(score, r)
    r = list(r)
    for i in range(5 * N):
        rt = list(r)
        best_score = o.evaluate(rt)
        best = rt[0]
        del rt[0]
        for j in range(o.num_kernels):
            if o.evaluate(rt + [j]) > best_score:
                best_score = o.evaluate(rt + [j])
                best = j
        rt.append(best)
        print(score, np.min(o.apply(r)), np.average(o.apply(r)), r)
        print(best_score, np.min(o.apply(rt)), np.average(o.apply(rt)), rt)
        r = rt
        score = best_score
    for k in r:
        print(kernels[k])
    return [list(kernels[k].kernels)[0] for k in r]

def get_optimized_kernels(N, files):
    """
    select N kernels that yield best performance for all benchmarks provided via `files`
    (those files must be generated via lwtensorTest.
    """

    selected_kernels = []
    for ta, tb in itertools.product([True,False],[True,False]):
        selected_kernels += get_optimized_kernels_helper(N, files, ta, tb)

    return selected_kernels


#floPS0, kernels0, mat0 = kernelAnalysis(sys.argv[1:])

parser = argparse.ArgumentParser()
parser.add_argument('N', type=int)
parser.add_argument('files', type=str, nargs='*')
parser.add_argument('--fallback', type=str, default=None)
parser.add_argument('--output', type=str, default=None)

args = parser.parse_args()

N = args.N
input_files = args.files

output = get_optimized_kernels(N, input_files)

#infer output name from selected kernel
if args.output is None:
    can = CandidateLwtlass(output[0]) # decode
    output_file = f"tc.{can.getTypeString()}.sm{can.ccTarget}.kernels"
else:
    output_file = args.output

f_out = open(output_file, 'w+')
if args.fallback is not None:
    f_fall = open(fallback_source, 'r+')
    for l in f_fall:
        f_out.write(l)
for l in output:
    f_out.write(l + '\n')
