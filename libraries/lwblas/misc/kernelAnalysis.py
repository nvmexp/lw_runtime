import sys
import itertools
import copy
import random
import math

def findAndSplit(keyword, line, splits):
    """ finds the keyword in line and splits 1) ',' then 2) '=' and asigns it to the returnd dict."""
    pos = line.find(keyword)
    if( pos == -1 ):
        return []
    pos += len(keyword)
    if( len(splits) == 0 ):
        return float(line[pos:].split()[0])

    tokens = line[pos:].split()[0].split(splits[0])
    if( len(splits) == 2 ):
        extent = {}
        for token in tokens:
            tk = token.split(splits[1])
            if(len(tk) < 2):
                continue
            extent[tk[0]] = int(tk[1])
        return extent
    else:
        #fix for 0-order tensors
        if(tokens[0].startswith("-")):
            return []
        else:
            return tokens

def getHashKey( extent, modeA, modeB, modeC ):
    extent_sorted = []
    for ex in extent:
        extent_sorted.append((ex,extent[ex]))
    extent_sorted.sort()
    key = "-extent"
    for ex in extent_sorted:
        key+= ex[0] + "="+ str(ex[1]) +","
    key += " -modeA"
    for a in modeA:
        key += a + ","
    key += " -modeB"
    for a in modeB:
        key += a + ","
    key += " -modeC"
    for a in modeC:
        key += a + ","
    return key

def intersect(l1, l2):
    l = []
    for a in l1:
        if( a in l2):
            l.append(a)
    return l
def product(modes , extent):
    total = 1
    for m in modes:
        total *= extent[m]
    return total

def getRegAndLmem(kernel):
    numRegs = int(findAndSplit("reg:", kernel, []))
    lmem = int(findAndSplit("lmem:", kernel, []))
    return numRegs, lmem

def computeSubsetPerformance(flops, kernelSubset):
    """
    Compute the performance of a subset of kernels/candidates w.r.t. to the fastest available kernel for each testcase
    """
    minSpeedup = 10000000000000
    avgSpeedup = 0
    worstTestCaseForSubset = ""
    skip = False # skip subsets that cannot implement every test case
    # compute the maximal speedup w.r.t. the fastest kernel (across all test cases)
    for tc in flops:
        highestFlops = flops[tc][0][1]
        speedup = 0
        for kernel, flop in flops[tc]:
            if (kernel in kernelSubset):
                # maximal speedup of kernels in kernelSubset
                speedup = max(flop/highestFlops, speedup)
                break
        if (speedup == 0):
            skip = True
            break;
        avgSpeedup += speedup
        if (speedup < minSpeedup):
            minSpeedup = min(minSpeedup, speedup)
            worstTestCaseForSubset = tc

    return minSpeedup, avgSpeedup / len(flops), skip, worstTestCaseForSubset 

def colwert(kernel):
    """
    Colwerts a kernel s.t. it can be passed into the generator script
    e.g., 2;b:32,32,1;op:0;v:4;t:128; -> (2, 32,32,128,4,["OpPackIdentity"])
    """
    if( kernel.find("op:") != -1 and
        kernel.find("t:") != -1 and
        kernel.find("v:") != -1):
        # elementwise kernel
        tokens = kernel.split(';')
        ndim = int(tokens[0])
        blocking = [1 for x in range(3)]
        nthreads = 0
        vec = 0
        op = ""
        for tok in tokens[1:]:
            if ('b:' in tok):
                blocking[0] = tok.split(':')[1].split(',')[0]
                blocking[1] = tok.split(':')[1].split(',')[1]
                blocking[2] = tok.split(':')[1].split(',')[2]
            elif ('t:' in tok):
                nthreads = int(tok.split(':')[1])
            elif ('v:' in tok):
                vec = int(tok.split(':')[1])
            elif ('op:' in tok):
                if(int(tok.split(':')[1]) == 0):
                    op = "OpPackIdentity"
                else:
                    op = "OpPackGeneric"
        return f"({ndim}, {blocking[0]}, {blocking[1]}, {blocking[2]}, {nthreads}, {vec}, True, [\"{op}\"])"
    else:
        # tensor contraction kernel
        return kernel

def parse(filename, flops, kernels, problemSizes, kernelIsFastest):
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
        m = product(modeM, extent)
        n = product(modeN, extent)
        k = product(modeK, extent)
        flop = findAndSplit("lwTENSOR:", l, [])
        algo = int(findAndSplit("-algo", l, []))
        if( algo == 1000 or algo < 0 ):
            continue # only analyze lwtlass

        pos = l.find("kernel:")
        if( pos == -1 ):
            print("WARNING", l)
            continue # TODO warning
        pos += len("kernel:")
        endPos = l.find("lmem:")
        endPos = l.find(";", endPos) #comma after lmem

        kernel = l[pos:endPos]
        if (not(kernel in kernels)):
            kernels.append(kernel) 

        if (l.find("key:") == -1):
            continue
        key = l.split("|")[0]
        key = key.replace("-algo+%d"%algo, "")
        #key = getHashKey( extent, modeA, modeB, modeC )
        problemSizes[key] = (m,n,k)
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

def brute_force(kernels, flops, kernelIsFastest):
    for kernel in kernels:
        if (kernel in kernelIsFastest):
            print("Fastest: %s %s"%(kernel, kernelIsFastest[kernel]))
    for kernel in kernels:
        if (not (kernel in kernelIsFastest)):
            print("never fastest: ", kernel)

    kernels = [kernel for kernel in kernelIsFastest] # remove slow kernels

    numSubsets = 0
    maxSubsetSize = min(6,len(kernels))
    for numKernels in range(maxSubsetSize):
        for kernelSubset in itertools.combinations(kernels, numKernels + 1):
            numSubsets += 1

    lwrrentSubset = 0
    # performance for 1,2, ... 
    subsetPerf = [[] for i in range(1, maxSubsetSize + 1)]
    counter = 0
    printEvery = int(numSubsets / 200)

    #
    # BRUTE FORCE: try all subsets of candidates and evaluate their perf w.r.t. to the maximal perforamnce
    #
    for numKernels in range(maxSubsetSize):
        # try all combinations of kernel subsets
        for kernelSubset in itertools.combinations(kernels, numKernels + 1):
            lwrrentSubset += 1
            if( lwrrentSubset % printEvery == 0):
                print("%d/%d (%.3f%%)"%(lwrrentSubset, numSubsets, 100*float(lwrrentSubset)/numSubsets))

            minSpeedup, avgSpeedup, skip, worstTestCaseForSubset = computeSubsetPerformance(flops, kernelSubset) # speedup w.r.t. to fastest kernel (value in [0,1])
                
            if (not skip):
                counter += 1
                subsetPerf[numKernels].append((list(kernelSubset), avgSpeedup, minSpeedup, worstTestCaseForSubset))
                #print(kernelSubset, avgSpeedup)
                #if( counter >= 20):
                #    exit(0)

    for kernel in kernels:
        if (kernel in kernelIsFastest):
            print("Fastest: %s %s"%(kernel, kernelIsFastest[kernel]))
    for kernel in kernels:
        if (not (kernel in kernelIsFastest)):
            print("never fastest: ", kernel)

    #
    # print the 5 best subsets that use numKernels many kernels
    #
    for numKernels in range(maxSubsetSize):
        subsetPerf[numKernels] = sorted(subsetPerf[numKernels], key=lambda t:t[1], reverse=True) #sort w.r.t. average speedup
        #subsetPerf[numKernels] = sorted(subsetPerf[numKernels], key=lambda t:t[2], reverse=True) #sort w.r.t. minimal speedup

        numSolutions = len(subsetPerf[numKernels])
        if (numSolutions < 1):
            print("no solution found for %d kernels"%numKernels)
        for i in range(0, min(5, numSolutions)):
            kernelSubset, avgSpeedup, minSpeedup, worstTestCaseForSubset = subsetPerf[numKernels][i]
            print(kernelSubset, "avg: %.3f min: %.3f"%(avgSpeedup, minSpeedup))

            # now we also add the kernel that increases the minimal speedup the most:
            kernel = flops[worstTestCaseForSubset][0][0]
            kernelSubset = kernelSubset + [kernel]

            minSpeedup, avgSpeedup, skip, worstTestCaseForSubset = computeSubsetPerformance(flops, kernelSubset) # speedup w.r.t. to fastest kernel (value in [0,1])
            print("  ",kernelSubset, "avg: %.3f min: %.3f"%(avgSpeedup, minSpeedup))
            generatorString = ""
            for kernel in kernelSubset:
                generatorString += colwert(kernel) + ",\n"
            print(generatorString)


class KernelSubset:
    def __init__(self, kernels, evaluation):
        self.kernels = kernels
        self.evaluation = evaluation
        self.histogram = [len([1 for i in evaluation.values() if (j * 0.1) <= i <= ((j + 1) * 0.1)]) for j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        self.wavg =  sum([(10 - i ) * self.histogram[i] for i in range(10)])
        self.thresh = len([val for key, val in evaluation.items() if val > 0.75])
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
    def bad_problems(self, threshold):
        return [problem for problem, value in self.evaluation.items() if value < threshold]


def heuristic_prune(current, kernels):
    # exclude a kernel that is worse in every case than the best kernel in current
    kernels = list(kernels)
    prune = []
    for idx, kernel in enumerate(kernels):
        if current.dominates(kernel): prune.append(idx)
    for idx in reversed(prune):
        del kernels[idx]
    return kernels

def reduce_skip(fn):
    def result(current, test): # return true if b is better
        if test.skip < current.skip: return True
        if test.skip == current.skip: return fn(current, test)
        return False
    return result

def heuristic_apply(heur_p, current, kernels):
    improve_thr_most = lambda current, test: test.thresh > current.thresh
    improve_avg_most = reduce_skip(lambda current, test: test.avg > current.avg)
    improve_min_most = reduce_skip(lambda current, test: test.min > current.min)
    improve_best_most = reduce_skip(lambda current, test: test.best > current.best)
    improve_wavg_most = reduce_skip(lambda current, test: test.wavg > current.wavg)
    def improve_hist_most_a(current, test):
        for c, t in zip(current.histogram[:-1], test.histogram[:-1]):
            if t < c: return True
            if t > c: return False
        return False
    improve_hist_most = reduce_skip(improve_hist_most_a)
    improve_mix_most = reduce_skip(lambda current, test: test.avg + test.min > current.avg + current.min)
    improve_mix2_most = reduce_skip(lambda current, test: test.avg * test.min > current.avg * current.min)
    improve = {'a': improve_avg_most, 'm': improve_min_most, 'b': improve_best_most, 'x': improve_mix_most, '2': improve_mix2_most, 'h': improve_hist_most, 'w': improve_wavg_most, 't': improve_thr_most}
    if heur_p == 'r':
        best_idx = random.randrange(len(kernels))
        best = kernels[best_idx]
    else:
        heur = improve[heur_p]
        best = current
        best_idx = -1
        for idx, kernel in enumerate(kernels):
            test = current.merge(kernel)
            if heur(best, test):
                best = test
                best_idx = idx
    new_kernels = list(kernels)
    if best_idx != -1:
        del new_kernels[best_idx]
    return best, new_kernels

def heuristic_relwrsive(current, kernels, path, seen = None):
    if seen is None: seen = set()
    if current in seen: return
    seen.add(current)
    if (current.avg > 0.8) and (current.min > 0.4):
        # print(current.avg, current.min, current.kernels)
        print("%.2f %.2f %d %s" % (current.avg, current.min, len(current.kernels), path), current.histogram)
    if len(current.kernels) == 6: return
    if len(kernels) == 0: return
    kernels = heuristic_prune(current, kernels)
    best_seen = [current]
    for heur_p in 't':
        best, new_kernels = heuristic_apply(heur_p, current, kernels)
        if best not in best_seen:
            best_seen.append(best)
            heuristic_relwrsive(best, new_kernels, path + heur_p, seen)

def heuristic_anneal(kernels, flops2):
    def cost_fn(c):
        #return 2 * (max(len(c.kernels), 7) - 7) - 4 * c.avg * c.min - c.avg - 6 * c.min + c.skip + 0.1 * len(c.kernels)
        #return 2 * (max(len(c.kernels), 7) - 7) - 4 * c.avg * c.min - c.avg - 7 * c.min + c.skip + 0.23 * len(c.kernels)
        return - 8 * c.min + c.skip + 0.2 * len(c.kernels)
    flops = {key: {kernel: flops / elem[0][1] for kernel, flops in elem} for key, elem in flops2.items()}
    def is_sym(kernel):
        kernel = list(map(int, kernel.split('b:')[1].split(';')[0].split(',')))
        return (kernel[1] == 1) or (kernel[0] == kernel[1])
    kernels = [KernelSubset.from_set(set([kernel]), flops) for kernel in kernels] # if is_sym(kernel)]
    current = KernelSubset.from_set(set(), flops)
    cost = cost_fn(current)
    best_cost = cost
    temp = 90
    while temp > 0.2:
        if current.kernels and ((len(current.kernels) > 10) or random.choice([True, False])):
            remove = random.choice(list(current.kernels))
            new_lwrrent = KernelSubset.from_set(current.kernels - set([remove]), flops)
            new_kernels = list(kernels)
            new_kernels.append(KernelSubset.from_set(set([remove]), flops))
        else:
            #new_lwrrent, new_kernels = heuristic_apply(random.choice('aammxb2hhhhhwwwr'), current, kernels)
            random.shuffle(kernels)
            new_lwrrent, new_kernels = heuristic_apply(random.choice('mh2'), current, kernels)
        # if size to big: there is a chance a random kernel gets deleted
        # iterate that
        # apply a random move
        new_cost = cost_fn(new_lwrrent)
        delta = cost - new_cost
        if delta > 0:
            cost = new_cost
            current = new_lwrrent
            kernels = new_kernels
            if False and current.skip == 0 and current.avg > 0.8 and current.min > 0.55 and len(current.kernels) < 8:
                print('A %.2f %.2f %.2f %d' % (cost, current.avg, current.min, len(current.kernels)))
        elif random.uniform(0, 1) < math.exp(delta / temp):
            cost = new_cost
            current = new_lwrrent
            kernels = new_kernels
            if False and current.skip == 0 and current.avg > 0.8 and current.min > 0.55 and len(current.kernels) < 8:
                print('B %.2f %.2f %.2f %d' % (cost, current.avg, current.min, len(current.kernels)))
        if cost < best_cost:
            print('C %.2f %.2f %.2f %d' % (cost, current.avg, current.min, len(current.kernels)), sorted(current.kernels), current.histogram)
            best_cost = cost
        # temp -= 0.01
        temp -= 0.005
    # print(current.bad_problems(0.6))

def heuristic_analyze(kernels, flops2):
    flops = {key: {kernel: flops / elem[0][1] for kernel, flops in elem} for key, elem in flops2.items()}
    # prune
    dominated = []
    better = lambda a, b: a > b
    equal = lambda a, b: abs(a - b) <= 0.05 * max(a, b)
    for k1 in kernels:
        if k1 in dominated: continue
        for k2 in kernels:
            if k2 in dominated: continue
            total = 0
            k1_better_k2 = 0
            k1_equal_k2 = 0
            k2_supports_more = 0
            for key in flops:
                if (k1 not in flops[key]) and (k2 not in flops[key]): continue
                if k2 not in flops[key]: continue
                if k1 not in flops[key]:
                    k2_supports_more += 1
                    break
                if equal(flops[key][k1], flops[key][k2]):
                    k1_equal_k2 += 1
                elif better(flops[key][k1], flops[key][k2]):
                    k1_better_k2 += 1
                total += 1
            if k2_supports_more > 0: continue
            if total != k1_better_k2 + k1_equal_k2: continue
            if k1_better_k2 == 0:
                if k1 >= k2:
                    continue
            dominated.append(k2)
    print('dominated', len(dominated))
    kernels = [k for k in kernels if k not in dominated]
    print('remaining', len(kernels))
    kernels = KernelSubset.from_set(set(['2;b:16,16,1;op:0;v:4;t:32;', '2;b:32,32,1;op:0;v:4;t:128;', '3;b:16,16,16;op:0;v:4;t:512;', '3;b:4,4,16;op:0;v:2;t:32;', '3;b:4,8,4;op:0;v:2;t:64;', '3;b:8,4,8;op:0;v:2;t:64;']), flops)
    kernels = KernelSubset.from_set(set(['2;b:16,16,1;op:0;v:4;t:32;', '2;b:32,32,1;op:0;v:4;t:128;', '3;b:16,16,16;op:0;v:4;t:512;',                                '3;b:4,4,16;op:0;v:2;t:32;', '3;b:4,8,4;op:0;v:2;t:32;',   '3;b:64,32,4;op:0;v:4;t:512;', '3;b:8,8,4;op:0;v:4;t:32;']), flops)
    print(kernels)
    print(kernels.bad_problems(0.6))
    

def heuristic(kernels, flops):
    # prune
    flops = {key: {kernel: flops / elem[0][1] for kernel, flops in elem} for key, elem in flops.items()}
    # dominated = []
    # better = lambda a, b: a > b
    # equal = lambda a, b: a == b
    # for k1 in kernels:
    #     if k1 in dominated: continue
    #     for k2 in kernels:
    #         if k2 in dominated: continue
    #         total = 0
    #         k1_better_k2 = 0
    #         k1_equal_k2 = 0
    #         k2_supports_more = 0
    #         for key in flops:
    #             if (k1 not in flops[key]) and (k2 not in flops[key]): continue
    #             if k2 not in flops[key]: continue
    #             if k1 not in flops[key]:
    #                 k2_supports_more += 1
    #                 break
    #             if better(flops[key][k1], flops[key][k2]):
    #                 k1_better_k2 += 1
    #             if equal(flops[key][k1], flops[key][k2]):
    #                 k1_equal_k2 += 1
    #             total += 1
    #         if k2_supports_more > 0: continue
    #         if total != k1_better_k2 + k1_equal_k2: continue
    #         if k1_better_k2 == 0: continue
    #         dominated.append(k2)
    # print(len(dominated))
    # kernels = [k for k in kernels if k not in dominated]
    # print(len(kernels))
    heuristic_relwrsive(KernelSubset.from_set(set(), flops), [KernelSubset.from_set(set([kernel]), flops) for kernel in kernels], '')
        
def heuristic_reverse(kernels, flops):
    flops = {key: {kernel: flops / elem[0][1] for kernel, flops in elem} for key, elem in flops.items()}
    kernels = KernelSubset.from_set(set(kernels), flops)
    i = 0
    while len(kernels.kernels) > 1:
        scheduleAvg = i * 0.01
        scheduleMin = (i // 2) * 0.01
        i += 1
        if i > 50: break
        print(scheduleAvg, scheduleMin)
        modified = True
        while modified:
            modified = False
            for kernel in kernels.kernels:
                new_kernels = KernelSubset.from_set(kernels.kernels - set([kernel]), flops)
                print(new_kernels.avg, kernels.avg, new_kernels.min, kernels.min, len(new_kernels.kernels), len(kernels.kernels))
                if ((kernels.avg - new_kernels.avg) <= scheduleAvg) and ((kernels.min - new_kernels.min) <= scheduleMin) and (new_kernels.skip == 0):
                    kernels = new_kernels
                    modified = True
        print(kernels)

def kernelAnalysis(filenames):
    flops = {}
    kernels = []
    problemSizes = {}
    kernelIsFastest = {}

    for filename in filenames:
        parse(filename, flops, kernels, problemSizes, kernelIsFastest)

    #kernelSubset = ['2;b:32,32,1;op:0;v:4;t:128;', # trained on easyEW
    #        '2;b:16,8,1;op:0;v:2;t:32;', '2;b:64,64,1;op:0;v:4;t:256;', '2;b:8,8,1;op:0;v:2;t:32;',
    #        '2;b:8,32,1;op:0;v:4;t:64;', '2;b:8,16,1;op:0;v:2;t:32;', '2;b:32,32,1;op:0;v:2;t:128;']
    ##kernelSubset = ['2;b:16,8,1;op:0;v:2;t:32;', # trained on rand300
    ##        '2;b:8,16,1;op:0;v:2;t:32;',
    ##        '2;b:16,16,1;op:0;v:4;t:32;',
    ##        '2;b:16,16,1;op:0;v:4;t:64;',
    ##        '2;b:32,32,1;op:0;v:2;t:128;',
    ##        '2;b:16,8,1;op:0;v:2;t:64;']
    #minSpeedup, avgSpeedup, skip, worstTestCaseForSubset = computeSubsetPerformance(flops, kernelSubset) # speedup w.r.t. to fastest kernel (value in [0,1])
    #print("  ",kernelSubset, "avg: %.3f min: %.3f"%(avgSpeedup, minSpeedup))
    #exit(0)

    BRUTE_FORCE = False
    if BRUTE_FORCE:
        brute_force(kernels, flops, kernelIsFastest)
    else:
        # heuristic_analyze(kernels, flops)
        heuristic_anneal(kernels, flops)
        heuristic_anneal(kernels, flops)
        heuristic_anneal(kernels, flops)
        heuristic_anneal(kernels, flops)
        heuristic_anneal(kernels, flops)
        #heuristic(kernels, flops)

if (len(sys.argv) == 1):
    print("usage: <benchmark> <benchmark>,...")
else:
    kernelAnalysis(sys.argv[1:])
