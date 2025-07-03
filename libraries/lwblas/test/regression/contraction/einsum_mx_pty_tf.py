import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import torch as th # pip install torch
import argparse
import time

import os
os.elwiron["MXNET_GPU_WORKER_NTHREADS"] = "1"
os.elwiron["MXNET_COPY_WORKER_NTHREADS"] = "1"


parser = argparse.ArgumentParser(description='Profile einsum')

parser.add_argument('--benchmark', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--dtype', type=str, 
                    choices=['float16', 'float32'], help='Data type',
                    default='float32')
parser.add_argument('--check', action='store_true',
                    help='Check correcness')
parser.add_argument('--niters', type=int, help='Number of iterations',
                    default='20')


args = parser.parse_args()
niters = args.niters
filename = args.benchmark
fo = args.output

def run_mxnet(test):
    import mxnet as mx
    mx.npx.set_np()
    equation = test[0]
    if args.dtype == 'float16':
        lhs = np.random.normal(0, 1, (test[1])).astype('float16')
        rhs = np.random.normal(0, 1, (test[2])).astype('float16')
        mx_lhs = mx.np.array(lhs, dtype=np.float16, ctx=mx.gpu())
        mx_rhs = mx.np.array(rhs, dtype=np.float16, ctx=mx.gpu())
    else:
        lhs = np.random.normal(0, 1, (test[1])).astype('float32')
        rhs = np.random.normal(0, 1, (test[2])).astype('float32')
        mx_lhs = mx.np.array(lhs, dtype=np.float32, ctx=mx.gpu())
        mx_rhs = mx.np.array(rhs, dtype=np.float32, ctx=mx.gpu())

    minTimeMX = 1e1000
    for i in range(niters):
        # test MXNet
        start = time.time()
        out_mxnet = mx.np.einsum(equation, mx_lhs, mx_rhs, optimize=False)
        mx.nd.waitall()
        myTime = time.time() - start
        minTimeMX = min(minTimeMX, myTime)

    # Correctness test
    if args.check:
        out_ref = np.einsum(equation, lhs, rhs)
        rtol=1e-04; atol=1e-04;
        if args.dtype == 'float16':
            rtol=1e-03; atol=1e-03;
        correctness_mx = np.allclose(out_mxnet.asnumpy(), out_ref, rtol=rtol, atol=atol)
        print('Correctness MXNet: ', correctness_mx)
    return minTimeMX 


def run_pyt(test):
    equation = test[0]
    if args.dtype == 'float16':
        lhs = np.random.normal(0, 1, (test[1])).astype('float16')
        rhs = np.random.normal(0, 1, (test[2])).astype('float16')
        th_lhs = th.from_numpy(lhs).half().lwca()
        th_rhs = th.from_numpy(rhs).half().lwca()
    else:
        lhs = np.random.normal(0, 1, (test[1])).astype('float32')
        rhs = np.random.normal(0, 1, (test[2])).astype('float32')
        th_lhs = th.from_numpy(lhs).float().lwca()
        th_rhs = th.from_numpy(rhs).float().lwca()

    minTimePyT = 1e1000
    for i in range(niters):
        # test PyTorch
        start = time.time()
        out_torch = th.einsum(equation, th_lhs, th_rhs).contiguous()
        th.lwca.synchronize()
        myTime = time.time() - start
        minTimePyT = min(minTimePyT, myTime)

    # Correctness test
    if args.check:
        out_ref = np.einsum(equation, lhs, rhs)
        rtol=1e-04; atol=1e-04;
        if args.dtype == 'float16':
            rtol=1e-03; atol=1e-03;
        correctness_th = np.allclose(out_torch.cpu().numpy(), out_ref, rtol=rtol, atol=atol)
        print('Correctness PyTorch: ', correctness_th)
    return minTimePyT 

def run_tf(test):
    equation = test[0]

    with tf.device('/GPU:0'):
        if args.dtype == 'float16':
            lhs = np.random.normal(0, 1, (test[1])).astype('float16')
            rhs = np.random.normal(0, 1, (test[2])).astype('float16')
            tf_lhs = tf.experimental.numpy.ndarray(lhs.shape, dtype=tf.float16, buffer=lhs)
            tf_rhs = tf.experimental.numpy.ndarray(rhs.shape, dtype=tf.float16, buffer=lhs)
        else:
            lhs = np.random.normal(0, 1, (test[1])).astype('float32')
            rhs = np.random.normal(0, 1, (test[2])).astype('float32')
            tf_lhs = tf.experimental.numpy.ndarray(lhs.shape, dtype=float, buffer=lhs)
            tf_rhs = tf.experimental.numpy.ndarray(rhs.shape, dtype=float, buffer=rhs)

        th.lwca.synchronize()
        minTime = 1e1000
        for i in range(niters):
            start = time.time()
            out = tf.einsum(equation, tf_lhs, tf_rhs)
            th.lwca.synchronize()
            myTime = time.time() - start
            minTime = min(minTime, myTime)

        if args.check:
            out_ref = np.einsum(equation, lhs, rhs)
            rtol=1e-04; atol=1e-04;
            if args.dtype == 'float16':
                rtol=1e-03; atol=1e-03;
            correctness = np.allclose(out.cpu().numpy(), out_ref, rtol=rtol, atol=atol)
            print('Correctness PyTorch: ', correctness)

    return minTime

def run_test(test):
    
    flops = 2. * np.prod(np.array(test[1]))

    gflopsMX  = flops/run_mxnet(test[0])/1e9
    gflopsPyT = flops/run_pyt(test[0])/1e9
    gflopsTf  = flops/run_tf(test[0])/1e9

    print(str(test[0]).replace(" ",""),
            "MXNet: %.2f gflops"%(gflopsMX),
            "pyt: %.2f gflops"%(gflopsPyT),
            "tf: %.2f gflops"%(gflopsTf),
            "pyt/mxnet: %.2f"%(gflopsMX / gflopsPyT), 'x',
            "tf/mxnet: %.2f"%(gflopsMX / gflopsTf), 'x',)

    return gflopsMX, gflopsPyT, gflopsTf

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
            nonEmptyTokens = []
            for tok in tokens:
                if (tok != ''):
                    nonEmptyTokens.append(tok)
            return nonEmptyTokens 

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
 
def parseLwtensorFile(filename):
    tests = []
    f = open(filename)
    for l in f:
        extent = findAndSplit("-extent", l, [',','='])
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
        M = product(modeM, extent)
        N = product(modeN, extent)
        K = product(modeK, extent)
        L = product(modeL, extent)
        modeA.reverse() #account for row-major layout
        modeB.reverse()
        modeC.reverse()

        shapeA = []
        for m in modeA:
            shapeA.append(extent[m])
        shapeB = []
        for m in modeB:
            shapeB.append(extent[m])

        einsum = "".join(modeA) + "," + "".join(modeB) + "->" + "".join(modeC)
        test = ([einsum, shapeA, shapeB], (M,N,K,L))
        tests.append(test)
    return tests

if __name__ == "__main__":
    tests = parseLwtensorFile(filename)

    f = open(fo + ".csv", "w+")
    f.write("MXNet (+lwTENSOR),PyTorch,TensorFlow\n")

    for test in tests:
        mx, pyt, tensorf = run_test(test)
        f.write(f"{mx:.2f},{pyt:.2f},{tensorf:.2f}\n")
    f.close()
    print(f"Successfully created {fo}.csv")









