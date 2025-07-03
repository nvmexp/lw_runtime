#!/usr/bin/elw python

# Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
#
# LWPU CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from LWPU CORPORATION is strictly prohibited.

# Examples of running:
# python ../util/ldpc/ldpc_perf_collect.py --use_fp16
# python ../util/ldpc/ldpc_perf_collect.py -m latency -f
# python ../util/ldpc/ldpc_perf_collect.py --mode ber -i ldpc_BG1_K8448_SNR%g_800_p_m.h5 --num_parity 5 -n 32 --use_fp16
# python ../util/ldpc/ldpc_perf_collect.py --mode ber -i ldpc_BG2_K3840_SNR%g_800_p_m.h5 -g 2 --num_parity 7 -n 32 --use_fp16 --min_snr 1 --max_snr 3.25 --normalization 0.8125 -o ldpc_ber_7.txt
# python ../util/ldpc/ldpc_perf_collect.py --mode ber -i ldpc_BG1_K8448_SNR%g_800_p_m.h5 --num_parity 46 -n 32 --use_fp16 --min_snr -2.5 --max_snr -1 --normalization 0.6875 -o ldpc_ber_46.txt

import os
import subprocess
import re
import argparse

#***********************************************************************
# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--use_fp16",  action="store_true",                             help="Use fp16 input LLR values")
parser.add_argument("-g", "--bg",        type=int,   default=1,                             help="Base graph (1 or 2)", choices=[1, 2])
parser.add_argument("-i", "--input_file",            default="ldpc_BG1_K8448_SNR8_80.h5",   help="Input file (for latency mode) or input file SNR format string (for ber mode)")
parser.add_argument("-m", "--mode",                  default="latency",                     help="perf collection mode ('latency' or 'ber')")
parser.add_argument("-n", "--num_iter",  type=int,   default=10,                            help="Number of LDPC iterations")
parser.add_argument("-o", "--output_file",                                                  help="Output file name")
parser.add_argument("-r", "--num_runs",  type=int,   default=100,                           help="Number of runs to time (latency mode only)")
parser.add_argument("-w", "--num_words", type=int,   default=1,                             help="Number of input codewords")
parser.add_argument("--exe_dir",                     default="./examples/error_correction", help="Path to lwphy_ex_ldpc exelwtable")
parser.add_argument("--exe",                         default="lwphy_ex_ldpc",               help="Name of exelwtable")
parser.add_argument("--data_dir",                    default="../../lwPHY_data",            help="Path to input HDF5 files")
parser.add_argument("--num_parity",      type=int,   default=4,                             help="Number of parity check nodes (ber mode only)")
parser.add_argument("--min_snr",         type=float, default=4.0,                           help="Minimum SNR (ber mode only)")
parser.add_argument("--max_snr",         type=float, default=6.25,                          help="Maximum SNR (ber mode only)")
parser.add_argument("--snr_step",        type=float, default=0.25,                          help="SNR step size (ber mode only)")
parser.add_argument("--normalization",   type=float,                                        help="Min-sum normalization factor")
args = parser.parse_args()

#***********************************************************************
# gen_cmd()
# Returns a string representation of the command to execute, given a
# dictionary of parameters
def gen_cmd(param_dict):
    return ('%s -i %s -p %i -n %i -r %i %s -g %i %s %s' %
            (os.path.join(param_dict['exe_dir'], param_dict['exe']),
             param_dict['input_file'],
             param_dict['num_parity'],
             param_dict['num_iter'],
             param_dict['num_runs'],
             ('-m %f' % param_dict['normalization']) if 'normalization' in param_dict else '',
             param_dict['BG'],
             '-f' if param_dict['use_fp16'] else '',
             ('-w %d' % param_dict['num_words']) if 'num_words' in param_dict else ''))

max_num_parity = [-1, 46, 42]

#re_str = r'Average \(([-+]?\d*\.\d+|\d+) runs\) elapsed time in usec = , throughput =  Gbps'
re_latency_str = r'Average \((\d+) runs\) elapsed time in usec = (.+), throughput = (.+) Gbps'

# bit error count = 0, bit error rate (BER) = (0 / 8448) = 0.00000e+00, block error rate (BLER) = (0 / 1) = 0.00000e+00
#re_ber_str = r'bit error count = (\d+), bit error rate (BER) = \(\d+ / \d+\) = .+, block error rate (BLER) = \(\d+ / \d+\) = .+'
re_ber_str = r'bit error count = \d+, bit error rate \(BER\) = \((\d+) / (\d+)\) = (.+), block error rate \(BLER\) = \((\d+) / (\d+)\) = (.+)'

#***********************************************************************
# get_results()
def get_results(params):
    #-------------------------------------------------------------------
    # Generate the command line string
    cmd = gen_cmd(params)
    print(cmd)
    #-------------------------------------------------------------------
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    res = {}
    while True:
        line = proc.stdout.readline().decode('ascii')
        #if line != b'':
        if line:
            m = re.search(re_latency_str, line)
            if m:
                #print(line)
                #print(m.group(1), m.group(2), m.group(3))
                res['num_parity'] = params['num_parity']
                res['num_runs']   = int(m.group(1))
                res['latency']    = float(m.group(2))
                res['throughput'] = float(m.group(3))
            else:
                m = re.search(re_ber_str, line)
                if m:
                    res['bit_error_count']   = int(m.group(1)) 
                    res['total_bits']        = int(m.group(2)) 
                    res['bit_error_rate']    = float(m.group(3)) 
                    res['block_error_count'] = int(m.group(4)) 
                    res['total_blocks']      = int(m.group(5)) 
                    res['block_error_rate']  = float(m.group(6)) 
        else:
            break
    if not res:
        raise RuntimeError('No output results string found for command "%s"' % cmd)
    print('num_parity=%i, latency=%.1f, throughput=%.2f, ber=%e, bler=%e' %
          (res['num_parity'], res['latency'], res['throughput'], res['bit_error_rate'], res['block_error_rate']))
    return res

#***********************************************************************
# do_latency_mode()
def do_latency_mode():
    results = []

    params = {'exe'        : args.exe,
              'exe_dir'    : args.exe_dir,
              'BG'         : args.bg, 
              'input_file' : os.path.join(args.data_dir, args.input_file),
              'num_parity' : 0,
              'num_iter'   : args.num_iter,
              'num_runs'   : args.num_runs,
              'use_fp16'   : args.use_fp16,
              'num_words'  : args.num_words}
    if args.normalization:
        params['normalization'] = args.normalization
    # Iterate over code rates (i.e. number of parity nodes)
    for mb in range(4, max_num_parity[params['BG']] + 1):
        params['num_parity'] = mb
        results.append(get_results(params))

    for r in results:
        print('%2d %7.1f %e %e' % (r['num_parity'], r['latency'], r['bit_error_rate'], r['block_error_rate']))
    if args.output_file:
        with open(args.output_file, 'w') as f:
            for r in results:
                f.write('%2d %7.1f %e %e\n' % (r['num_parity'], r['latency'], r['bit_error_rate'], r['block_error_rate']))

#***********************************************************************
# do_ber_mode()
def do_ber_mode():
    results = []
    # Don't specify the number of words - use the number in the input file.
    # Do only 1 run - no need to average over multiple iterations for timing.
    params = {'exe'        : args.exe,
              'exe_dir'    : args.exe_dir,
              'BG'         : args.bg, 
              'input_file' : '',
              'num_parity' : args.num_parity,
              'num_iter'   : args.num_iter,
              'num_runs'   : 1,
              'use_fp16'   : args.use_fp16}
    if args.normalization:
        params['normalization'] = args.normalization

    # Iterate over an SNR range that is appropriate for the given number
    # of parity check nodes
    snr = args.min_snr
    while snr <= args.max_snr:
        params['input_file'] = os.path.join(args.data_dir, args.input_file % (snr))
        res = get_results(params)
        res['snr'] = snr
        results.append(res)
        snr = snr + args.snr_step

    for r in results:
        print('%.2f %e %e' % (r['snr'], r['bit_error_rate'], r['block_error_rate']))
    if args.output_file:
        with open(args.output_file, 'w') as f:
            for r in results:
                f.write('%.2f %e %e\n' % (r['snr'], r['bit_error_rate'], r['block_error_rate']))

if args.mode == 'latency':
    do_latency_mode()
elif args.mode == 'ber':
    do_ber_mode()
else:
    raise RuntimeError('Invalid mode: "%s"' % args.mode)
