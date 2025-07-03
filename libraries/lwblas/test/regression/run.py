import json                                                       
import os
from argparse import ArgumentParser                                   
from subprocess import Popen, DEVNULL
from time import sleep

# This script can be used to ilwoke lwtensorTest on a system with multiple GPUs
# to run parallel and launch new tasks when previous tasks complete.
# It is a more sophisticated regression.sh

def main(args):
    os.elwiron['LWTENSOR_TEST_VERBOSE'] = '1'
    os.elwiron['LWTENSOR_DISABLE_LWBLAS'] = '1'
    os.elwiron['LD_LIBRARY_PATH'] = os.path.join(args.binary_root, 'lib', '11') + ':' + os.elwiron.get('LD_LIBRARY_PATH', '')
    binary = os.path.join(args.binary_root, 'bin', 'lwtensorTest')
    tasks = json.load(open(args.testlist))
    gpus = list(range(args.gpu_count))
    active_tasks = []
    while tasks:
        while gpus:
            task = tasks.pop(0)
            name = task.get('name', task['file'].split('.sh')[0])
            task['name'] = name
            options = task.get('options', '').split()
            filename = os.path.join(args.output_dir, args.gpu_tag + '_' + name + '_' + args.binary_tag + '.log')
            if os.path.exists(filename):
                content = open(filename).readlines()
                if len(content) > 0 and 'PASSED' in content[-1]:
                    print('Skipping ' + name)
                    continue
            gpu = gpus.pop(0)
            task['gpu'] = gpu
            print('Starting ' + name + ' on ' + str(gpu))
            output = open(filename, 'w+')
            task['output'] = output
            process_args = [binary, '-d%d' % gpu, '-file', task['file']] + options
            task['process'] = Popen(process_args, stdout=output, stderr=output, stdin=DEVNULL)
            active_tasks.append(task)
        while not gpus:
            for idx, task in enumerate(active_tasks):
                if task['process'].poll() is not None:
                    task = active_tasks.pop(idx)
                    print('Completed {} with code {} on {}'.format(task['name'], task['process'].returncode, task['gpu']))
                    gpus.append(task['gpu'])
                    task['output'].close()
                    break
            if not gpus:
                sleep(60)
    for task in active_tasks:
        print('Waiting for {} on {}'.format(task['name'], task['gpu']))
        task['process'].wait()
        task['output'].close()
        

if __name__ == '__main__':                                        
    p = ArgumentParser()                                 
    p.add_argument('--testlist', type=str)                        
    p.add_argument('--gpu-tag', type=str)                         
    p.add_argument('--gpu-count', type=int)                       
    p.add_argument('--binary-root', type=str)                     
    p.add_argument('--binary-tag', type=str)                      
    p.add_argument('--output-dir', type=str)
    args = p.parse_args()
    main(args)
