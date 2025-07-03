import sys
from collections import defaultdict

# Report: Max 90% 25% 5% 1% 0% -1% -5% -25% -90% Min Speedup
# Average speedup, Median speedup

def read_file(f, result, name):
    for l in f:
        if 'GFLOPS' not in l: continue
        if '-algo+1000' in l: continue
        l = l.split('|')
        result[l[0]][name] = float(l[1].strip().split(' ')[0].split(':')[1])

def compute_speedup(result, a, b):
    for elem in list(result):
        try:
            if result[elem][a] > result[elem][b]:
                result[elem][a + '/' + b] = result[elem][a] / result[elem][b] - 1
            else:
                result[elem][a + '/' + b] = - result[elem][b] / result[elem][a] + 1
        except:
            del result[elem]

def sort_dict(result, by):
    return sorted(result.items(), key=lambda a: a[1][by])

def summarize(result, by):
    num = len(result) - 1
    # boxplot
    for f in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]:
        idx = int(f * num)
        print(f'box {f} {result[idx][1][by]:.2f}', end=' ')
    print()
    # avg
    avg = sum([elem[1][by] for elem in result]) / len(result)
    print(f'avg {avg:.2f}')
    # number > 0, < 0, > 0.01, < -0.01, > 0.05, < -0.05
    for f in [0, 0.01, 0.05, 0.1, 0.5, 1]:
        count = len([elem for elem in result if elem[1][by] > f])
        print(f'thr >+{f} {count}', end=' ')
        count = len([elem for elem in result if elem[1][by] < -f])
        print(f'thr <-{f} {count}', end=' ')
    print()
    # top 3, low 3
    for elem in result[:3]:
        print(f'top | {elem[0]} | {elem[1]}')
    for elem in result[-3:]:
        print(f'bot | {elem[0]} | {elem[1]}')

result = defaultdict(dict)
read_file(open(sys.argv[1], 'r'), result, 'a')
read_file(open(sys.argv[2], 'r'), result, 'b')
compute_speedup(result, 'a', 'b')
result = sort_dict(result, 'a/b')
summarize(result, 'a/b')
