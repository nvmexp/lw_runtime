# colwert lwblas test name to the layers can be read from lwblas_run.py

option_list = ['-P',   '-Pin',  '-Pout',  '-R',   '-m',
               '-n',   '-k',    '-b',     '-K',   '-ta',
               '-tb',  '-Ha',   '-Hb',    '-Q',   '-ocrc',
               '-csv', '-x',    '-y',     '-a',   '-A',
               '-B',   '-T',    '-lda',   '-ldb', '-u',
               '-r',   '-U',    '-S',     '-c',   '-f',
               '-M',   '-e',    '-g',     '-s',   '-sing',
               '-V',   '-E',    '-Y',     '-ldc', '-transA',
               '-na',  '-nb',   '-v',     '-z',   '-transB',
               '-p',   '-pm',   '-ps',    '-h',   '-backdoor',
               '-C',   '-o',    '-G',     '-t',   '-algorithm',
               '-N',   '-H',    '-div',   '-d',   '-mathMode',
               '-D',   '-MT']

def testListRead(test_name_str):
    """ 
    input: test_item1, test_item2
    return the lines of a file, format likes:
    ['aaa aaa aaa',
     'bbb bbb bbb',
     ...
     ]
    """
    new_lines = []
    if test_name_str:
        lines = test_name_str.split(",")
        for line in lines:
            tmp = line.strip()
            if tmp and (not tmp in new_lines):
                new_lines.append(tmp)
        return new_lines
    else:
        print "No test item is specified"
        return None

def getFlag(string):
    """
    check the flag for a given string, such as '-Ps', return '-P'
    """
    str_len = len(string)
    flag = ''
    for i in range(str_len):
        if string[0:i] in option_list:
            flag = string[0:i]
    if string in option_list:#for -K, no value behind
        flag = string
    if flag:
        return flag
    elif (string == 'lwblasTest'          or 
          string == 'lwblasTest_cnp'      or 
          string == 'lwblasMgTest'        or 
          string == 'lwblasMgTest_static' or 
          string == 'lwblasTest'          ):
       return None 
    else:
        print 'Error: %s not found. If the option is correct, it should be added to the option_list'%string
        return None

def transformTests(line):
    """
    from: lwblasTest -Rsbhbmv -Pc -c50 -c -ocrc (line)
    to  : ['R:sbhbmv', 'P:c', 'c:50', 'c:', 'ocrc:']
    """
    line_list = line.strip().split()

    newLabelLine = []

    for item in line_list:
        flag = getFlag(item)
        tmp = ''
        if flag:
            value = item[len(flag):]
            tmp = flag[1:] + ':' + value
        if tmp and (not tmp in newLabelLine):
            newLabelLine.append(tmp)

    return newLabelLine


def export_layers(newLabelLines, idx):
    """
    input: ['R:sbhbmv', 'P:c', 'c:50', 'c:', 'ocrc:']
    output: 'R:sbhbmv * P:c * c:50 * c: * ocrc:'
    """
    test_str = '"%d_TestName" = '%idx
    for item in newLabelLines[:-1]:
        test_str += item
        test_str += ' * '
    test_str += newLabelLines[-1]
    test_str += '\n'
    return test_str


def testToLayer(strings):
    """
    input: "lwblasTest -Rgemm -m128, lwblasTest -Rgemm -m127"
    output: "0_TestName = R:gemm * m:128\n1_TestName = R:gemm * m:127\n"
    """
    test_lines = testListRead(strings)
    layers = ''
    for idx, test_line in enumerate(test_lines):
        test_list = transformTests(test_line)
        layer = export_layers(test_list, idx)
        layers += layer
    return layers

