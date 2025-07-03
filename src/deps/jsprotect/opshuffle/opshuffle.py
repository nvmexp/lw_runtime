#!/usr/bin/elw python3

# stdin:  jsopcode.tbl
# output: shuffled opcode table

import random
import sys

opcode_table = list(filter(lambda line: line.startswith("OPDEF("), sys.stdin.readlines()))

opcodes = list(range(len(opcode_table)))
random.shuffle(opcodes)

for index in range(len(opcodes)):
    line = opcode_table[index].split(",", 2)
    print(line[0] + "," + str(opcodes[index]) + "," + line[2])
