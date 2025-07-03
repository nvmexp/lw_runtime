#!/usr/bin/elw python

# Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
#
# LWPU CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from LWPU CORPORATION is strictly prohibited.

import os
import fnmatch
import tarfile
import sys
#import itertools (not in Python 2)


NUM_SETS = 8   # Sets in 5G NR are indexed from 0 to 7

Z_iLS = [[ 2,  4,  8,  16, 32,   64, 128, 256],  # 0
         [ 3,  6, 12,  24, 48,   96, 192, 384],  # 1
         [ 5, 10, 20,  40, 80,  160, 320     ],  # 2
         [ 7, 14, 28,  56, 112, 224          ],  # 3
         [ 9, 18, 36,  72, 144, 288          ],  # 4
         [11, 22, 44,  88, 176, 352          ],  # 5
         [13, 26, 52, 104, 208               ],  # 6
         [15, 30, 60, 120, 240               ]]  # 7

def find_set_index_for_Z(Z):
    set_idx = -1
    for idx, Zset in enumerate(Z_iLS):
        if Z in Zset:
            set_idx = idx
            break
    return set_idx

def generate_row_degree_hist(hist_bg1, hist_bg2):
    import matplotlib.pyplot as plt
    # Set the default background color to white
    plt.rcParams['figure.facecolor'] = 'white'
    hist_bg2_extended = hist_bg2 + [0] * (len(hist_bg1) - len(hist_bg2))
    lwm_hist = [t[0] + t[1] for t in zip(hist_bg1, hist_bg2_extended)]
    
    fig = plt.figure(figsize=(5,8))

    #itertools.accumulate in Python 3
    #print(hist_bg1)
    prefix_sum = [sum(hist_bg1[:k+1]) for k in range(len(hist_bg1))]
    #print(prefix_sum)
    num_rows = prefix_sum[-1]
    #print('num_rows = ', num_rows)
    degree_num_edges = [t[0] * t[1] for t in zip(hist_bg1, range(len(hist_bg1)))]
    #print(degree_num_edges)
    num_edges = sum(degree_num_edges)
    lt_half = [v for v in prefix_sum if v <= (num_rows / 2)]
    #print(lt_half)
    #median_degree = next(v for i,v in enumerate(prefix_sum) if v >= num_edges / num_rows)
    median_degree = len(lt_half)
    if lt_half[-1] == num_rows / 2:
        median_degree = median_degree + 0.5
    print('BG1: median = %d, mean = %.1f' % (median_degree, float(num_edges) / num_rows))
    ax  = fig.add_subplot(311)
    ax.grid(zorder=0)
    plt.bar(range(len(hist_bg1)), height = hist_bg1, align='center')
    ax.set_axisbelow(True)
    plt.xticks(range(2, 22, 2))
    plt.yticks(range(0, 30, 2))
    plt.xlim([0, 20])
    plt.ylabel('BG1')
    plt.text(14, 20, 'median = %d\nmean = %.1f' % (median_degree, float(num_edges) / num_rows), bbox=dict(facecolor='white'))

    prefix_sum = [sum(hist_bg2[:k+1]) for k in range(len(hist_bg2))]
    num_rows = prefix_sum[-1]
    degree_num_edges = [t[0] * t[1] for t in zip(hist_bg2_extended, range(len(hist_bg2_extended)))]
    num_edges = sum(degree_num_edges)
    #print(prefix_sum)
    #print(sum(num_edges))
    lt_half = [v for v in prefix_sum if v <= (num_rows / 2)]
    #print('num_rows = ', num_rows)
    #print(lt_half)
    #median_degree = next(v for i,v in enumerate(prefix_sum) if v >= num_edges / num_rows)
    median_degree = len(lt_half)
    if lt_half[-1] == num_rows / 2:
        median_degree = median_degree + 0.5
    print('BG2: median = %d, mean = %.1f' % (median_degree, float(num_edges) / num_rows))
    ax  = fig.add_subplot(312)
    plt.bar(range(len(hist_bg2_extended)), height = hist_bg2_extended, align='center')
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    plt.xticks(range(2, 22, 2))
    plt.yticks(range(0, 30, 2))
    plt.xlim([0, 20])
    plt.ylabel('BG2')
    plt.text(14, 20, 'median = %d\nmean = %.1f' % (median_degree, float(num_edges) / num_rows), bbox=dict(facecolor='white'))
       
    ax  = fig.add_subplot(313)
    plt.bar(range(len(lwm_hist)), height = lwm_hist, align='center')
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    plt.xticks(range(2, 22, 2))
    plt.yticks(range(0, 30, 2))
    plt.xlim([0, 20])
    plt.ylabel('BG1 + BG2')
    
    plt.show(block=False)
    plt.savefig('row_degree_histogram.png')
    plt.close(fig)
    
#def generate_plots():
    #import matplotlib.pyplot as plt
    # Set the default background color to white
    #plt.rcParams['figure.facecolor'] = 'white'
    #BG_edges_per_N = {1 : [], 2 : []}
    #BG_rates       = {1 : [], 2 : []}

    #for (BG_idx, max_mb, Kb) in [(1, 47, 22), (2, 43, 10)]:
    #    BG = BG_dict[BG_idx][0]
    #    for mb in range(4, max_mb):
    #        num_cols = Kb + mb
    #        rate = Kb / float(num_cols - 2)
    #        num_rows = BG.num_rows - (BG.num_columns - num_cols)
    #        row_list = BG.row_list[0:num_rows]
    #        N = num_cols - 2
    #        num_edges = len([(t[0] + 1) for row in row_list for t in row])
    #        BG_edges_per_N[BG_idx].append(float(num_edges) / N)
    #        BG_rates[BG_idx].append(rate)
    #        print('mb = %d, Nrows = %d, Ncols = %d, K = %d, N = %d, rate = %.2f, num_edges = %d, num_edges_per_N = %.2f' % (mb, num_rows, num_cols, Kb, N, rate, num_edges, float(num_edges) / N))

    #fig        = plt.figure(figsize=(10,5))
    #ax         = fig.add_subplot(111)
    #plt.plot(BG_rates[1], BG_edges_per_N[1], 'bo',
    #         BG_rates[2], BG_edges_per_N[2], 'go',
    #         [0.5], [86.0 / 24.0],           'ro')
    #ax.set_title('')
    #ax.set_xlabel('Code Rate')
    #ax.set_ylabel('Edges / Coded Bit')
    #ax.set_ylim(bottom = 0)
    #ax.grid()
    #ax.legend(['BG1', 'BG2', '802.11n (1/2)'], numpoints=1, loc='lower right')
    #plt.show(block=False)
    #plt.savefig('code_rate_edges_per_bit.png')
    #plt.close(fig)

########################################################################
# BaseGraph
class BaseGraph(object):
    """Class to represent a 5G NR Base Graph (either BG1 or BG2). The base
    graph contains the positions of the non-zero permutation matrices.
    However, the actual permutation values are specific to the different
    sets, as described in 3GPP 38.212, Tables 5.3.2-1, -2, and -3.
    """
    def __init__(self, index, row_list):
        self.BG_index = index
        # If given lists of (column, shift) tuples, extract the column.
        # Otherwise, use the list values as column values.
        if isinstance(row_list[0][0], tuple):
            self.row_list = []
            for row in row_list:
                self.row_list.append([t[0] for t in row])
        else:
            self.row_list = row_list
        self.num_rows       = len(self.row_list)
        # Find the maximum column index from ALL tuples taken from ALL rows
        self.num_columns    = max([(t + 1) for row in self.row_list for t in row])
        self.num_edges      = len([(t + 1) for row in self.row_list for t in row])
        self.max_row_degree = max([len(row) for row in self.row_list])
        self.min_row_degree = min([len(row) for row in self.row_list])
        self.col_list       = [[] for col in range(self.num_columns)]
        for row_idx, row in enumerate(self.row_list):
            for col in row:
                #print('Adding row %d to col %d\n' % (row, col))
                self.col_list[col].append(row_idx)
        # Column index list: A list of lists, one for each column. 
        # List elements are tuples, with the first value being the
        # row of the element in that column, and the second value
        # being the index of that element within its row. For example:
        #
        #    1    0    0    1    0     0     1
        #    0    0    1    1    1     0     1
        #    0    1    1    0    0     1     0
        #    0    0    0    0    0     1     0
        #
        #   COL 0: (0, 0)    (element in row 0 is also the 0th element in that row)
        #   COL 1: (2, 0)
        #   COL 2: (1, 0) (2, 1)
        #   COL 3: (0, 1) (1, 1)
        #   COL 4: (1, 2)
        #   COL 5: (2, 2) (3, 0)
        #   COL 6: (0, 2) (1, 3)
        self.col_index_list       = [[] for col in range(self.num_columns)]
        for row_idx, row in enumerate(self.row_list):
            for idx, col in enumerate(row):
                #print('Adding row %d to col %d\n' % (row, col))
                self.col_index_list[col].append((row_idx, idx))
        self.min_col_degree = min([len(col) for col in self.col_list])
        self.max_col_degree = max([len(col) for col in self.col_list])
        self.col_degree_hist = [0 for cnt in range(self.max_col_degree + 1)]
        for col_idx, col in enumerate(self.col_list):
            #print('Adding 1 to length %d for index %d\n' % (len(col), col_idx))
            self.col_degree_hist[len(col)] = self.col_degree_hist[len(col)] + 1
        self.row_degree_hist = [0 for cnt in range(self.max_row_degree + 1)]
        for row in self.row_list:
            self.row_degree_hist[len(row)] = self.row_degree_hist[len(row)] + 1
    def row(self, idx):
        return self.row_list[idx]
    def __str__(self):
        str = os.linesep
        str = str + 'BG%d: num_rows = %d, num_cols = %d, num_edges = %d, min_row_degree = %d, max_row_degree = %d, max_col_degree = %d' % \
                    (self.BG_index, self.num_rows, self.num_columns, self.num_edges, self.min_row_degree, self.max_row_degree, self.max_col_degree) + os.linesep
        str = str + self.adj_graph_str()
        str = str + self.position_table_str()
        return str
    def __repr__(self):
        return str(self)
    def position_table_str(self):
        """Return a string representation of the position table"""
        str = os.linesep
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # make table of positions for each row
        str = str + '     ' + ''.join([('%3d ' % i) for i in range(self.max_row_degree)]) + os.linesep
        str = str + '     ' + ''.join(['-' for i in range(4 * self.max_row_degree)])      + os.linesep
        for (row_idx, row) in enumerate(self.row_list):
            str = str + ('%02d : ' % row_idx)
            row_extended = row + [-1 for i in range(self.max_row_degree - len(row))]
            for col in row_extended:
                str = str + ('%3d ' % col)
            str = str + os.linesep
        return str
    def filled_position_table(self):
        """Return a list of lists with non-zero matrix positions, using -1 to fill the matrix"""
        pos = []
        for row in self.row_list:
            row_extended = row + [-1 for i in range(self.max_row_degree - len(row))]
            pos.append(row_extended)
        return pos
    def adj_graph_str(self):
        """Return a string representation of the adjacency graph"""
        str = os.linesep
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Print the adjacency graph with a '*' for each non-zero matrix
        col_labels = [' ' for i in range(self.num_columns * 2)]
        col_labels[0] = '0'
        for i in range(self.num_columns // 10):
            col_labels[(i + 1) * 20 - 1] = '%d' % (i + 1)
            col_labels[(i + 1) * 20]     = '0'
        str = str + '     ' + ''.join(col_labels) + os.linesep
        for (row_idx, row) in enumerate(self.row_list):
            elems = [' ' for i in range(self.num_columns)]
            for col in row:
                elems[col] = '*'
            str = str + ('%02d : ' % row_idx) + ' '.join(elems) +  (' (%d)' % len(row)) + os.linesep
        str = str + os.linesep
        return str
    def get_csr_row_array(self):
        row_array = [0]
        for row in self.row_list:
            row_array.append(len(row) + row_array[-1])
        return row_array
    def get_csr_columns(self):
        col_array = []
        for row in self.row_list:
            col_array.append(row)
        return col_array
    def get_csr_col_array(self):
        col_array = []
        for row in self.row_list:
            col_array.extend(row)
        return col_array
    def get_csc_col_array(self):
        col_array = [0]
        for col in self.col_list:
            col_array.append(len(col) + col_array[-1])
        return col_array
    def get_csc_rows(self):
        row_array = []
        for col in self.col_list:
            row_array.append(col)
        return row_array
    def get_csc_row_array(self):
        row_array = []
        for col in self.col_list:
            row_array.extend(col)
        return row_array
    def get_csc_row_pairs(self):
        row_pair_array = []
        for col in self.col_index_list:
            row_pair_array.append(col)
        return row_pair_array
    def get_row_degrees(self):
        return [len(row) for row in self.row_list]

########################################################################
# SetBaseGraph
class SetBaseGraph(BaseGraph):
    """Class to represent a Low Density Parity Check (LDPC) Matrix"""
    def __init__(self, BG_index, set_index, set_row_list):
        super(SetBaseGraph, self).__init__(BG_index, set_row_list)
        self.set_index      = set_index
        self.set_row_list   = set_row_list  # list of lists of tuples with (column, shift)
        self.max_shift      = max([t[1] for row in self.set_row_list for t in row])
        self.set_col_list   = [[] for col in range(self.num_columns)] # list of lists of tuples with (row, shift)
        for set_row_idx, set_row in enumerate(self.set_row_list):
            for t in set_row:
                col = t[0]
                #print('Adding row %d to col %d\n' % (set_row_idx, col))
                self.set_col_list[col].append((set_row_idx, t[1]))
    def __str__(self):
        str = os.linesep
        #str = str + '-----' + ''.join(['-' for i in range(self.num_columns * 2)]) + os.linesep
        str = str + 'BG%dS%d: num_rows = %d, num_cols = %d, num_edges = %d, max_row_degree = %d, max_shift = %d' % \
                    (self.BG_index, self.set_index, self.num_rows, self.num_columns, self.num_edges, self.max_row_degree, self.max_shift)
        str = str + self.position_shift_table_str()
        return str
    def __repr__(self):
        return str(self)
    def position_shift_table_str(self):
        str = os.linesep
        # Create a string with a table of positions and shift values
        str = str + '     ' + ''.join([('%3d ' % i) for i in range(self.max_row_degree)])
        str = str + '          ' + ''.join([('%3d ' % i) for i in range(self.max_row_degree)]) + os.linesep
        str = str + '     ' + ''.join(['-' for i in range(4 * self.max_row_degree)])
        str = str + '          ' + ''.join(['-' for i in range(4 * self.max_row_degree)]) + os.linesep
        for (row_idx, row) in enumerate(self.set_row_list):
            str = str + ('%02d : ' % row_idx)
            row_extended = row + [(-1, -1) for i in range(self.max_row_degree - len(row))]
            for col in row_extended:
                str = str + ('%3d ' % col[0])
            str = str + '          '
            for col in row_extended:
                str = str + ('%3d ' % col[1])
            str = str + os.linesep
        return str
    def filled_shift_table(self):
        """Return a list of lists with shifts for non-zero matrices, using -1 to fill the matrix"""
        shift_table = []
        for row in self.set_row_list:
            row_extended = [t[1] for t in row] + [-1 for i in range(self.max_row_degree - len(row))]
            shift_table.append(row_extended)
        return shift_table
    def filled_shift_Z_table(self, Z):
        """Return a list of lists with the tuple (column, shift modulo Z) for non-zero matrices, using (-1, -1) to fill the matrix"""
        shift_table = []
        for row in self.set_row_list:
            row_extended = [(t[0], t[1] % Z) for t in row] + [(-1, -1) for i in range(self.max_row_degree - len(row))]
            shift_table.append(row_extended)
        return shift_table
    def position_shift_Z(self, Z):
        shift_array = []
        for set_row in self.set_row_list:
            shift_array.append([s[1] % Z for s in set_row])
        return shift_array
    def position_shift_Z_columns(self, Z):
        shift_array = []
        for set_col in self.set_col_list:
            shift_array.append([s[1] % Z for s in set_col])
        return shift_array
    def position_shift_csr_array(self, Z):
        shift_array = []
        for set_row in self.set_row_list:
            shift_array.extend([s[1] % Z for s in set_row])
        return shift_array

def load_from_files(path = 'NR_5G_LDPC_BaseGraphs.tar.gz'):
    BG_dict = {}
    SG_dict = {}
    tfile      = tarfile.open('NR_5G_LDPC_BaseGraphs.tar.gz', 'r:gz')
    for BG in [1, 2]:
        #-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        # Get a list of files for this base graph and sort it
        #BG_file_list = [file for file in os.listdir(path) if fnmatch.fnmatch(file, 'BG%d_row*.txt' % BG) ]
        #BG_file_list.sort()
        file_names = [f for f in tfile.getnames() if f.startswith('BG%d' % BG)]
        file_names.sort()
        #-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        # List of lists of column indices - one for each row
        BG_row_list = []
        # List of (list of list of tuples) - one list for each row,
        # tuple is (column index, permutation)
        BG_set_list = [[] for i in range(NUM_SETS)]
        #-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        # Iterate over base graph files: We have 1 file for each row
        for name in file_names:
            # Create a list of tuples for each set in the BG
            set_row_list = [[] for i in range(NUM_SETS)]
            row_list     = []
            row_file = tfile.extractfile(name)
            # Each line has the following format:
            # col_index set_0 set_1 set_2 set_3 set_4 set_5 set_6 set_7
            for line in row_file.readlines():
                values = [int(val) for val in line.split()]
                if len(values) != NUM_SETS + 1:
                    raise Exception('Incorrect number of values (%d) found in row for %s' % (len(values), file))
                # Add the value for each set to the appropriate row list
                row_list.append(values[0])
                for iLS in range(NUM_SETS):
                    set_row_list[iLS].append((values[0], values[iLS + 1]))
            # Add the base graph row list
            BG_row_list.append(row_list)
            # Add each row lists to the appropriate set
            for iLS in range(NUM_SETS):
                BG_set_list[iLS].append(set_row_list[iLS])
        #-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        BG_dict[BG] = BaseGraph(BG, BG_row_list)
        SG_dict[BG] = [SetBaseGraph(BG, idx, row_list) for (idx, row_list) in enumerate(BG_set_list)]
    return (BG_dict, SG_dict)
    
(BG, SG) = load_from_files()

#print(type(BG_dict[1][0]).__name__)
#print(BG_dict[1][0])
#print(BG_dict[1][1])
#print(BG_dict[1][0])
print(BG[1])
print(BG[2])
#print(SG_dict)
print(SG[1][0])
print(SG[2][0])
#generate_plots()

lw_copyright = """/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */
"""

def define_2D_array(f, desc):
    f.write('%s %s[%s][%s] = \n{\n' % (desc['type'], desc['name'], desc['num_rows'], desc['num_columns']))
    data = desc['data']
    num_rows = len(data)
    num_columns = len(data[0])
    #print('data.num_rows = %d, data.num_columns = %d' % (num_rows, num_columns))
    for idx_row in range(num_rows):
        f.write('    { ')
        for idx_col in range(num_columns):
            f.write('%3d' % data[idx_row][idx_col])
            if idx_col != (num_columns - 1):
                f.write(', ')
            else:
                f.write('}')
        if idx_row != (num_rows - 1):
            f.write(',')
        f.write('\n')
    f.write('};\n')

def define_1D_array(f, desc):
    f.write('%s %s[%s] = \n{' % (desc['type'], desc['name'], desc['num_elements']))
    data = desc['data']
    num_elements = len(data)
    for idx in range(num_elements):
        if 0 == idx % 10:
            f.write('\n    ')
        f.write(desc['format'] % data[idx])
        if idx != (num_elements - 1):
            f.write(',')
        else:
            f.write('\n')
    f.write('};\n')

def write_columns(f, desc):
    f.write('%s %s[%s] = \n{\n' % (desc['type'], desc['name'], desc['num_elements']))
    data = desc['data']
    num_rows = len(data)
    for row_idx in range(num_rows):
        num_columns = len(data[row_idx])
        for column_idx, column in enumerate(data[row_idx]):
            f.write(desc['format'] % column)
            if (column_idx != num_columns - 1) or (row_idx != num_rows - 1):
                f.write(', ')
            else:
                f.write(' ')
        f.write('/* row %2d */\n' % row_idx)
    f.write('};\n\n')

def write_rows(f, desc):
    f.write('%s %s[%s] = \n{\n' % (desc['type'], desc['name'], desc['num_elements']))
    data = desc['data']
    num_cols = len(data)
    for col_idx in range(num_cols):
        num_rows = len(data[col_idx])
        for row_idx, row in enumerate(data[col_idx]):
            f.write(desc['format'] % row)
            if (row_idx != num_rows - 1) or (col_idx != num_cols - 1):
                f.write(', ')
            else:
                f.write(' ')
        f.write('/* col %2d */\n' % col_idx)
    f.write('};\n\n')

#-----------------------------------------------------------------------
# write_Z_graph()
#
# Writes a filled table assuming the following structures:
# struct column_info_t
# {
#     int16_t index;
#     int16_t shift;
# };
# struct bg1_row_t
# {
#     int16_t       row_degree;
#     column_info_t column_info[BG1_MAX_ROW_DEG];
# };
# struct bg1_Z_graph_t
# {
#     bg1_row_t rows[BG1_M];
# };
def write_Z_graph(f, desc):
    f.write('%s %s = \n{\n    {\n' % (desc['type'], desc['name']))
    set_graph = desc['data']
    filled_shift_table = set_graph.filled_shift_Z_table(desc['Z'])
    for idx_row in range(len(filled_shift_table)):
        f.write('        {%2i, {' % len(set_graph.row_list[idx_row]) )
        row = filled_shift_table[idx_row]
        for idx_col in range(len(row)):
            f.write('{%3i,%3i}' % (row[idx_col][0], row[idx_col][1]))
            if idx_col != len(row) - 1:
                f.write(', ')
        f.write('}}')
        if idx_row != set_graph.num_rows - 1:
            f.write(',')
        else:
            f.write(' ')
        f.write(' // rows[%d]\n' % idx_row)

    f.write('    }\n};\n\n')

#-----------------------------------------------------------------------
# write_Z_graph_2D()
#
# Writes a filled 2-D array containing elements of the following
# structure:
# struct column_info_t
# {
#     int16_t index;
#     int16_t shift;
# };
def write_Z_graph_2D(f, desc):
    f.write('%s %s[%s][%s] = \n{\n' % (desc['type'], desc['name'], desc['num_elements'][0], desc['num_elements'][1]))
    set_graph = desc['data']
    filled_shift_table = set_graph.filled_shift_Z_table(desc['Z'])
    for idx_row in range(len(filled_shift_table)):
        f.write('    {')
        row = filled_shift_table[idx_row]
        for idx_col in range(len(row)):
            f.write('{%2i,%3i}' % (row[idx_col][0], row[idx_col][1]))
            if idx_col != len(row) - 1:
                f.write(', ')
        f.write('}')
        if idx_row != set_graph.num_rows - 1:
            f.write(',')
        else:
            f.write(' ')
        f.write(' // row %d\n' % idx_row)
    f.write('\n};\n\n')

def write_row_pairs(f, desc):
    f.write('%s %s[%s] = \n{\n' % (desc['type'], desc['name'], desc['num_elements']))
    data = desc['data']
    num_cols = len(data)
    for col_idx in range(num_cols):
        num_rows = len(data[col_idx])
        for row_idx, t in enumerate(data[col_idx]):
            f.write(('{ ' + desc['format'] + ',' + desc['format'] + ' }') % (t[0], t[1]))
            if (row_idx != num_rows - 1) or (col_idx != num_cols - 1):
                f.write(', ')
            else:
                f.write(' ')
        f.write('/* col %2d */\n' % col_idx)
    f.write('};\n\n')

def write_shift_array(f, desc):
    f.write('%s %s[%s] = \n{\n' % \
            (desc['type'], desc['name'], desc['length']));
    for idx_set in range(NUM_SETS):
        f.write('    { // BEGIN iLS = %d' % idx_set)
        f.write('\n')
        set_graph = desc['data'][idx_set]
        shift_table = set_graph.filled_shift_table()
        for idx_row in range(set_graph.num_rows):
            f.write('        {')
            row = shift_table[idx_row]
            for idx_col in range(len(row)):
                f.write('%3d' % row[idx_col])
                if idx_col != len(row) - 1:
                    f.write(', ')
            f.write('}')
            if idx_row != set_graph.num_rows - 1:
                f.write(',')
            else:
                f.write(' ')
            f.write(' // row %d\n' % idx_row)
        f.write('    }')
        if idx_set != (NUM_SETS - 1):
            f.write(',')
        else:
            f.write(' ')
        f.write(' // END iLS = %d' % idx_set)
        f.write('\n')
    f.write('};\n\n');

def write_shift_for_each_Z(f, desc):
    f.write('%s %s[%s] = \n{\n' % \
            (desc['type'], desc['name'], desc['length']))
    data = desc['data']
    for Z_idx, Z_tuple in enumerate(data):
        iLS     = Z_tuple[0]
        Z_value = Z_tuple[1]
        Z_shift = Z_tuple[2]
        if Z_value >= desc['min']:
            f.write('    { /* iLS = %d, Z = %d */\n' % (iLS, Z_value))
            num_rows = len(Z_shift)
            for row_idx, row in enumerate(Z_shift):
                f.write('    ')
                num_columns = len(row)
                for col_idx, col in enumerate(row):
                    f.write(desc['format'] % col)
                    if (col_idx != num_columns) or (row_idx != num_rows):
                        f.write(', ')
                f.write('\n')
            f.write('    }')
            if Z_idx != len(data) - 1:
                f.write(',')
            f.write('\n')
    f.write('};\n')

def write_shift_for_each_Z_column(f, desc):
    f.write('%s %s[%s] = \n{\n' % \
            (desc['type'], desc['name'], desc['length']))
    data = desc['data']
    for Z_idx, Z_tuple in enumerate(data):
        iLS     = Z_tuple[0]
        Z_value = Z_tuple[1]
        Z_shift = Z_tuple[2]
        if Z_value >= desc['min']:
            num_cols = len(Z_shift)
            f.write('    { /* iLS = %d, Z = %d */\n' % (iLS, Z_value))
            for col_idx, col in enumerate(Z_shift):
                f.write('    ')
                num_rows = len(col)
                for row_idx, row in enumerate(col):
                    f.write(desc['format'] % row)
                    if (row_idx != num_rows) or (col_idx != num_cols):
                        f.write(', ')
                f.write('\n')
            f.write('    }')
            if Z_idx != len(data) - 1:
                f.write(',')
            f.write('\n')
    f.write('};\n')

#bg1_csr = BG[1].get_csr_row_array()
#print(bg1_csr)
#bg1_csr = BG[1].get_csr_col_array()
#print(bg1_csr)
#print(BG[1].col_list)
#print(BG[1].col_degree_hist)

#bg1_csr = SG[1][0].position_shift_csr_array(32)
#print(bg1_csr)

#bg1_csr = SG[1][0].position_shift_Z(32)
#print(bg1_csr)
#print(BG[1].row_degree_hist)
#print(BG[2].row_degree_hist)
#generate_row_degree_hist(BG[1].row_degree_hist, BG[2].row_degree_hist)
    
Z_all = [ Z for Z_row in Z_iLS for Z in Z_row ]
max_Z = max(Z_all)
# To conserve constant memory, we don't generate a table for
# all Z values.
Z_gt_32 = [Z for Z_row in Z_iLS for Z in Z_row if Z >= 32]
#print(Z_gt_32)

def write_non_intersecting_rows_adj_list(bg, fname):
    f = open(fname, 'w')
    for row_idx, row in enumerate(bg.row_list):
        can_pair = []
        row_set = set(row)
        for row_cmp_idx, row_cmp in enumerate(bg.row_list):
            if row_cmp_idx == row_idx:
                continue
            if row_set.isdisjoint(row_cmp):
                can_pair.append((row_cmp_idx, len(row) + len(row_cmp)))
        f.write('%d ' % row_idx)
        for p in can_pair:
            #sys.stdout.write('%d(%d) ' % (p[0], p[1]))
            f.write('%d ' % p[0])
        f.write('\n')
    f.close()
    
########################################################################
# Print rows that can be paired together
write_non_intersecting_rows_adj_list(BG[1], 'bg1_non_intersecting.txt')
write_non_intersecting_rows_adj_list(BG[2], 'bg2_non_intersecting.txt')

########################################################################
# Write a source file
fSource = open('nrLDPC.lwh', 'w')
fSource.write(lw_copyright)
fSource.write('\n// Note: This file has been automatically generated.\n\n')
fSource.write('#include <stdint.h>\n')
fSource.write('#include "vector_types.h" /* LWCA header */\n\n')
fSource.write('#define NUM_SETS %d\n' % NUM_SETS)
fSource.write('#define NUM_Z %d\n' % len([z for row in Z_iLS for z in row]))
fSource.write('#define NUM_Z_TABLES %d\n' % len(Z_gt_32))
fSource.write('#define MAX_Z %d\n' % max_Z)
fSource.write('#define BG1_M %d\n' % BG[1].num_rows)
fSource.write('#define BG1_N %d\n' % BG[1].num_columns)
fSource.write('#define BG1_MAX_ROW_DEG %d\n' % BG[1].max_row_degree)
fSource.write('#define BG1_MIN_ROW_DEG %d\n' % BG[1].min_row_degree)
fSource.write('#define BG1_NNZ %d\n' % BG[1].num_edges)
fSource.write('#define BG2_M %d\n' % BG[2].num_rows)
fSource.write('#define BG2_N %d\n' % BG[2].num_columns)
fSource.write('#define BG2_MAX_ROW_DEG %d\n' % BG[2].max_row_degree)
fSource.write('#define BG2_NNZ %d\n\n' % BG[2].num_edges)



########################################################################
#
# BG1
#
########################################################################

#define_2D_array(fSource, {'type'       : '__device__ __constant__ int8_t',
#                          'name'       : 'bg1_pos',
#                          'num_rows'   : 'BG1_M',
#                          'num_columns': 'BG1_MAX_ROW_DEG',
#                          'data'       : BG[1].filled_position_table()})
#define_2D_array(fSource, {'type'       : '__device__ __constant__ int8_t',
#                          'name'       : 'bg2_pos',
#                          'num_rows'   : 'BG2_M',
#                          'num_columns': 'BG2_MAX_ROW_DEG',
#                          'data'       : BG[2].filled_position_table()})

#fSource.write('typedef int16_t bg1_shift_matrix_t[BG1_M][BG1_MAX_ROW_DEG];\n');
#fSource.write('typedef int16_t bg2_shift_matrix_t[BG2_M][BG2_MAX_ROW_DEG];\n');
#write_shift_array(fSource, {'type': '__device__ __constant__ bg1_shift_matrix_t',
#                            'name': 'bg1_shift_array',
#                            'length' : 'NUM_SETS',
#                            'data'   : SG[1]})

fSource.write('/****************************************************************\n');
fSource.write(' * BG1                                                          *\n');
fSource.write(' ****************************************************************/\n');
fSource.write('\ntypedef int16_t bg1_shift_csr_t[BG1_NNZ];\n');

define_1D_array(fSource, {'type'         : '__device__ __constant__ int16_t',
                          'name'         : 'bg1_csr_row_array',
                          'num_elements' : 'BG1_M + 1',
                          'format'       : '%4d',
                          'data'         : BG[1].get_csr_row_array()})
# Note: this could be uint8_t - columns indices are less than 68
#define_1D_array(fSource, {'type'         : '__device__ __constant__ uint16_t',
#                          'name'         : 'bg1_csr_col_array',
#                          'num_elements' : 'BG1_NNZ',
#                          'format'       : '%4d',
#                          'data'         : BG[1].get_csr_col_array()})
write_columns(fSource, {'type'         : '__device__ __constant__ int8_t',
                        'name'         : 'bg1_csr_col_array',
                        'num_elements' : 'BG1_NNZ',
                        'format'       : '%4d',
                        'data'         : BG[1].get_csr_columns()})

# Collect precomputed shift values for each Z
bg1_shift_Z = []
for set_idx, set in enumerate(Z_iLS):
    #print(set_idx, set)
    for Z in set:
        bg1_shift_Z.append((set_idx, Z, SG[1][set_idx].position_shift_Z(Z)))
#print(len(bg1_shift_Z))

write_shift_for_each_Z(fSource, {'type'    : '__device__ __constant__ bg1_shift_csr_t',
                                 'name'    : 'bg1_shift_Z',
                                 'length'  : 'NUM_Z_TABLES',
                                 'format'  : '%4d',
                                 'data'    : bg1_shift_Z,
                                 'min'     : 32})

# Generate a table to find the index in bg1_shift_Z for a given Z value
Z_shift_index = [Z_gt_32.index(i) if i in Z_gt_32 else -1 for i in range(max_Z + 1)]
#print(Z_shift_index)
define_1D_array(fSource, {'type'         : '__device__ __constant__ int8_t',
                          'name'         : 'bg1_Z_shift_index',
                          'num_elements' : 'MAX_Z+1',
                          'format'       : '%3d',
                          'data'         : Z_shift_index})

# Compressed sparse column data structures for column-based processing
fSource.write('\ntypedef int16_t bg1_shift_csc_t[BG1_NNZ];\n');
define_1D_array(fSource, {'type'         : '__device__ __constant__ int16_t',
                          'name'         : 'bg1_csc_col_array',
                          'num_elements' : 'BG1_N + 1',
                          'format'       : '%4d',
                          'data'         : BG[1].get_csc_col_array()})
# Unused at the moment 
#write_rows(fSource, {'type'         : '__device__ __constant__ int16_t',
#                     'name'         : 'bg1_csc_row_array',
#                     'num_elements' : 'BG1_NNZ',
#                     'format'       : '%4d',
#                     'data'         : BG[1].get_csc_rows()})

fSource.write('\ntypedef uchar2 csc_row_idx_t;\n');
write_row_pairs(fSource, {'type'         : '__device__ __constant__ csc_row_idx_t',
                          'name'         : 'bg1_csc_row_idx_array',
                          'num_elements' : 'BG1_NNZ',
                          'format'       : '%3d',
                          'data'         : BG[1].get_csc_row_pairs()})

# Collect precomputed shift values for each Z
bg1_col_shift_Z = []
for set_idx, set in enumerate(Z_iLS):
    #print(set_idx, set)
    for Z in set:
        bg1_col_shift_Z.append((set_idx, Z, SG[1][set_idx].position_shift_Z_columns(Z)))
#print(bg1_col_shift_Z)

write_shift_for_each_Z_column(fSource, {'type'    : '__device__ __constant__ bg1_shift_csc_t',
                                        'name'    : 'bg1_col_shift_Z',
                                        'length'  : 'NUM_Z_TABLES',
                                        'format'  : '%4d',
                                        'data'    : bg1_col_shift_Z,
                                        'min':     32})
########################################################################
#
# BG2
#
########################################################################
fSource.write('/****************************************************************\n');
fSource.write(' * BG2                                                          *\n');
fSource.write(' ****************************************************************/\n');

fSource.write('\ntypedef int16_t bg2_shift_csr_t[BG2_NNZ];\n');

define_1D_array(fSource, {'type'         : '__device__ __constant__ int16_t',
                          'name'         : 'bg2_csr_row_array',
                          'num_elements' : 'BG2_M + 1',
                          'format'       : '%4d',
                          'data'         : BG[2].get_csr_row_array()})
write_columns(fSource, {'type'         : '__device__ __constant__ int8_t',
                        'name'         : 'bg2_csr_col_array',
                        'num_elements' : 'BG2_NNZ',
                        'format'       : '%4d',
                        'data'         : BG[2].get_csr_columns()})

########################################################################
# Write a second source file
fSource2 = open('nrLDPC_flat.lwh', 'w')
fSource2.write(lw_copyright)
fSource2.write('\n// Note: This file has been automatically generated.\n\n')
fSource2.write('#include <stdint.h>\n')
fSource2.write('#include "vector_types.h" /* LWCA header */\n\n')

fSource2.write('#define BG1_M %d\n' % BG[1].num_rows)
fSource2.write('#define BG1_MAX_ROW_DEG %d\n\n' % BG[1].max_row_degree)
fSource2.write('#define BG1_NUM_KERNEL_NODES %d\n\n' % 26)

fSource2.write('struct ldpc_column_info_t\n{\n    int16_t index;\n    int16_t shift;\n};\n')

#fSource2.write('struct bg1_row_t\n{\n    int16_t       row_degree;\n    column_info_t column_info[BG1_MAX_ROW_DEG];\n};\n')
#fSource2.write('struct bg1_Z_graph_t\n{\n    bg1_row_t rows[BG1_M];\n};\n')

iLS = find_set_index_for_Z(384)

#write_Z_graph(fSource2, {'type' : '__device__ __constant__ bg1_Z_graph_t',
#                         'name' : 'bg1_%i' % 384,
#                         'Z'    : 384,
#                         'data' : SG[1][iLS] })

define_1D_array(fSource2, {'type'         : '__device__ __constant__ int32_t',
                           'name'         : 'bg1_row_degrees',
                           'num_elements' : 'BG1_M',
                           'format'       : '%4d',
                           'data'         : BG[1].get_row_degrees()})

write_Z_graph_2D(fSource2, {'type'         : '__device__ __constant__ ldpc_column_info_t',
                            'name'         : 'bg1_%i' % 384,
                            'num_elements' : ('BG1_M', 'BG1_MAX_ROW_DEG'),
                            'Z'            : 384,
                            'data'         : SG[1][iLS] })

########################################################################
# Write a source file with offsets into shared memory (instead of array
# indices, to avoid having to multiply by sizeof(T)
fSource3 = open('nrLDPC_offset.lwh', 'w')
fSource3.write(lw_copyright)
fSource3.write('\n// Note: This file has been automatically generated.\n\n')
fSource3.write('#include <stdint.h>\n')
fSource3.write('#include "vector_types.h" /* LWCA header */\n\n')

fSource3.write('#define BG1_M %d\n' % BG[1].num_rows)
fSource3.write('#define BG1_MAX_ROW_DEG %d\n\n' % BG[1].max_row_degree)
fSource3.write('#define BG1_NUM_KERNEL_NODES %d\n\n' % 26)

#fSource3.write('struct ldpc_column_info_t\n{\n    int16_t index;\n    int16_t shift;\n};\n')

#fSource2.write('struct bg1_row_t\n{\n    int16_t       row_degree;\n    column_info_t column_info[BG1_MAX_ROW_DEG];\n};\n')
#fSource2.write('struct bg1_Z_graph_t\n{\n    bg1_row_t rows[BG1_M];\n};\n')

iLS = find_set_index_for_Z(384)

#write_Z_graph(fSource2, {'type' : '__device__ __constant__ bg1_Z_graph_t',
#                         'name' : 'bg1_%i' % 384,
#                         'Z'    : 384,
#                         'data' : SG[1][iLS] })

define_1D_array(fSource3, {'type'         : '__device__ __constant__ int32_t',
                           'name'         : 'bg1_row_degrees',
                           'num_elements' : 'BG1_M',
                           'format'       : '%4d',
                           'data'         : BG[1].get_row_degrees()})

#print(BG[1].filled_position_table())

define_2D_array(fSource3, {'type'       : '__device__ __constant__ int32_t',
                           'name'       : 'bg1_pos',
                           'num_rows'   : 'BG1_M',
                           'num_columns': 'BG1_MAX_ROW_DEG',
                           'data'       : BG[1].filled_position_table()})
define_2D_array(fSource3, {'type'       : '__device__ __constant__ int32_t',
                           'name'       : 'bg1_384_shift',
                           'num_rows'   : 'BG1_M',
                           'num_columns': 'BG1_MAX_ROW_DEG',
                           'data'       : SG[1][iLS].filled_shift_table()})

#write_Z_graph_2D(fSource3, {'type'         : '__device__ __constant__ ldpc_column_info_t',
#                            'name'         : 'bg1_%i' % 384,
#                            'num_elements' : ('BG1_M', 'BG1_MAX_ROW_DEG'),
#                            'Z'            : 384,
#                            'data'         : SG[1][iLS] })

########################################################################
# Write a source file with C++ templates, to avoid using constant memory

sep = '////////////////////////////////////////////////////////////////////////\n'
fSource4 = open('nrLDPC_templates.lwh', 'w')
fSource4.write(lw_copyright)
fSource4.write('\n// Note: This file has been automatically generated.\n\n')
fSource4.write('#include <stdint.h>\n\n')

fSource4.write('namespace ldpc2\n{\n')

fSource4.write(sep)
fSource4.write('// max_row_degree\n')
fSource4.write('// Provides the maximum row degree for check nodes as a function of\n')
fSource4.write('// base graph index (1 or 2)\n')
fSource4.write('template <int BG> struct max_row_degree;\n')
fSource4.write('template<> struct max_row_degree<1> { static const int value = %d; };\n'   % BG[1].max_row_degree)
fSource4.write('template<> struct max_row_degree<2> { static const int value = %d; };\n\n' % BG[2].max_row_degree)

fSource4.write(sep)
fSource4.write('// max_info_nodes\n')
fSource4.write('// Provides the maximum number of info (or non-parity) nodes as a function of\n')
fSource4.write('// base graph index (1 or 2). Note that for BG2, the number of information nodes\n')
fSource4.write('// can be less that the maximum.\n')
fSource4.write('template <int BG> struct max_info_nodes;\n')
fSource4.write('template<> struct max_info_nodes<1> { static const int value = %d; };\n'   % 22)
fSource4.write('template<> struct max_info_nodes<2> { static const int value = %d; };\n\n' % 10)

fSource4.write(sep)
fSource4.write('// max_parity_nodes\n')
fSource4.write('// Provides the maximum number of parity nodes as a function of base graph\n')
fSource4.write('// index (1 or 2).\n')
fSource4.write('template <int BG> struct max_parity_nodes;\n')
fSource4.write('template<> struct max_parity_nodes<1> { static const int value = %d; };\n'   % BG[1].num_rows)
fSource4.write('template<> struct max_parity_nodes<2> { static const int value = %d; };\n\n' % BG[2].num_rows)

fSource4.write(sep)
fSource4.write('// max_variable_nodes\n')
fSource4.write('// Provides the maximum number of variable nodes as a function of base graph\n')
fSource4.write('// index (1 or 2).\n')
fSource4.write('template <int BG> struct max_variable_nodes;\n')
fSource4.write('template<> struct max_variable_nodes<1> { static const int value = %d; };\n'   % BG[1].num_columns)
fSource4.write('template<> struct max_variable_nodes<2> { static const int value = %d; };\n\n' % BG[2].num_columns)

fSource4.write(sep)
fSource4.write('// num_kernel_nodes\n')
fSource4.write('// Provides the number of "kernel" nodes as a function of\n')
fSource4.write('// base graph index (1 or 2). The number of kernel nodes is\n')
fSource4.write('// the number of "systematic" or "information" nodes plus\n')
fSource4.write('// the square matrix with a bidiagonal structure.\n')
fSource4.write('// See:\n')
fSource4.write('// "Algebra-Assisted Construction of Quasi-Cyclic LDPC Codes for\n')
fSource4.write('// 5G New Radio," H. Li, B. Bai, X. Mu, J. Zhang, and H. Xu. IEEE\n')
fSource4.write('// Access, vol. 6, pp. 50229-50244, 2018.\n')
fSource4.write('template <int BG> struct num_kernel_nodes;\n')
fSource4.write('template<> struct num_kernel_nodes<1> { static const int value = %d; };\n'   % 26)
fSource4.write('template<> struct num_kernel_nodes<2> { static const int value = %d; };\n\n' % 14)

fSource4.write(sep)
fSource4.write('// set_index\n')
fSource4.write('// Provides the set index for a given lifting size Z\n')
fSource4.write('// See Table 5.3.2-1, 3GPP 38.212\n')
fSource4.write('template <int Z> struct set_index;\n')

for idx, iLS_seq in enumerate(Z_iLS):
    fSource4.write('// iLS = %d\n' % idx)
    for Z in iLS_seq:
        fSource4.write('template <> struct set_index<%d> { static const int value = %d; };\n' % (Z, idx))

fSource4.write('\n')

fSource4.write(sep)
fSource4.write('// row_degree\n')
fSource4.write('// Provides degree (number of nonzero shifted permutation matrices) as\n')
fSource4.write('// a function of base graph and check node row.\n')
fSource4.write('// See Tables 5.3.2-2 and 5.3.2-3, 3GPP 38.212\n')
fSource4.write('template <int BG, int ROW_INDEX> struct row_degree;\n')

for BG_idx in [1, 2]:
    for row_idx, row in enumerate(BG[BG_idx].row_list):
        fSource4.write('template <> struct row_degree<%d, %d> { static const int value = %d; };\n' % (BG_idx, row_idx, len(row)))
    fSource4.write('\n')

fSource4.write(sep)
fSource4.write('// isolated_edge_count\n')
fSource4.write('// "Isolated" edges are base graph edges that are used by only 1 parity\n')
fSource4.write('// check node. As such, we do not need to update the stored values. (Use\n')
fSource4.write('// during the next iteration would subtract the same value that was\n')
fSource4.write('// added.)\n')
fSource4.write('template <bool THasIsolatedEdge> struct isolated_edge_count;\n')
fSource4.write('template <> struct isolated_edge_count<true>  { static const int value = 1; };\n')
fSource4.write('template <> struct isolated_edge_count<false> { static const int value = 0; };\n\n')

fSource4.write(sep)
fSource4.write('// update_row_degree\n')
fSource4.write('// Parity check nodes with an isolated edge at the end can avoid updating\n')
fSource4.write('// the value, since it is only used by one node. For 5G base graphs, rows 4\n')
fSource4.write('// and higher have an isolated edge.\n')
fSource4.write('template <int BG, int ROW_INDEX>\n')
fSource4.write('struct update_row_degree\n')
fSource4.write('{\n')
fSource4.write('    static const int value = (row_degree<BG, ROW_INDEX>::value - isolated_edge_count<(ROW_INDEX > 3)>::value);\n')
fSource4.write('};\n\n');


fSource4.write(sep)
fSource4.write('// vnode_index\n')
fSource4.write('// Variable node indices (shared by all lifting sets for a given base\n')
fSource4.write('// graph).\n')
fSource4.write('// See Tables 5.3.2-2 and 5.3.2-3, 3GPP 38.212\n')
fSource4.write('template <int BG, int CHECK_NODE, int INDEX> struct vnode_index;\n')
for BG_idx in [1, 2]:
    for row_idx, row in enumerate(BG[BG_idx].row_list):
        fSource4.write('// BG%d CHECK NODE %d (degree: %d)\n' % (BG_idx, row_idx, len(row)))
        for val_idx, val in enumerate(row):
            fSource4.write('template <> struct vnode_index<%d, %d, %d> { static const int value = %d; };\n' % (BG_idx, row_idx, val_idx, val))
    fSource4.write('\n')
fSource4.write('\n')
    
fSource4.write(sep)
fSource4.write('// vnode_shift\n')
fSource4.write('// Variable node shift values as a function of base graph,\n')
fSource4.write('// index set, check node index, and index within the row.\n')
fSource4.write('// See Tables 5.3.2-2 and 5.3.2-3, 3GPP 38.212\n')
fSource4.write('template <int BG, int ILS, int CHECK_NODE, int INDEX> struct vnode_shift;\n')
#for BG_idx in [1]:
for BG_idx in [1, 2]:
    for iLS_idx in range(len(Z_iLS)):
        sgraph = SG[BG_idx][iLS_idx]
        for row_idx, row in enumerate(sgraph.set_row_list):
            fSource4.write('// ILS = %d, CHECK_NODE = %d\n' % (iLS_idx, row_idx))
            for val_idx, val in enumerate(row):
                fSource4.write('template <> struct vnode_shift<%d, %d, %d,  %d> { static const int value =  %d; };\n' % (BG_idx, iLS_idx, row_idx, val_idx, val[1]))

fSource4.write('\n\n' + sep)
fSource4.write('// vnode_shift_mod\n')
fSource4.write('// Variable node shift values as a function of base graph, lifting size\n')
fSource4.write('// (Z), check node index, and index within the row. The modulo operation\n')
fSource4.write('// (on lifting size) is performed.\n')
fSource4.write('// See Tables 5.3.2-2 and 5.3.2-3, 3GPP 38.212\n')
fSource4.write('template <int BG, int Z, int CHECK_NODE, int INDEX> struct vnode_shift_mod\n')
fSource4.write('{\n')
fSource4.write('    static const unsigned short value = vnode_shift<BG, set_index<Z>::value, CHECK_NODE, INDEX>::value % Z;\n')
fSource4.write('};\n\n')

fSource4.write(sep)
fSource4.write('// vnode_base_offset\n')
fSource4.write('// Returns the index of the first APP value for the requested non-zero\n')
fSource4.write('// block matrix in the CHECK_NODE row.\n')
fSource4.write('// value = col_index * Z\n')
fSource4.write('template <int BG, int Z, int CHECK_NODE, int INDEX> struct vnode_base_offset\n')
fSource4.write('{\n')
fSource4.write('    static const unsigned short value = vnode_index<BG, CHECK_NODE, INDEX>::value * Z;\n')
fSource4.write('};\n\n')

fSource4.write(sep)
fSource4.write('// wrap_index\n')
fSource4.write('// --base offset---->|\n')
fSource4.write('//     <---shift---->|\n')
fSource4.write('//     | - - - - - - - - - - - - - - - - - - - - |\n')
fSource4.write('//  0  |             x                           |\n')
fSource4.write('//  1  |               x                         |\n')
fSource4.write('//  2  |                 x                       |\n')
fSource4.write('//  3  |                   x                     |\n')
fSource4.write('//  4  |                     x                   |\n')
fSource4.write('//  5  |                       x                 |\n')
fSource4.write('//  6  |                         x               |\n')
fSource4.write('//  7  |                           x             |\n')
fSource4.write('//  8  |                             x           |\n')
fSource4.write('//  9  |                               x         |\n')
fSource4.write('// 10  |                                 x       |\n')
fSource4.write('// 11  |                                   x     |\n')
fSource4.write('// 12  |                                     x   |\n')
fSource4.write('// 13  |                                       x |\n')
fSource4.write('// 14  | x.......................................|<-- wrap index\n')
fSource4.write('// 15  |   x                                     |\n')
fSource4.write('// 16  |     x                                   |\n')
fSource4.write('// 17  |       x                                 |\n')
fSource4.write('// 18  |         x                               |\n')
fSource4.write('// 19  |           x                             |\n')
fSource4.write('//     | - - - - - - - - - - - - - - - - - - - - |\n')
fSource4.write('template <int BG, int Z, int CHECK_NODE, int INDEX> struct wrap_index\n')
fSource4.write('{\n')
fSource4.write('    static const unsigned short value = Z - vnode_shift_mod<BG, Z, CHECK_NODE, INDEX>::value;\n')
fSource4.write('};\n\n')

fSource4.write(sep)
fSource4.write('// vnode_shift_offset\n')
fSource4.write('// Provides (vnode_colum * Z) + shift_mod for given values of the\n')
fSource4.write('// base graph (BG), lifting size (Z), CHECK_NODE, and row index INDEX.\n')
fSource4.write('template <int BG, int Z, int CHECK_NODE, int INDEX> struct vnode_shift_offset\n')
fSource4.write('{\n')
fSource4.write('    static const unsigned short value = (vnode_index<BG, CHECK_NODE, INDEX>::value * Z) +\n')
fSource4.write('                                        vnode_shift_mod<BG, Z, CHECK_NODE, INDEX>::value;\n')
fSource4.write('};\n\n')


fSource4.write('} // namespace ldpc2\n')
