from enum import Enum
import sys
import re

matcher_case = re.compile(r'\[\s*(\d+)\] (.*)')
matcher_case_end = re.compile(r'/////////////////////////////////////////////////////////////////')

matcher_node_left = re.compile(r'if\((\w+) <= (\d+)\)')

matcher_leaf = re.compile(r'// (\d+) elements')
matcher_leaf_cases = re.compile(r'// Case Indices: \[([\d,]+)\]')
matcher_leaf_begin = re.compile(r'static const int r\[\d*\] = \{(\d+),')
matcher_leaf_entry = re.compile(r'(\d+)')

Keys = Enum('Keys', 'm n k ta tb data_type comp_type hmma_elig')


cases = {}
case_lists = []
nodes = []
leafs = []

class case:
    def __init__(self, id, flags):
        self.id = id
        self.flags = flags
        self.index = 0
        cases[id] = self

class case_list:
    def __init__(self):
        self.elements = []
        self.index = 0
        case_lists.append(self)

class node:
    def __init__(self, key, threshold):
        self.is_node = True
        self.is_leaf = False
        self.key = key
        self.threshold = threshold
        self.left = None
        self.right = None
        self.index = 0
        nodes.append(self)

class leaf:
    def __init__(self):
        self.is_node = False
        self.is_leaf = True
        self.values = []
        self.cases = case_list()
        self.index = 0
        leafs.append(self)

def process_node(file, key, threshold):

    new_node = node(Keys[key], threshold)

    while not new_node.left:
        l = f.readline()
        m = matcher_node_left.search(l)
        if m:
            new_node.left = process_node(f, m.group(1), m.group(2))
            continue
        m = matcher_leaf.search(l)
        if m:
            new_node.left = process_leaf(f, m.group(1))
            continue

    # print 'search for right path "{}"'.format('if\({} > {}\)'.format(key, threshold))
    while True:
        l = f.readline()
        m = re.search(r'if\({} > {}\)'.format(key, threshold), l)
        if m:
            break
    while not new_node.right:
        l = f.readline()
        m = matcher_node_left.search(l)
        if m:
            new_node.right = process_node(f, m.group(1), m.group(2))
            continue
        m = matcher_leaf.search(l)
        if m:
            new_node.right = process_leaf(f, m.group(1))
            continue
    
    return new_node

def process_leaf(file, elements):

    new_leaf = leaf()

    m = matcher_leaf_cases.search(f.readline())
    for case_id in m.group(1).split(','):
        new_leaf.cases.elements.append(cases[int(case_id)])

    m = matcher_leaf_begin.search(f.readline())
    new_leaf.values.append(m.group(1))

    while True:
        l = f.readline()
        m = matcher_leaf_entry.search(l)
        if m:
            # print 'appending entry {}'.format(m.group(1))
            new_leaf.values.append(m.group(1))
        else:
            break

    # print 'leaf len = {}'.format(len(new_leaf.values))
    return new_leaf

def process_case(id_text, flags):
    new_case = case(int(id_text), flags)
    # print('new case: {}'.format(new_case.id))

hname = sys.argv[1]
fname = '{}.cpp'.format(hname)

root_node = None

with open(fname) as f:
    after_first_case = False
    while True:
        l = f.readline()
        if not l:
            break

        m = matcher_case.search(l)
        if m:
            process_case(m.group(1), m.group(2))
            after_first_case = True
        elif after_first_case:        
            m = matcher_case_end.search(l)
            if m:
                break

    while True:
        l = f.readline()
        if not l:
            break

        m = matcher_node_left.search(l)
        if m:
            root_node = process_node(f, m.group(1), m.group(2))
            break

# index everything
i = 0
for case in cases.values():
    case.index = i
    i = i + 1

i = 0
for leaf in leafs:
    leaf.index = i
    i = i + 1

i = 0
for node in nodes:
    node.index = i
    i = i + 1

# print the tree
print('''#include <stdio.h>
// #define LWPRINTF printf
#define LWPRINTF(...) 

#ifdef DEBUG
#define DEBUG_HEURISTICS_TREE_TRAVERSE
#endif
static const unsigned LEAF = 1u << 31;
struct node_t{
  int key;
  int threshold;
  unsigned left, right;
};
struct leaf_t{
  int values[32];
#ifdef DEBUG_HEURISTICS_TREE_TRAVERSE
  int case_list_start;
#endif
};''')
print('''#ifdef DEBUG_HEURISTICS_TREE_TRAVERSE
struct case_t{
  const char *flags;
};
static const case_t cases[] = {''')
for case in cases.values():
    print('  /* {} */ {{"{}"}},'.format(case.index, case.flags))
print('};')
print('''// -1 terminated lists of cases
static const int case_lists[] = {''')
case_list_index = 0
for case_list in case_lists:
    case_list.index = case_list_index
    print('  /* {} */ {}, -1,'.format(case_list.index, ', '.join(str(case.index) for case in case_list.elements)))
    case_list_index = case_list_index + len(case_list.elements) + 1
print('''};
#define CASE_LIST(list) , list
#else
#define CASE_LIST(...)
#endif // DEBUG_HEURISTICS_TREE_TRAVERSE''')
print('static const node_t nodes[] = {')
print('  /* keys: m, n, k, ta, tb, Atype, computeType, hmma_eligible */')
for node in nodes:
    print('  /* {} */ {{{}, {}, {}{}, {}{}}},'.format(node.index, node.key.value - 1, node.threshold,
                                             'LEAF | ' if node.left.is_leaf else '', node.left.index,
                                             'LEAF | ' if node.right.is_leaf else '', node.right.index))
print('};')
print('static const leaf_t leafs[] = {')
for leaf in leafs:
    print('  /* {} */ {{{{{}}} CASE_LIST({})}},'.format(leaf.index, ', '.join(leaf.values), leaf.cases.index))
print('};')
print('''
#ifdef DEBUG_HEURISTICS_TREE_TRAVERSE
static const char* key_strings[]={{{}}};
#endif'''.format(", ".join('"{}"'.format(k.name) for k in list(Keys))))
print(
    '\nconst int (*{}(int (&keys)[{}]))[32] {{'.format(hname, len(Keys)))
print('''  int next = 0;
  LWPRINTF("  heur tree traverse: ");
  while((next & LEAF) == 0){
    const node_t &node = nodes[next];
    if(keys[node.key] <= node.threshold) {
      LWPRINTF(" %s <= %d;", key_strings[node.key], node.threshold);
      next = node.left;
    } else {
      LWPRINTF(" %s > %d;", key_strings[node.key], node.threshold);
      next = node.right;
    }
  }

  const leaf_t &leaf = leafs[next & ~(LEAF)];

  #ifdef DEBUG_HEURISTICS_TREE_TRAVERSE
  LWPRINTF(" -> \\n");
  for(int case_list_index = leaf.case_list_start; case_lists[case_list_index] != -1; case_list_index++) {
    const case_t &test_case = cases[case_lists[case_list_index]];
    (void)key_strings;(void)test_case; // silence unused variable warning when LWPRINTF is disabled
    LWPRINTF("    [%06d] %s\\n", case_list_index, test_case.flags);
  }
  #endif

  return &leaf.values;
}
''')
