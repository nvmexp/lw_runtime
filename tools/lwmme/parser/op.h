#ifndef _OP_H
#define _OP_H

#include <vector>
#include <sstream>

#include "parse.h"
//#include "mmecommon.h"
#include "common/mmebitset.h"

#include <vector>
#include <set>
#include <map>

#include <assert.h>

#include <stdio.h>
#include <stdint.h>

enum MMEVersion {
    MME1,
    MME2,
};

class Graph {
public:
    Graph() {;}
    void addEdge(int a, int b) {
        edges[a].insert(b);
        edges[b].insert(a);
        nodes.insert(a);
        nodes.insert(b);
    }
    void removeEdge(int a, int b) {
        if (nodes.find(a) == nodes.end() ||
            nodes.find(b) == nodes.end())
            return;
        edges[a].erase(b);
        edges[b].erase(a);
    }
    void removeNode(int n) {
        std::map<int, std::set<int> >::iterator i;

        for (i=edges.begin(); i!=edges.end(); ++i) {
            (*i).second.erase(n);
        }

        edges[n].clear();

        nodes.erase(n);
    }
    bool hasEdge(int a, int b) const {
        std::map<int, std::set<int> >::const_iterator i = edges.find(a);
        if (i == edges.end()) return false;
        return (*i).second.find(b) != (*i).second.end();
    }
    bool hasNode(int n) const {
        return nodes.find(n) != nodes.end();
    }
    int degree(int n) const {
        std::map<int, std::set<int> >::const_iterator i = edges.find(n);
        assert((*i).second.find(n) != (*i).second.end());
        const int degree = static_cast<int>((*i).second.size()) - 1;
        assert((degree >= 0)
               && (static_cast<unsigned>(degree) == (*i).second.size() - 1));
        return degree;
    }
    std::set<int> neighborList(int n) const {
        std::map<int, std::set<int> >::const_iterator i = edges.find(n);
        assert((*i).second.find(n) != (*i).second.end());
        return (*i).second;
    }
    int numActiveNodes() const {
        const int numNodes = nodes.size();
        assert((numNodes >= 0)
               && (static_cast<unsigned>(numNodes) == nodes.size()));
        return numNodes;
    }
    void printDot() const {
        printf("graph {\n");
        for (std::map<int, std::set<int> >::const_iterator i = edges.begin(); i != edges.end(); ++i) {
            for (std::set<int>::const_iterator j = (*i).second.begin(); j != (*i).second.end(); ++j) {
                if ((*i).first < (*j)) {
                    printf("  %d -- %d\n", (*i).first, (*j));
                }
            }
        }
        printf("}\n");
    }
protected:
    std::map<int, std::set<int> > edges;
    std::set<int> nodes;
};

// Numbers [0, NUM_FIXED_TEMPS-1] are fixed registers to the allocator
// [NUM_FIXED_TEMPS, NUM_VIRT_TEMPS-1] are virtual registers to be allocated
// User temps are only allowed in the range [0, NUM_USER_TEMPS-1] (really only [NUM_FIXED_TEMPS, NUM_USER_TEMPS-1]
static const int MME1_FIXED_TEMPS = 8;
static const int MME64_FIXED_TEMPS = 24;
static const int NUM_FIXED_TEMPS = 32;
static const int NUM_USER_TEMPS = 128;
static const int NUM_VIRT_TEMPS = 256;

// Track a fake virtual for method liveness for the spilling code
const int NUM_TRACKED = NUM_VIRT_TEMPS+1;
const int METHOD_VIRTUAL = NUM_VIRT_TEMPS;

class SpillInfo {
public:
    SpillInfo() : base(-1), count(-1) {;}

    int base;
    int count;
};

// program.cpp
std::vector<unsigned int> assemble(const std::string &input, std::string *optComment, std::vector<Pragma> &pragmas);

// program2.cpp
std::vector<unsigned int> assemble2(const std::string &input, std::string *optComment, std::vector<Pragma> &pragmas);
void pgo2(std::vector<unsigned int> &ucode, MMEVersion ver, const SpillInfo &spill, const std::vector<Pragma> &pragmas);

// llasm.cpp
std::string llDisassemble(unsigned int input);
std::vector<unsigned int> llAssemble(std::string str, std::string *optComment, std::vector<Pragma> &pragmas);
void initLLAsm();
void destroyLLAsm();

// llasm2.cpp
// Takes a pointer to the first of the three dwords
std::string llDisassemble2(uint32_t *input);
std::vector<uint32_t> llAssemble2(std::string str, std::string *optComment, std::vector<Pragma> &pragmas);

// runtest.cpp
bool runTests(const std::vector<unsigned int> &ucode, MMEVersion ver,
    const SpillInfo &spill, const std::vector<Pragma> &pragmas, bool requireTests,
    std::string *optComment = NULL, int *pgoCost = NULL);

// programutil.cpp
extern SpillInfo spill;
void PackConstMethods(std::vector<ScanOp> &prog, int maxIncrement, bool defaultFlowRemoveEnd);
void validateSliceBits(ScanSlice s);

#endif //def _OP_H
