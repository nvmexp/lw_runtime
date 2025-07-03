#ifndef _MME64_H
#define _MME64_H 1

#include "op.h"
#include "reg.h"
#include "out.h"
#include "pred.h"
#include "extended.h"
#include "special.h"
#include "common/mmebitset.h"
#include <vector>

const int MME64NumAlu = 2;
const int MME64NumOut = 2;
const int MME64NumLoad = 2;
const int MME64PredModeBits = 3;
const int MME64PredicateBits = MME64Reg::bits+MME64PredModeBits;
const int MME64ImmedBits = 16;
const int MME64OpBits = MME64Op::bits+3*MME64Reg::bits+MME64ImmedBits;
// Assume that we can save a bit on the method enum
const int MME64OutputBits = (MME64Out::bits-1) + MME64Out::bits;
const int MME64GroupBits = 1 + MME64PredicateBits + MME64NumAlu*MME64OpBits + MME64NumOut*MME64OutputBits;

typedef mme::bitset<MME64GroupBits> MME64HW;

struct MME64Global {
    MME64Reg::RegEnum pred;
    MME64Pred::PredEnum predMode;
    bool endNext;

    bool operator== (const MME64Global &o) const {
        return
            pred == o.pred &&
            predMode == o.predMode &&
            endNext == o.endNext;
    }
    bool operator!=(const MME64Global &o) const { return !((*this) == o); }
};

static const MME64Global MME64GlobalDefault = {
    MME64Reg::ZERO, MME64Pred::UUUU, false
};

struct MME64Alu {
    MME64Op::OpEnum op;
    MME64Reg::RegEnum dst;
    MME64Reg::RegEnum src[2];
    uint16_t immed;

    bool operator== (const MME64Alu &o) const {
        return
            op == o.op &&
            dst == o.dst &&
            src[0] == o.src[0] &&
            src[1] == o.src[1] &&
            immed == o.immed;
    }
    bool operator!=(const MME64Alu &o) const { return !((*this) == o); }
};

static const MME64Alu MME64AluDefault = {
    MME64Op::ADD, MME64Reg::ZERO, {MME64Reg::ZERO, MME64Reg::ZERO}, 0
};

struct MME64Output {
    MME64Out::OutEnum method;
    MME64Out::OutEnum emit;
    bool operator== (const MME64Output &o) const {
        return method == o.method &&
            emit == o.emit;
    }
    bool operator!=(const MME64Output &o) const { return !((*this) == o); }
};

static const MME64Output MME64OutputDefault = {
    MME64Out::NONE, MME64Out::NONE
};

struct MME64Group {
    MME64Global global;
    std::vector<MME64Alu> alu;
    std::vector<MME64Output> output;

    // Check to see if this group can actually be packed into the bits allotted
    // Fill the vectors with default as needed
    // Errors out on failure
    void validate();
    // Checks for semantic errors in the group. Can be skipped to generate
    // illegal groups for error testing
    // Errors out on failure
    void legalize();

    MME64Group() : global(MME64GlobalDefault) {;}
    // Unpack from the binary representation
    MME64Group(const MME64HW &hw);
    // Unpack from the binary representation in 3 32-bit words
    MME64Group(const uint32_t*);
    // Internal delegating constructor
    void init(const MME64HW &hw);
    // Pack into the binary representation
    MME64HW pack() const;
    // Pack into the binary representation into the 3 32-bit words starting at the argument
    void pack(uint32_t*) const;
    // Disassemble to text. Verbose dumps even non-NOP groups.
    // decodeExtImmed decodes extended immediates for branch and merge operations
    std::string disassemble(bool verbose = false, bool decodeExtImmed = false) const;
};

#endif // _MME64_H
