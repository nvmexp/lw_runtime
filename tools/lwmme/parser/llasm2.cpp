#pragma warning (disable:4291)
#define _CRT_SELWRE_NO_WARNINGS

#include <cstring>
#include <stdlib.h>
#include <string>
#include <map>
#include "parser/op.h"
#include "playback/mme.h"

#include "mme64/mme64.h"

#define VERBOSE 0

using namespace std;
using namespace mme;

void MME64Group::validate()
{
    if (alu.size() > MME64NumAlu) {
        mmeError("Too many alu ops in group");
    }
    while (alu.size() < MME64NumAlu) {
        alu.push_back(MME64AluDefault);
    }

    if (output.size() > MME64NumOut) {
        mmeError("Too many output ops in group");
    }
    while (output.size() < MME64NumOut) {
        output.push_back(MME64OutputDefault);
    }
}

void MME64Group::legalize()
{
    // ALU destinations must be either unique within the group or ZERO
    for (int i=0; i<MME64NumAlu; i++) {
        if (alu[i].dst == MME64Reg::ZERO) continue;
        for (int j=i+1; j<MME64NumAlu; j++) {
            if (alu[i].dst == alu[j].dst) {
                mmeError("Non-ZERO ALU destinations match within a group");
            }
        }
    }

    // ADDC and SUBB may not be in ALU0 and must be paired with
    // ADD and SUB respectively in the next lower ALU
    // MULH and MULHU may not be in ALU0 and must be paired with
    // MUL and MULU respectively in the next lower ALU
    for (int i=0; i<MME64NumAlu; i++) {
        switch (alu[i].op) {
        case MME64Op::MULH:
            if (i == 0 || (alu[i-1].op != MME64Op::MUL && alu[i - 1].op != MME64Op::MULU)) {
                mmeError("MULH not paired with MUL in next lower ALU");
            }
            break;
        case MME64Op::ADDC:
            if (i == 0 || alu[i-1].op != MME64Op::ADD) {
                mmeError("ADDC not paired with ADD in next lower ALU");
            }
            break;
        case MME64Op::SUBB:
            if (i == 0 || alu[i-1].op != MME64Op::SUB) {
                mmeError("SUBB not paired with SUB in next lower ALU");
            }
            break;
        default:
            break;
        }
    }

    // Branch ops cannot be present in a group with a predicate other than !ZERO
    // There can only be one branch op in a group and it must be in ALU0
    // There can only be one STATE in a group
    int numBranches = 0;
    int numStates = 0;
    vector<bool> aluOutputAllowed(MME64NumAlu);
    for (int i=0; i<MME64NumAlu; i++) {
        switch (alu[i].op) {
        case MME64Op::STATE:
            if (numStates != 0) {
                mmeError("More than one STATE in a group");
            }
            numStates++;
            // fallthrough
        case MME64Op::MUL:
        case MME64Op::MULU:
        case MME64Op::DWRITE:
            aluOutputAllowed[i] = false;
            break;
        case MME64Op::JAL:
        case MME64Op::BLT:
        case MME64Op::BLTU:
        case MME64Op::BLE:
        case MME64Op::BLEU:
        case MME64Op::BEQ:
        case MME64Op::LOOP:
            if (global.predMode != MME64Pred::UUUU) {
                mmeError("Branch present in a group with predicate mode other than UUUU");
            }
            if (i != 0) {
                mmeError("Branch present in an ALU other than ALU0");
            }
            if (numBranches != 0) {
                mmeError("More than one branch in a group");
            }
            numBranches++;
            aluOutputAllowed[i] = false;
            break;
        default:
            aluOutputAllowed[i] = true;
            break;
        }
    }

    // If an ALU is exelwting a deferred write instruction  lwrrently STATE, MUL, or MULU,
    // it cannot be used as a method or emit.
    for (int i=0; i<MME64NumOut; i++) {
        if (output[i].method >= MME64Out::ALU0 &&
            output[i].method < MME64Out::ALU0 + MME64NumAlu) {

            if (!aluOutputAllowed[output[i].method - MME64Out::ALU0]) {
                mmeError("Cannot output from a deferred-write or branch ALU op");
            }
        }
        if (output[i].emit >= MME64Out::ALU0 &&
            output[i].emit < MME64Out::ALU0 + MME64NumAlu) {

            if (!aluOutputAllowed[output[i].emit - MME64Out::ALU0]) {
                mmeError("Cannot output from a deferred-write or branch ALU op");
            }
        }
    }

    // LOADn can only be used as a source if LOADn-1 is also used
    vector<bool> loadUsed(MME64NumLoad, false);
    for (int i=0; i<MME64NumAlu; i++) {
        for (int j=0; j<2; j++) {
            MME64Reg::RegEnum src = alu[i].src[j];
            if (src >= MME64Reg::LOAD0 &&
                src < MME64Reg::LOAD0 + MME64NumLoad) {

                loadUsed[src-MME64Reg::LOAD0] = true;
            }
        }
    }

    for (int i=0; i<MME64NumOut; i++) {
        if (output[i].method >= MME64Out::LOAD0 &&
            output[i].method < MME64Out::LOAD0 + MME64NumLoad) {

            loadUsed[output[i].method - MME64Out::LOAD0] = true;
        }
        if (output[i].emit >= MME64Out::LOAD0 &&
            output[i].emit < MME64Out::LOAD0 + MME64NumLoad) {

            loadUsed[output[i].emit - MME64Out::LOAD0] = true;
        }
    }

    for (int i=1; i<MME64NumLoad; i++) {
        if (loadUsed[i] && !loadUsed[i-1]) {
            mmeError("LOADn used as a source without LOADn-1");
        }
    }
}

MME64HW MME64Group::pack() const {
    MME64HW rv;

    // ZZZ Scalability
    rv.insert<0,0>(global.endNext ? 1 : 0);
    rv.insert<4,1>(global.predMode);
    rv.insert<9,5>(global.pred);

    rv.insert<14,10>(alu[0].op);
    rv.insert<19,15>(alu[0].dst);
    rv.insert<24,20>(alu[0].src[0]);
    rv.insert<29,25>(alu[0].src[1]);
    rv.insert<45,30>(alu[0].immed);

    rv.insert<50,46>(alu[1].op);
    rv.insert<55,51>(alu[1].dst);
    rv.insert<60,56>(alu[1].src[0]);
    rv.insert<65,61>(alu[1].src[1]);
    rv.insert<81,66>(alu[1].immed);

    rv.insert<84,82>(output[0].method);
    rv.insert<88,85>(output[0].emit);

    rv.insert<91,89>(output[1].method);
    rv.insert<95,92>(output[1].emit);

    return rv;
}

void MME64Group::pack(uint32_t *words) const {
    MME64HW hwInst = pack();

    words[0] = hwInst.extract<MME64GroupBits, 64>().toUint();
    words[1] = hwInst.extract<63, 32>().toUint();
    words[2] = hwInst.extract<31, 0>().toUint();
}

void MME64Group::init(const MME64HW &hw) {
    MME64Alu newAlu;
    MME64Output newoutput;

    // ZZZ Scalability
    global.endNext = hw.extract<0, 0>().toUint() ? true : false;
    global.predMode = (MME64Pred::PredEnum)hw.extract<4, 1>().toUint();
    global.pred = (MME64Reg::RegEnum)hw.extract<9, 5>().toUint();

    newAlu.op = (MME64Op::OpEnum)hw.extract<14, 10>().toUint();
    newAlu.dst = (MME64Reg::RegEnum)hw.extract<19, 15>().toUint();
    newAlu.src[0] = (MME64Reg::RegEnum)hw.extract<24, 20>().toUint();
    newAlu.src[1] = (MME64Reg::RegEnum)hw.extract<29, 25>().toUint();
    newAlu.immed = hw.extract<45, 30>().toUint();
    alu.push_back(newAlu);

    newAlu.op = (MME64Op::OpEnum)hw.extract<50, 46>().toUint();
    newAlu.dst = (MME64Reg::RegEnum)hw.extract<55, 51>().toUint();
    newAlu.src[0] = (MME64Reg::RegEnum)hw.extract<60, 56>().toUint();
    newAlu.src[1] = (MME64Reg::RegEnum)hw.extract<65, 61>().toUint();
    newAlu.immed = hw.extract<81, 66>().toUint();
    alu.push_back(newAlu);

    newoutput.method = (MME64Out::OutEnum)hw.extract<84, 82>().toUint();
    newoutput.emit = (MME64Out::OutEnum)hw.extract<88, 85>().toUint();
    output.push_back(newoutput);

    newoutput.method = (MME64Out::OutEnum)hw.extract<91, 89>().toUint();
    newoutput.emit = (MME64Out::OutEnum)hw.extract<95, 92>().toUint();
    output.push_back(newoutput);
}

MME64Group::MME64Group(const uint32_t *instructions) {
    MME64HW bits;

    // Pack the three dwords into the instruction
    bits.insert<31, 0>(mme::bitset<32>(instructions[2]));
    bits.insert<63, 32>(mme::bitset<32>(instructions[1]));
    bits.insert<MME64GroupBits, 64>(mme::bitset<MME64GroupBits - 64 + 1>(instructions[0]));

    init(bits);
}

MME64Group::MME64Group(const MME64HW &hw) {
    init(hw);
}

static string DissassembleReg(MME64Reg::RegEnum reg) {
    if (reg >= MME64Reg::VIRTUAL0) {
        char vregstr[128];
        sprintf(vregstr, "V%d", (int)(reg - MME64Reg::VIRTUAL0));
        return string(vregstr);
    } else {
        return MME64Reg::mapping[reg].name;
    }
}

string MME64Group::disassemble(bool verbose, bool decodeExtImmed) const {
    vector<string> ops;

    if (verbose || global != MME64GlobalDefault) {
        string globalOp;
        if (global.endNext) {
            globalOp += "ENDNEXT, ";
        }
        globalOp += "?";
        globalOp += MME64Pred::mapping[global.predMode].name;
        if (verbose || global.predMode != MME64Pred::UUUU) {
            globalOp += ", ";
            globalOp += DissassembleReg(global.pred);
        }
        ops.push_back(globalOp);
    }

    // We can elide later ops, but we have to print NOPs up to the
    // first used op for alignment
    int lastUsedAlu = 0;
    for (int i=0; i<MME64NumAlu; i++) {
        if (verbose || alu[i] != MME64AluDefault) {
            lastUsedAlu = i;
        }
    }

    for (int i=0; i<=lastUsedAlu; i++) {
        string aluOp;

        aluOp += MME64Op::mapping[alu[i].op].name + " ";
        aluOp += DissassembleReg(alu[i].dst) + ", ";
        aluOp += DissassembleReg(alu[i].src[0]) + ", ";
        aluOp += DissassembleReg(alu[i].src[1]) + ", ";
        char immedStr[128] = "";
        if (decodeExtImmed) {
            switch (alu[i].op) {
            case MME64Op::EXTENDED:
                {
                    MME64Extended::ExtendedEnum op = (MME64Extended::ExtendedEnum)GetBits(15, 12, alu[i].immed);
                    assert(op == MME64Extended::ReadFromSpecial || op == MME64Extended::WriteToSpecial);
                    // Can't fully disassemble the immediate because the address comes from the source (usually immediate),
                    // not fixed to the immediate
                    sprintf(immedStr, "%s, 0x%04x", MME64Extended::mapping[op].name.c_str(), GetBits(11, 0, alu[i].immed));
                }
                break;
            case MME64Op::MERGE:
                {
                    uint32_t srcBit = GetBits(4, 0, alu[i].immed);
                    uint32_t width = GetBits(9, 5, alu[i].immed);
                    uint32_t dstBit = GetBits(14, 10, alu[i].immed);
                    sprintf(immedStr, "%d, %d, %d", srcBit, width, dstBit);
                }
                break;
            case MME64Op::JAL:
                if (GetBits(14, 14, alu[i].immed)) {
                    strcat(immedStr, "ADDSRC0, ");
                }
                if (GetBits(15, 15, alu[i].immed)) {
                    strcat(immedStr, "ADDPC, ");
                }
                if (GetBits(13, 13, alu[i].immed)) {
                    strcat(immedStr, "DS, ");
                }
                {
                    char buf[16];
                    sprintf(buf, "0x%04x", GetBits(12, 0, alu[i].immed));
                    strcat(immedStr, buf);
                }
                break;
            case MME64Op::BLT:
            case MME64Op::BLTU:
            case MME64Op::BLE:
            case MME64Op::BLEU:
            case MME64Op::BEQ:
                if (GetBits(15, 15, alu[i].immed)) {
                    strcat(immedStr, "T, ");
                }
                if (GetBits(14, 14, alu[i].immed)) {
                    strcat(immedStr, "PTAKEN, ");
                }
                if (GetBits(13, 13, alu[i].immed)) {
                    strcat(immedStr, "DS, ");
                }
                {
                    char buf[16];
                    sprintf(buf, "0x%04x", GetBits(12, 0, alu[i].immed));
                    strcat(immedStr, buf);
                }
                break;
            default:
                break;
            }
        }
        if (immedStr[0] == '\0') {
            sprintf(immedStr, "0x%04x", alu[i].immed);
        }
        aluOp += immedStr;

        ops.push_back(aluOp);
    }
    for (int i=0; i<MME64NumOut; i++) {
        if (verbose || output[i] != MME64OutputDefault) {
            string outOp;

            outOp += "METH ";
            outOp += MME64Out::mapping[output[i].method].name + ", ";
            outOp += MME64Out::mapping[output[i].emit].name;

            ops.push_back(outOp);
        }
    }

    string rv;

    for (int i=0; i<(int)ops.size(); i++) {
        if (i != 0) rv += " | ";
        rv += ops[i];
    }

    return rv;
}

namespace mme2 {

static vector<string> split(const string & str, const string & delims=", \t\n")
{
    string::size_type lwr = 0;
    string::size_type next = 0;

    vector<string> tokens;

    while (string::npos != lwr || string::npos != next) {
        if (next != lwr) tokens.push_back(str.substr(lwr, next - lwr));
        lwr = str.find_first_not_of(delims, next);
        next = str.find_first_of(delims, lwr);
    }

    return tokens;
}

static vector<string> tokenize(const string & str, const vector<string> &tokenTypes, const string & skip=" \t\n")
{
    string::size_type lwr = 0;
    string::size_type next = 0;

    vector<string> tokens;

    while (string::npos != lwr || string::npos != next) {
        if (next != lwr) tokens.push_back(str.substr(lwr, next - lwr));
        // Skip the skip parser
        lwr = str.find_first_not_of(skip, next);
        if (lwr == string::npos) break;
        // Find the first token type that matches
        int tt = -1;
        for (int i=0; i<(int)tokenTypes.size(); i++) {
            if (tokenTypes[i].find(str[lwr]) != string::npos) {
                tt = i;
                break;
            }
        }
        if (tt == -1) {
            mmeError("Failed to find matching token type");
        }
        next = str.find_first_not_of(tokenTypes[tt], lwr);
    }

    return tokens;
}

class Parser {
public:
    Parser(vector<string> tokens) : m_lwr(0), m_tokens(tokens) {}
    int remaining() const
    {
        const int tokenSize = static_cast<int>(m_tokens.size());
        assert((tokenSize >= 0)
               && (static_cast<unsigned>(tokenSize) == m_tokens.size()));
        return tokenSize - m_lwr;
    }
    string peek() const
    {
        if (m_lwr >= (int)m_tokens.size()) return "";
        return m_tokens[m_lwr];
    }
    void expect(const string &str)
    {
        if (m_lwr >= (int)m_tokens.size()) mmeError("Unexpected end of input");
        if (m_tokens[m_lwr] != str) mmeError("Unexpected separator");
        m_lwr++;
    }
    MME64Op::OpEnum parseOp()
    {
        if (m_lwr >= (int)m_tokens.size()) mmeError("Unexpected end of input");
        for (int i=0; i<MME64Op::count; i++) {
            if (m_tokens[m_lwr] == MME64Op::mapping[i].name) {
                m_lwr++;
                return MME64Op::mapping[i].val;
            }
        }
        mmeError("Unexpected op name");
        return (MME64Op::OpEnum)0; // Never exelwted
    }
    MME64Reg::RegEnum parseReg()
    {
        if (m_lwr >= (int)m_tokens.size()) mmeError("Unexpected end of input");
        for (int i=0; i<MME64Reg::count; i++) {
            if (m_tokens[m_lwr] == MME64Reg::mapping[i].name) {
                m_lwr++;
                return MME64Reg::mapping[i].val;
            }
        }
        mmeError("Unexpected reg name");
        return (MME64Reg::RegEnum)0; // Never exelwted
    }
    MME64Extended::ExtendedEnum parseExtended()
    {
        if (m_lwr >= (int)m_tokens.size()) mmeError("Unexpected end of input");
        for (int i = 0; i<MME64Extended::count; i++) {
            if (m_tokens[m_lwr] == MME64Extended::mapping[i].name) {
                m_lwr++;
                return MME64Extended::mapping[i].val;
            }
        }
        mmeError("Unexpected extended op name");
        return (MME64Extended::ExtendedEnum)0; // Never exelwted
    }
    bool peekReg()
    {
        if (m_lwr >= (int)m_tokens.size()) mmeError("Unexpected end of input");
        for (int i = 0; i<MME64Reg::count; i++) {
            if (m_tokens[m_lwr] == MME64Reg::mapping[i].name) {
                return true;
            }
        }
        return false;
    }
    MME64Pred::PredEnum parsePred()
    {
        if (m_lwr >= (int)m_tokens.size()) mmeError("Unexpected end of input");
        for (int i = 0; i<MME64Pred::count; i++) {
            if (m_tokens[m_lwr] == MME64Pred::mapping[i].name) {
                m_lwr++;
                return MME64Pred::mapping[i].val;
            }
        }
        mmeError("Unexpected reg name");
        return (MME64Pred::PredEnum)0; // Never exelwted
    }
    MME64Out::OutEnum parseOut()
    {
        if (m_lwr >= (int)m_tokens.size()) mmeError("Unexpected end of input");
        for (int i=0; i<MME64Out::count; i++) {
            if (m_tokens[m_lwr] == MME64Out::mapping[i].name) {
                m_lwr++;
                return MME64Out::mapping[i].val;
            }
        }
        mmeError("Unexpected out name");
        return (MME64Out::OutEnum)0; // Never exelwted
    }
    int parseImmed()
    {
        const char *start = m_tokens[m_lwr].c_str();
        char *end = NULL;

        int rv = strtol(start, &end, 0);

        if (end == NULL || end != start+strlen(start)) {
            mmeError("Cannot parse immediate");
        }

        m_lwr++;

        return rv;
    }
    int parseBasicImmed(const map<string,int> &labels, int pc) {
        int immed = 0;
        // Basic immediates can be either a label or an integer
        if (labels.find(peek()) != labels.end()) {
            int offset = labels.find(peek())->second - pc;
            SetBits(immed, 12, 0, offset);
            expect(peek());
        } else {
            immed = parseImmed();
            if ((((immed << 16) >> 16) != immed) &&
                (immed & 0xffff0000) != 0) {
                mmeError("Immediate must be a sign-extended 16-bit value");
            }
        }
        return immed;
    }
private:
    int m_lwr;
    vector<string> m_tokens;
};

}

using namespace mme2;

std::string llDisassemble2(uint32_t *input) {
    MME64HW bits;

    bits.insert<31,0>(mme::bitset<32>(input[2]));
    bits.insert<63,32>(mme::bitset<32>(input[1]));
    bits.insert<MME64GroupBits,64>(mme::bitset<MME64GroupBits-64+1>(input[0]));

    MME64Group group(bits);

    // Doing verbose disassembly so that reassembly is bit-exact
    return group.disassemble(true, true);
}

vector<uint32_t> llAssemble2(string str, string *optComment, vector<Pragma> &pragmas) {
    vector<string> lines = split(str, "\n");
    vector<string> tokenTypes;
    vector<uint32_t> rv;
    bool started = false;
    bool legalize = true;

    tokenTypes.push_back("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_");
    tokenTypes.push_back(".");
    tokenTypes.push_back(",");
    tokenTypes.push_back("?");
    tokenTypes.push_back("|");
    tokenTypes.push_back("!");

    vector<vector<string> > groups;

    map<string,int> labels;

    for (int i=0; i<(int)lines.size(); i++) {
        string &groupString = lines[i];

        // Skip whitespace and comments
        size_t start = groupString.find_first_not_of(string(" \t"));
        if (start == string::npos ||
            groupString[start] == '#' ||
            groupString.substr(start, start+2) == "//")
            continue;

        // Check for the tag to skip legality check
        if (!groupString.compare(start, 10, "@mme allowillegal")) {
            legalize = false;
        }

        // Ignore anything before the header
        if (!groupString.compare(start, 10, "!!LLMME2.0")) {
            started = true;
            continue;
        } else {
            if (!started) continue;
        }

        size_t colon = groupString.find_first_of(":");
        if (colon != string::npos && colon > 0) {
            const int numGroups = static_cast<int>(groups.size());
            assert((numGroups >= 0)
                   && (static_cast<unsigned>(numGroups) == groups.size()));
            labels.insert(pair<string,int>(groupString.substr(start, colon-start), numGroups));
            continue;
        }

        vector<string> insts = split(groupString, "|");

        if (insts.size() == 0) continue;

        if (insts[0] == "END") break;

        groups.push_back(insts);
    }

    for (int i=0; i<(int)groups.size(); i++) {
        MME64Group group;

        for (int j=0; j<(int)groups[i].size(); j++) {
            string &op = groups[i][j];

#if VERBOSE
            printf(" Assembling op \"%s\"\n", op.c_str());
#endif
            vector<string> tokens = tokenize(op, tokenTypes);
            Parser p(tokens);

#if VERBOSE
            printf(" ");
            for (int t=0; t<(int)tokens.size(); t++) {
                printf(" \"%s\",", tokens[t].c_str());
            }
            printf("\n");
#endif

            if (p.peek() == "?" || p.peek() == "ENDNEXT") {
                // Global op
                while (p.peek() != "") {
                    while (p.peek() == ",") p.expect(",");
                    if (p.peek() == "ENDNEXT") {
                        p.expect("ENDNEXT");
                        group.global.endNext = true;
                    } else if (p.peek() == "?") {
                        p.expect("?");
                        if (p.peek() == "!" || p.peekReg()) {
                            // Legacy mode for trace compatibility
                            if (p.peek() == "!") {
                                p.expect("!");
                                group.global.predMode = MME64Pred::FFFF;
                            } else {
                                group.global.predMode = MME64Pred::TTTT;
                            }
                            group.global.pred = p.parseReg();
                        } else {
                            group.global.predMode = p.parsePred();
                            if (p.peek() == ",") {
                                p.expect(",");
                                group.global.pred = p.parseReg();
                            }
                        }
                    }
                }
            } else if (p.peek() == "METH") {
                // Method op
                p.expect("METH");
                MME64Output output;
                output.method = p.parseOut();
                p.expect(",");
                output.emit = p.parseOut();
                group.output.push_back(output);
            } else {
                // ALU op
                MME64Alu alu = MME64AluDefault;
                alu.op = p.parseOp();
                alu.dst = p.parseReg();
                p.expect(",");
                alu.src[0] = p.parseReg();
                p.expect(",");
                alu.src[1] = p.parseReg();
                if (p.peek() == ",") {
                    p.expect(",");
                    if (p.remaining() > 1) {
                        alu.immed = 0;
                        while (p.remaining() > 0) {
                            // Parse out the extended immediate if we have a matching op
                            switch (alu.op) {
                            case MME64Op::EXTENDED:
                                {
                                    SetBits(alu.immed, 15, 12, p.parseExtended());
                                    p.expect(",");
                                    SetBits(alu.immed, 11, 0, p.parseImmed());
                                }
                                break;
                            case MME64Op::MERGE:
                                {
                                    uint32_t srcBit = p.parseImmed();
                                    if (srcBit > 31) {
                                        mmeError("Invalid MERGE source bit");
                                    }
                                    SetBits(alu.immed, 4, 0, srcBit);

                                    p.expect(",");

                                    uint32_t width = p.parseImmed();
                                    if (width > 31) {
                                        mmeError("Invalid MERGE width");
                                    }
                                    SetBits(alu.immed, 9, 5, width);

                                    p.expect(",");

                                    uint32_t dstBit = p.parseImmed();
                                    if (dstBit > 31) {
                                        mmeError("Invalid MERGE dest bit");
                                    }
                                    SetBits(alu.immed, 14, 10, dstBit);
                                }
                                break;
                            case MME64Op::JAL:
                                if (p.peek() == "ADDSRC0") {
                                    p.expect(p.peek());
                                    SetBits(alu.immed, 14, 14, 1);
                                } else if (p.peek() == "ADDPC") {
                                    p.expect(p.peek());
                                    SetBits(alu.immed, 15, 15, 1);
                                } else if (p.peek() == "DS") {
                                    p.expect(p.peek());
                                    SetBits(alu.immed, 13, 13, 1);
                                } else {
                                    SetBits(alu.immed, 12, 0, p.parseBasicImmed(labels, i));
                                }
                                break;
                            case MME64Op::BLT:
                            case MME64Op::BLTU:
                            case MME64Op::BLE:
                            case MME64Op::BLEU:
                            case MME64Op::BEQ:
                                if (p.peek() == "T") {
                                    p.expect(p.peek());
                                    SetBits(alu.immed, 15, 15, 1);
                                } else if (p.peek() == "PTAKEN") {
                                    p.expect(p.peek());
                                    SetBits(alu.immed, 14, 14, 1);
                                } else if (p.peek() == "DS") {
                                    p.expect(p.peek());
                                    SetBits(alu.immed, 13, 13, 1);
                                } else {
                                    SetBits(alu.immed, 12, 0, p.parseBasicImmed(labels, i));
                                }
                                break;
                            default:
                                mmeError("Unknown extended immediate op");
                            }
                            if (p.peek() == ",") {
                                p.expect(",");
                            }
                        }
                    } else {
                        alu.immed = p.parseBasicImmed(labels, i);
                    }
                }
                group.alu.push_back(alu);
            }
        }

        group.validate();
        if (legalize) {
            group.legalize();
        }

#if VERBOSE
        printf("Disassembly: %s\n", group.disassemble(false).c_str());
        printf("Verbose:  %s\n", group.disassemble(true).c_str());
#endif

        MME64HW hwInst = group.pack();
        MME64Group checkGroup(hwInst);
#if VERBOSE
        printf("Unpacked: %s\n", checkGroup.disassemble(true).c_str());
#endif

        assert(checkGroup.disassemble(true) == group.disassemble(true));
        assert(checkGroup.disassemble(true, true) == group.disassemble(true, true));

        // ZZZ Scalability
        rv.push_back(hwInst.extract<MME64GroupBits,64>().toUint());
        rv.push_back(hwInst.extract<63,32>().toUint());
        rv.push_back(hwInst.extract<31,0>().toUint());
    }

    // Find and run the tests
    parse(str.substr(0, str.find("!!LLMME")), pragmas);

    return rv;
}
