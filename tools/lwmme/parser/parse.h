#ifndef _PARSE_H
#define _PARSE_H 1

#include <vector>
#include <string>
#include <stdint.h>

#include <assert.h>

#define ARRAYSIZE(x) (sizeof(x)/sizeof((x)[0]))


class Pragma {
public:
    std::string name;
    std::vector<unsigned int> exprValues;
    std::string stringValue;
    std::vector<unsigned int> opArgValues;
    std::string opName;

    void expandOp();
};

static inline bool charEquali(char a, char b) {
    return toupper(a) == toupper(b);
}

static inline bool stringEquali(const std::string& s1, const std::string& s2) {
    return (s1.size() == s2.size())
        && equal(s1.begin(), s1.end(), s2.begin(), charEquali);
}

class Symbol {
public:
    Symbol() : num(0), width(-1) {;}
    Symbol(unsigned int n, int w) : num(n), width(w) {;}

    unsigned int getNum() const { assert(width != 32); return num; }
    int getWidth() const { assert(width != 32); return width; }

    unsigned int &getNum() { return num; }

    // Should only be used by the parser, but because the generated code is C
    // we don't have an easy way to restrict it
    void setWidth(int w) { width = w; }

    bool operator==(const Symbol &other) const { return num == other.num && width == other.width; }
    bool operator!=(const Symbol &other) const { return !((*this) == other); }

    bool operator<(const Symbol &other) const { return ((uint64_t(width) << 32) | num) < ((uint64_t(other.width) << 32) | other.num); }

    Symbol operator++(int) { 
        Symbol rv = *this;

        num += width;

        return rv;
    }
private:
    unsigned int num;
    int width;
};

struct TokenInfo {
    TokenInfo() { ptr = NULL; i = 0; }
    TokenInfo(const int64_t &x) { i=x; ptr=NULL; }
    TokenInfo(void *const &x) { ptr=x; i=0; }
    void *ptr;
    int64_t i;
    operator int64_t () { return i; }
};
#define YYSTYPE TokenInfo

enum CompareSubop {
    CMP_EQ,
    CMP_NE,
    CMP_GT,
    CMP_GE,
    CMP_LT,
    CMP_LE,
};

// Source-level representation
enum ScanOpcode {
    // Unary
    SOP_MOV,
    SOP_NOT,

    // Binary
    SOP_ADD,
    SOP_ADDC,
    SOP_SUB,
    SOP_SUBB,
    SOP_OR,
    SOP_AND,
    SOP_XOR,
    SOP_NAND,
    SOP_ANDNOT,
    SOP_MUL,
    SOP_MULU,
    SOP_SLL,
    SOP_SRL,
    SOP_SRA,

    // Flow control
    SOP_IF_EQ,
    SOP_IF_NE,
    SOP_IF_GT,
    SOP_IF_GE,
    SOP_IF_LT,
    SOP_IF_LE,
    SOP_WHILE_EQ,
    SOP_WHILE_NE,
    SOP_WHILE_GT,
    SOP_WHILE_GE,
    SOP_WHILE_LT,
    SOP_WHILE_LE,
    SOP_LOOP,
    SOP_LOOP_SW,
    SOP_ELSE,
    SOP_END,
    SOP_SWITCH,
    SOP_CASE,
    SOP_DEFAULT,

    // Internal opcodes to reorganize from the raw scanned representation
    SOP_IF,
    SOP_WHILE,
    SOP_IFE_END,
    SOP_WHILE_END,
    SOP_LOOP_END,
    SOP_LOOP_SW_END,
    SOP_SWITCH_END,
    SOP_CASE_END,
    SOP_DEFAULT_END,

    // Special
    SOP_EMIT,
    SOP_METHOD,
    SOP_EMITONE,
    SOP_LOAD,
    SOP_MERGE,
    SOP_STATE,
    SOP_DREAD,
    SOP_DWRITE,
    SOP_LOADMETH,
    SOP_LOADFIFO,

    // 64-bit
    SOP_ADD64, 
    SOP_SUB64,
    SOP_MUL64,
    SOP_MULU64,

    // Used for the parser - shouldn't ever exist inside the compiler
    SOP_ILWALID,
};

enum ScanType {
    ST_NULL,
    ST_REG,
    ST_CONST,
};

struct ScanOperand {
    ScanType type;
    int constWidth;
    int64_t constVal;
    Symbol reg;
    ScanOperand() { type = ST_NULL; constVal = 0; reg = Symbol(); constWidth = 1; }
    ScanOperand(int64_t c) { type = ST_CONST; constVal = c; reg = Symbol(); constWidth = 1; }
    ScanOperand(Symbol r) { type = ST_REG; reg = r; constVal = 0; constWidth = 1; }
};

struct ScanSlice {
    bool constant;
    Symbol reg;
    int lower;
    int upper;
    ScanSlice() {
        constant = true;
        reg = Symbol();
        lower = upper = -1;
    }
    ScanSlice(Symbol r) { constant = false; reg = r; }
    ScanSlice(int u, int l) { constant = true; upper=u; lower=l; }
};  

struct ScanOp {
    ScanOpcode op;
    CompareSubop cmp;
    ScanOperand dst;
    ScanOperand src[2];
    ScanSlice dstSlice;
    ScanSlice src1Slice;
    int label;
    int labelTop;    // Label of the top of the current block (IF, WHILE, etc) or -1 if not a block bottom
    int labelBottom; // Label of the bottom of the current block (ELSE or END) or -1 if not a block top
    int flowNest;    // Number of flow control levels encompassing but not including this instruction 
    std::string filename;
    int lineno;
    int lastPragma;

    ScanOp() { label = labelTop = labelBottom = -1; op = SOP_ILWALID; cmp = CMP_EQ; flowNest = -1; /*dst = src[0] = src[1] = Symbol(0, 1);*/ filename = std::string("xxx"); lineno = -1; lastPragma = -1; }
    bool writesReg(Symbol reg) {
        return dst.type == ST_REG && dst.reg == reg;
    }
    bool readsReg(Symbol reg) {
        for (int sr=0; sr<2; sr++) {
            if (src[sr].type == ST_REG && src[sr].reg == reg) return true;
        }
        if (!dstSlice.constant && dstSlice.reg == reg) return true;
        if (!src1Slice.constant && src1Slice.reg == reg) return true;
        return false;
    }
    bool usesReg(Symbol reg) { return writesReg(reg) || readsReg(reg); }
};

// The extern mmeError is expected to abort (exit, exception, whatever) if called
extern void mmeError(std::string s, const ScanOp *op = NULL);
extern void mmeWarning(std::string category, std::string msg);
extern void mmeTrace(const char *msg);

std::vector<ScanOp> parse(std::string text, std::vector<Pragma> &outPragmas);

#endif //ndef _PARSE_H
