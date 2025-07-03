#include <string>

namespace MME64Out {
enum OutEnum {
    NONE,
    ALU0,
    ALU1,
    LOAD0,
    LOAD1,
    IMMED0,
    IMMED1,
    RESERVED,
    IMMEDHIGH0,
    IMMEDHIGH1,
    IMMED32_0,
};

    enum { count = 11 };
    enum { bits = 4 };

static const struct { std::string name; OutEnum val;} mapping[] = {
    {"NONE", NONE },
    {"ALU0", ALU0 },
    {"ALU1", ALU1 },
    {"LOAD0", LOAD0 },
    {"LOAD1", LOAD1 },
    {"IMMED0", IMMED0 },
    {"IMMED1", IMMED1 },
    {"RESERVED", RESERVED },
    {"IMMEDHIGH0", IMMEDHIGH0 },
    {"IMMEDHIGH1", IMMEDHIGH1 },
    {"IMMED32_0", IMMED32_0 },
};
};
