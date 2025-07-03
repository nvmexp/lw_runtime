#include <string>

namespace MME64Pred {
enum PredEnum {
    UUUU,
    TTTT,
    FFFF,
    TTUU,
    FFUU,
    TFUU,
    TUUU,
    FUUU,
    UUTT,
    UUTF,
    UUTU,
    UUFT,
    UUFF,
    UUFU,
    UUUT,
    UUUF,
};

    enum { count = 16 };
    enum { bits = 4 };

static const struct { std::string name; PredEnum val;} mapping[] = {
    {"UUUU", UUUU },
    {"TTTT", TTTT },
    {"FFFF", FFFF },
    {"TTUU", TTUU },
    {"FFUU", FFUU },
    {"TFUU", TFUU },
    {"TUUU", TUUU },
    {"FUUU", FUUU },
    {"UUTT", UUTT },
    {"UUTF", UUTF },
    {"UUTU", UUTU },
    {"UUFT", UUFT },
    {"UUFF", UUFF },
    {"UUFU", UUFU },
    {"UUUT", UUUT },
    {"UUUF", UUUF },
};
};
