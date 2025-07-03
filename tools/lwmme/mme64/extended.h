#include <string>

namespace MME64Extended {
enum ExtendedEnum {
    ReadFromSpecial,
    WriteToSpecial,
};

    enum { count = 2 };
    enum { bits = 1 };

static const struct { std::string name; ExtendedEnum val;} mapping[] = {
    {"ReadFromSpecial", ReadFromSpecial },
    {"WriteToSpecial", WriteToSpecial },
};
};
