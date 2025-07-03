#include <string>

namespace MME64Special {
enum SpecialEnum {
    LoadSource,
    MmeConfig,
    PTimerHigh,
    PTimerLow,
    Scratch,
};

    enum { count = 5 };
    enum { bits = 3 };

static const struct { std::string name; SpecialEnum val;} mapping[] = {
    {"LoadSource", LoadSource },
    {"MmeConfig", MmeConfig },
    {"PTimerHigh", PTimerHigh },
    {"PTimerLow", PTimerLow },
    {"Scratch", Scratch },
};
};
