#ifndef TH_GENERIC_FILE
#error "You must define TH_GENERIC_FILE before including THGenerateFloatTypes.h"
#endif

#ifndef THGenerateManyTypes
#define THFloatLocalGenerateManyTypes
#define THGenerateManyTypes
#endif

#include "THGenerateFloatType.h"
#include "THGenerateDoubleType.h"

#ifdef THFloatLocalGenerateManyTypes
#undef THFloatLocalGenerateManyTypes
#undef THGenerateManyTypes
#undef TH_GENERIC_FILE
#endif
