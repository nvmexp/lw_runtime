#ifndef THC_GENERIC_FILE
#error "You must define THC_GENERIC_FILE before including THGenerateIntType.h"
#endif

#define real int
#define accreal long
#define Real Int
#define CReal LwdaInt
#define THC_REAL_IS_INT
#line 1 THC_GENERIC_FILE
#include THC_GENERIC_FILE
#undef real
#undef accreal
#undef Real
#undef CReal
#undef THC_REAL_IS_INT

#ifndef THCGenerateAllTypes
#undef THC_GENERIC_FILE
#endif
