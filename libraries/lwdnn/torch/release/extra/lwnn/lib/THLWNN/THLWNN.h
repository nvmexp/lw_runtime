#include <THC/THC.h>

#define THCIndexTensor THLwdaLongTensor
#define THCIndexTensor_(NAME) THLwdaLongTensor_ ## NAME
typedef long THCIndex_t;

#define THNN_(NAME) TH_CONCAT_3(THNN_, CReal, NAME)

#include "generic/THLWNN.h"
#include <THC/THCGenerateFloatTypes.h>
