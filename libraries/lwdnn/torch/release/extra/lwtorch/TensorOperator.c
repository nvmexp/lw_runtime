#include "torch/utils.h"
#include "luaT.h"
#include "THC.h"

#include "THCTensorMath.h"

#define lwtorch_TensorOperator_(NAME) TH_CONCAT_4(lwtorch_,CReal,TensorOperator_,NAME)
#define torch_Tensor_(NAME) TH_CONCAT_4(torch_,CReal,Tensor_,NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,CReal,Tensor)
#define lwtorch_Tensor_(NAME) TH_CONCAT_4(lwtorch_,CReal,Tensor_,NAME)

#include "generic/TensorOperator.c"
#include "THCGenerateAllTypes.h"
