#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricFullColwolution.c"
#else

void THNN_(VolumetricFullColwolution_updateOutput)(
  THNNState *state,
  THTensor *input,          // 4D or 5D (batch) tensor
  THTensor *output,
  THTensor *weight,         // weight tensor (nInputPlane x nOutputPlane x kT x kH x kW)
  THTensor *bias,
  THTensor *finput,         // internal columns buffer
  THTensor *fgradInput,     // internal ones buffer
  int dT, int dW, int dH,   // stride of the colwolution
  int pT, int pW, int pH,   // padding
  int aT, int aW, int aH)   // extra output adjustment
{
  THNN_(VolumetricFullDilatedColwolution_updateOutput)(
      state, input, output, weight, bias, finput, fgradInput,
      dT, dW, dH, pT, pW, pH, 1, 1, 1, aT, aW, aH);
}

void THNN_(VolumetricFullColwolution_updateGradInput)(
  THNNState *state,
  THTensor *input,
  THTensor *gradOutput,
  THTensor *gradInput,
  THTensor *weight,
  THTensor *finput,
  THTensor *fgradInput,     // only used by lwca impl
  int dT, int dW, int dH,   // stride
  int pT, int pW, int pH,   // padding
  int aT, int aW, int aH)   // extra output adjustment
{
  THNN_(VolumetricFullDilatedColwolution_updateGradInput)(
      state, input, gradOutput, gradInput, weight, finput, fgradInput,
      dT, dW, dH, pT, pW, pH, 1, 1, 1, aT, aW, aH);
}

void THNN_(VolumetricFullColwolution_accGradParameters)(
  THNNState *state,
  THTensor *input,
  THTensor *gradOutput,
  THTensor *gradWeight,
  THTensor *gradBias,
  THTensor *finput,
  THTensor *fgradInput,
  int dT, int dW, int dH,   // stride
  int pT, int pW, int pH,   // padding
  int aT, int aW, int aH,   // extra output adjustment
  accreal scale_)
{
  THNN_(VolumetricFullDilatedColwolution_accGradParameters)(
      state, input, gradOutput, gradWeight, gradBias, finput, fgradInput,
      dT, dW, dH, pT, pW, pH, 1, 1, 1, aT, aW, aH, scale_);
}

#endif
