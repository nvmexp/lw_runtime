#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialFullColwolution.c"
#else

void THNN_(SpatialFullColwolution_updateOutput)(
    THNNState *state,
    THTensor *input,
    THTensor *output,
    THTensor *weight,
    THTensor *bias,
    THTensor *columns,
    THTensor *ones,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int adjW, int adjH)
{
  THNN_(SpatialFullDilatedColwolution_updateOutput)(
    state, input, output, weight, bias, columns, ones,
    kW, kH, dW, dH, padW, padH, 1, 1, adjW, adjH);
  }

void THNN_(SpatialFullColwolution_updateGradInput)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradInput,
    THTensor *weight,
    THTensor *gradColumns,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int adjW, int adjH)
{
  THNN_(SpatialFullDilatedColwolution_updateGradInput)(
    state, input, gradOutput, gradInput, weight, gradColumns,
    kW, kH, dW, dH, padW, padH, 1, 1, adjW, adjH);
}

void THNN_(SpatialFullColwolution_accGradParameters)(
    THNNState *state,
    THTensor *input,
    THTensor *gradOutput,
    THTensor *gradWeight,
    THTensor *gradBias,
    THTensor *columns,
    THTensor *ones,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int adjW, int adjH,
    accreal scale_)
{
THNN_(SpatialFullDilatedColwolution_accGradParameters)(
    state, input, gradOutput, gradWeight, gradBias, columns, ones,
    kW, kH, dW, dH, padW, padH, 1, 1, adjW, adjH, scale_);
}

#endif
