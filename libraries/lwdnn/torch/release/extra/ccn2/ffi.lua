local ffi = require 'ffi'

ffi.cdef[[
void colwFilterActs(THCState* state, THLwdaTensor* images, THLwdaTensor* filters, THLwdaTensor* targets,
                    int imgSizeY, int numModulesY, int numModulesX,
                    int paddingStart, int moduleStride,
                    int numImgColors, int numGroups);
void colwFilterActsSt(THCState* state, THLwdaTensor* images, THLwdaTensor* filters, THLwdaTensor* targets,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int numImgColors, int numGroups,
                   float scaleTargets, float scaleOutput);

void localFilterActs(THCState* state, THLwdaTensor* images, THLwdaTensor* filters, THLwdaTensor* targets,
                     int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                     int numImgColors, int numGroups);
void localFilterActsSt(THCState* state, THLwdaTensor* images, THLwdaTensor* filters, THLwdaTensor* targets,
                     int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                     int numImgColors, int numGroups,
                     float scaleTargets, float scaleOutput);

void colwImgActs(THCState* state, THLwdaTensor* hidActs, THLwdaTensor* filters, THLwdaTensor* targets,
                 int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups);
void colwImgActsSt(THCState* state, THLwdaTensor* hidActs, THLwdaTensor* filters, THLwdaTensor* targets,
                 int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
                 float scaleTargets, float scaleOutput);

void localImgActs(THCState* state, THLwdaTensor* hidActs, THLwdaTensor* filters, THLwdaTensor* targets,
                  int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups);
void localImgActsSt(THCState* state, THLwdaTensor* hidActs, THLwdaTensor* filters, THLwdaTensor* targets,
                  int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
                  float scaleTargets, float scaleOutput);

void colwWeightActs(THCState* state, THLwdaTensor* images, THLwdaTensor* hidActs, THLwdaTensor* targets,
                    int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart,
                    int moduleStride, int numImgColors, int numGroups, int sumWidth);
void colwWeightActsSt(THCState* state, THLwdaTensor* images, THLwdaTensor* hidActs, THLwdaTensor* targets,
                    int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride,
                    int numImgColors, int numGroups, int sumWidth,
                    float scaleTargets, float scaleOutput);

void localWeightActs(THCState* state, THLwdaTensor* images, THLwdaTensor* hidActs, THLwdaTensor* targets,
                     int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart,
                     int moduleStride, int numImgColors, int numGroups);

void localWeightActsSt(THCState* state, THLwdaTensor* images, THLwdaTensor* hidActs, THLwdaTensor* targets,
                     int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride,
                     int numImgColors, int numGroups, float scaleTargets, float scaleOutput);

void addBias(THCState* state, THLwdaTensor* output, THLwdaTensor* bias);
void gradBias(THCState* state, THLwdaTensor* output, THLwdaTensor* gradBias, float scale);

void addSumCols(THCState* state, THLwdaTensor*output, THLwdaTensor*input); // used for partialSum

void colwLocalMaxPool(THCState* state, THLwdaTensor* images, THLwdaTensor* target, int numFilters,
                      int subsX, int startX, int strideX, int outputsX);
void colwLocalMaxUndo(THCState* state, THLwdaTensor* images, THLwdaTensor* maxGrads, THLwdaTensor* maxActs, THLwdaTensor* target,
                      int subsX, int startX, int strideX, int outputsX);

void colwCrossMapMaxPool(THCState* state, THLwdaTensor* images, THLwdaTensor* target, const int startF, const int poolSize,
                         const int numOutputs, const int stride, const int imgSize);
void colwCrossMapMaxPoolUndo(THCState* state, THLwdaTensor* images, THLwdaTensor* maxGrads, THLwdaTensor* maxActs, THLwdaTensor* target,
                             const int imgSize, const int startF, const int poolSize,
                             const int stride, const float scaleTargets, const float scaleOutputs);

void colwLocalAvgPool(THCState* state, THLwdaTensor* images, THLwdaTensor* target, int numFilters,
                      int subsX, int startX, int strideX, int outputsX);
void colwLocalAvgUndo(THCState* state, THLwdaTensor* avgGrads, THLwdaTensor* target,
                      int subsX, int startX, int strideX, int outputsX, int imgSize);

void colwResponseNorm(THCState* state, THLwdaTensor* images, THLwdaTensor* denoms, THLwdaTensor* target, int numFilters, int sizeX, float addScale, float powScale, float minDiv);
void colwResponseNormUndo(THCState* state, THLwdaTensor* outGrads, THLwdaTensor* denoms, THLwdaTensor* inputs, THLwdaTensor* acts, THLwdaTensor* target, int numFilters,
                         int sizeX, float addScale, float powScale, float scaleTargets, float scaleOutput);

void colwContrastNorm(THCState* state, THLwdaTensor* images, THLwdaTensor* meanDiffs, THLwdaTensor* denoms, THLwdaTensor* target, int numFilters, int sizeX, float addScale, float powScale, float minDiv);
void colwContrastNormUndo(THCState* state, THLwdaTensor* outGrads, THLwdaTensor* denoms, THLwdaTensor* meanDiffs, THLwdaTensor* acts, THLwdaTensor* target, int numFilters,
                         int sizeX, float addScale, float powScale, float scaleTargets, float scaleOutput);

void colwResponseNormCrossMap(THCState* state, THLwdaTensor* images, THLwdaTensor* target, int numFilters, int sizeF, float addScale,
                              float powScale, float minDiv, bool blocked);
void colwResponseNormCrossMapUndo(THCState* state, THLwdaTensor* outGrads, THLwdaTensor* inputs, THLwdaTensor* acts, THLwdaTensor* target, int numFilters,
                         int sizeF, float addScale, float powScale, float minDiv, bool blocked, float scaleTargets, float scaleOutput);

void colwResizeBilinear(THCState* state, THLwdaTensor* images, THLwdaTensor* target, int imgSize, int tgtSize, float scale);
]]

local path = package.searchpath('libccn2', package.cpath)
if not path then
   path = require 'ccn2.config'
end
assert(path, 'could not find libccn2.so')
ccn2.C = ffi.load(path)
