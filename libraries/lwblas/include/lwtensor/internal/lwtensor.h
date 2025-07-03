#pragma once

/**
 * A colwolved mode is used for tensor colwolutions (via lwtensorColwolution()) to establish a
 * connection between a mode in the activation, filter and output tensor.
 */
typedef struct
{
    int32_t modeActivation; //< mode in the activation tensor
    int32_t modeFilter;     //< mode in the filter tensor
    int32_t modeOutput;     //< mode in the output tensor
    uint32_t padding;       //< padding for the activation tensor along this mode
    int32_t stride;         //< stride of the filter within the activation tensor
    int32_t dilation;       //< dilation along this mode
} lwtensorColwolvedMode_t;

/**
 * \brief Opaque structure for the colwolution descriptor
 */
typedef struct { int64_t fields[256]; /*!< Data */ } lwtensorColwolutionDescriptor_t;


lwtensorStatus_t lwtensorInitTensorDescriptor(const lwtensorHandle_t* handle,
                                              lwtensorTensorDescriptor_t* desc_,
                                              const uint32_t numModes,
                                              const int64_t * const extent,
                                              const int64_t * const stride,
                                              const lwdaDataType_t dataType,
                                              const lwtensorOperator_t op,
                                              const uint32_t vectorWidth,
                                              const uint32_t vectorModeIndex);

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */
lwtensorStatus_t lwtensorContractionDescriptorInfo(const lwtensorHandle_t* handle,
                                                   const lwtensorContractionDescriptor_t* desc,
                                                   char* dst, int sz);

lwtensorStatus_t lwtensorContractionPlanInfo(const lwtensorHandle_t* handle,
                                                   const lwtensorContractionPlan_t* plan,
                                                   char* dst, int sz);

/**
 * \brief Limits the search space of viable candidates (a.k.a. algorithms)
 *
 * \param[in] handle Opaque handle holding lwTENSOR's library context.
 * \param[out] find
 * \param[in] kernel Allows users to select a kernel that is uses with the previously selected algo.
 * \retval LWTENSOR_STATUS_SUCCESS
 * \retval LWTENSOR_STATUS_ILWALID_VALUE
 * \retval LWTENSOR_STATUS_NOT_SUPPORTED
 */
lwtensorStatus_t lwtensorContractionFindSetKernel(const lwtensorHandle_t* handle,
                                                  lwtensorContractionFind_t* const find,
                                                  int kernel);

/**
 */
lwtensorStatus_t lwtensorInitColwolvedMode(
        const lwtensorHandle_t* handle,
        lwtensorColwolvedMode_t* colwolvedMode,
        uint32_t padding, int32_t stride, int32_t dilation,
        int32_t modeActivation, int32_t modeFilter, int32_t modeOutput);

/**
 * Initializes the colwolution descriptor.
 *
 * TODO add descA, .. here
 */
lwtensorStatus_t lwtensorInitColwolutionDescriptor(
        const lwtensorHandle_t* handle,
        lwtensorContractionDescriptor_t *desc,
        const lwtensorTensorDescriptor_t *descActivation_, const int32_t modeActivation[], uint32_t alignmentRequirementA,
        const lwtensorTensorDescriptor_t *descFilter_,     const int32_t modeFilter[],     uint32_t alignmentRequirementB,
        const lwtensorTensorDescriptor_t *descOutput_,     const int32_t modeOutput[],     uint32_t alignmentRequirementC,
        const uint32_t numColwolvedModes, const lwtensorColwolvedMode_t colwolvedModes[],
        const uint32_t numGroups,
        const lwtensorComputeType_t typeCompute,
        const lwtensorOperator_t opOut);

/**
 * Computes the extent of the output tensor for a specific colwolution.
 *
 * \param[in] modeOutput 
 * \param[out] extent numModes-dimensional array that will contain the extents for the
 * output tensor for the specified colwolution, activation tensor and filter tensor.
 */
lwtensorStatus_t lwtensorColwolutionGetOutput(
    const lwtensorHandle_t* handle,
    const lwtensorColwolutionDescriptor_t *descColw,
    const lwtensorTensorDescriptor_t      *descActivation,
    const lwtensorTensorDescriptor_t      *descFilter,
    uint32_t                              numModes,
    int64_t                               extent[]);

#if defined(__cplusplus)
}
#endif /* __cplusplus */
