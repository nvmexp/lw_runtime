#ifndef LWTENSOR_INTERNAL_ELEMENTWISE_PROTOTYPE_H
#define LWTENSOR_INTERNAL_ELEMENTWISE_PROTOTYPE_H

#include <lwda_runtime.h>

#include <lwtensor/internal/types.h>

namespace LWTENSOR_NAMESPACE
{
    class Context;
    lwtensorStatus_t elementwiseTrinaryExelwte(
            const Context *ctx,
            const void * alpha, const void * A,
            const void *  beta, const void * B,
            const void * gamma, const void * C, void * const D,
            const ElementwisePlan & plan, const lwdaStream_t stream);
    lwtensorStatus_t elementwiseBinaryExelwte(
            const Context *ctx,
            const void * const alpha, const void * const A,
            const void * const gamma, const void * const C,
                                            void * const D,
            const ElementwisePlan & plan, const lwdaStream_t stream);
    lwtensorStatus_t permutationExelwte(
            const Context *ctx,
            const void * const alpha, const void * const A,
                                            void * const D,
            const ElementwisePlan & plan, const lwdaStream_t stream);
    /**
     * \brief Compute the order of the modes that the elementwise kernels will proceed.
     * \param [in] modeA the modes in tensor A
     * \param [in] modeB the modes in tensor B
     * \param [in] modeC the modes in tensor C (which contains all modes)
     * \param [in] useB whether tensor B is provided
     * \param [inout] mode_order the unique order that the elementwise kernels will proceed
     * \post the vectorized mode of C (if present) must be the first mode in mode_order;
     * similarly, if A is vectorized it must be the second mode
     */
    void getModeOrder(
            const ModeList &modeA,
            const ModeList &modeB,
            const ModeList &modeC,
            const bool useB,
            ModeList &mode_order);

    /**
     *  \brief Validate the input arguments of lwtensorElementwiseInternal_L0.
     *  \return the error code
     */
    lwtensorStatus_t pwValidateInput(
            const Context *ctx,
            const void* alpha, const TensorDescriptor* descA, const mode_type* modeA,
            const void* beta,  const TensorDescriptor* descB, const mode_type* modeB,
            const void* gamma, const TensorDescriptor* descC, const mode_type* modeC,
                               const TensorDescriptor* descD,
            const lwtensorOperator_t opA,
            const lwtensorOperator_t opB,
            const lwtensorOperator_t opC,
            const lwtensorOperator_t opAB,
            const lwtensorOperator_t opABC,
            const lwdaDataType_t typeCompute, const bool useA, const bool useB, const bool useC);

    /**
     *  \brief The L0 function funnels the three public elementwise functions (ElementwiseTrinary,
     *  ElementwiseBinary, and Permutation) into one unique functions. Input arguments are validated
     *  here. The modes and extents of each tensor are sorted using its strides as keys. We then
     *  fuse (collapse) modes that are contiguous in more than two tensors for optimization.
     *  Finally, we call the L1 function that generates the inputs for the LWCA kernels.
     *  \return the error code
     */
    lwtensorStatus_t lwtensorElementwiseInternal_L0(
            const Context *ctx,
            const void * alpha, const TensorDescriptor *descA, const mode_type* modeA, const uint32_t alignmentRequirementA,
            const void * beta,  const TensorDescriptor *descB, const mode_type* modeB, const uint32_t alignmentRequirementB,
            const void * gamma, const TensorDescriptor *descC, const mode_type* modeC, const uint32_t alignmentRequirementC,
                                const TensorDescriptor * const descD, const mode_type* const modeD, const uint32_t alignmentRequirementD,
            lwtensorOperator_t opA,
            lwtensorOperator_t opB,
            lwtensorOperator_t opC,
            lwtensorOperator_t opAB,
            lwtensorOperator_t opABC,
            const lwdaDataType_t typeCompute, ElementwisePlan * plan);

    /**
     *  \brief The L1 function generate the arguments for GPU kernels. The arguments are passed into
     *  the L2 function that contains a decision tree that branches according to the data types,
     *  operator types, and the leading strides of each tensor.
     *  \return the error code
     */
    lwtensorStatus_t lwtensorElementwiseInternal_L1(
            const Context *ctx,
            const void* const alpha, const lwdaDataType_t typeA, const StrideMap &strideA, const ModeList &modeA, const uint32_t alignmentRequirementA,
            const void* const beta,  const lwdaDataType_t typeB, const StrideMap &strideB, const ModeList &modeB, const uint32_t alignmentRequirementB,
            const void* const gamma, const lwdaDataType_t typeC, const StrideMap &strideC, const ModeList &modeC, const uint32_t alignmentRequirementC, const uint32_t alignmentRequirementD,
            const ExtentMap &extent,
            const lwtensorOperator_t opA,
            const lwtensorOperator_t opB,
            const lwtensorOperator_t opC,
            const lwtensorOperator_t opAB,
            const lwtensorOperator_t opABC,
            const lwdaDataType_t typeCompute,
            ElementwisePlan *plan);


    lwtensorStatus_t elementwiseTrinaryCreate(
            const Context *ctx,
            const void* alpha, const TensorDescriptor & descA, const int32_t * modeA, const uint32_t alignmentRequirementA,
            const void* beta,  const TensorDescriptor & descB, const int32_t * modeB, const uint32_t alignmentRequirementB,
            const void* gamma, const TensorDescriptor & descC, const int32_t * modeC, const uint32_t alignmentRequirementC,
            const TensorDescriptor & descD, const int32_t * modeD, const uint32_t alignmentRequirementD,
            const lwtensorOperator_t opAB, 
            const lwtensorOperator_t opABC, 
            const lwdaDataType_t typeCompute, ElementwisePlan & plan);
    lwtensorStatus_t elementwiseBinaryCreate(
            const Context *ctx,
            const void * alpha, const TensorDescriptor & descA, const int32_t * modeA, const uint32_t alignmentRequirementA,
            const void * gamma, const TensorDescriptor & descC, const int32_t * modeC, const uint32_t alignmentRequirementC,
            const TensorDescriptor & descD, const int32_t * modeD, const uint32_t alignmentRequirementD,
            const lwtensorOperator_t opAC, const lwdaDataType_t typeCompute,
            ElementwisePlan & plan);
    lwtensorStatus_t permutationCreate(
            const Context *ctx,
            const void * alpha, const TensorDescriptor & descA, const int32_t * modeA, const uint32_t alignmentRequirementA,
            const TensorDescriptor & descD, const int32_t * modeD, const uint32_t alignmentRequirementD,
            const lwdaDataType_t typeCompute,
            ElementwisePlan & plan);
} /* end namespace LWTENSOR_NAMESPACE */

#endif /* define LWTENSOR_INTERNAL_ELEMENTWISE_PROTOTYPE_H */
