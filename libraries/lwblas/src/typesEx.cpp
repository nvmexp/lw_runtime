#include <unordered_map>
#include <list>
#include <limits>
#include <assert.h>

#include <lwtensor/internal/exceptions.h>
#include <lwtensor/internal/elementwisePrototype.h>
#include <lwtensor/internal/reduction.h>
#include <lwtensor/internal/typesEx.h>
#include <lwtensor/internal/contractionDescriptor.h>
#include <lwtensor/internal/computeEngine.h>
#include <lwtensor/internal/tensorContraction.h>
#include <lwtensor/internal/utilEx.h>
#include <lwtensor/internal/defines.h>
#include <lwtensor/internal/context.h>
#include <lwtensor/internal/lwblasLtHandles.h>

namespace LWTENSOR_NAMESPACE 
{

    lwtensorStatus_t ContractionDescriptor::initContractionDescriptor(
                          const Context* ctx,
                          const TensorDescriptor* descA, const int32_t modeA[], const uint32_t alignmentRequirementA,
                          const TensorDescriptor* descB, const int32_t modeB[], const uint32_t alignmentRequirementB,
                          const TensorDescriptor* descC, const int32_t modeC[], const uint32_t alignmentRequirementC,
                          const TensorDescriptor* descD, const int32_t modeD[], const uint32_t alignmentRequirementD,
                          lwtensorComputeType_t minimumComputeType)
    {
        this->unsetInitialized();
        if( !isValidComputeType(minimumComputeType) )
        {
            return ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "typeCompute is invalid.");
        }

        if (descC != descD)
        {
            return ctx->logError(LWTENSOR_STATUS_NOT_SUPPORTED,
                    "Current limitation: descC and descD must be identical "
                    "for now (please request this feature).");
        }

        if (modeC != modeD)
        {
            for(int i=0; i < descD->getNumModes(); ++i)
            {
                if (modeC[i] != modeD[i])
                {
                    return ctx->logError(LWTENSOR_STATUS_NOT_SUPPORTED,
                            "Current limitation: modeC and modeD must be identical "
                            "for now (please request this feature).");
                }
            }
        }
        
        if (descA == nullptr)
        {
            return ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "Invalid argument: descA is nullptr.");
        }
        else if (modeA == nullptr)
        {
            return ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "Invalid argument: modeA is nullptr.");
        }
        else if (descB == nullptr)
        {
            return ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "Invalid argument: descB is nullptr.");
        }
        else if (modeB == nullptr)
        {
            return ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "Invalid argument: modeB is nullptr.");
        }
        else if (descC == nullptr)
        {
            return ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "Invalid argument: descC is nullptr.");
        }
        else if (modeC == nullptr && descC->getNumModes() > 0)
        {
            return ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "Invalid argument: modeC is nullptr.");
        }
        else if (descD == nullptr)
        {
            return ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "Invalid argument: descD is nullptr.");
        }
        else if (modeD == nullptr && descD->getNumModes() > 0)
        {
            return ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "Invalid argument: modeD is nullptr.");
        }

        if (descA->isVectorized() || (descB && descB->isVectorized()) || descC->isVectorized())
        {
            return ctx->logError(LWTENSOR_STATUS_NOT_SUPPORTED, "Tensor contractions do not support ''vectorized'' tensors.");
        }

        ModeRenamer renamer;
        ModeList renameModeA = renamer.rename(modeA, descA->getNumModes());
        modeA = renameModeA.data();
        ModeList renameModeB = renamer.rename(modeB, descB->getNumModes());
        modeB = renameModeB.data();
        ModeList renameModeC = renamer.rename(modeC, descC->getNumModes());
        modeC = renameModeC.data();

        if (! renamer.valid())
        {
            return ctx->logError(LWTENSOR_STATUS_NOT_SUPPORTED, "Too many distinct modes were passed");
        }

        // can have at most 3 * max_modes modes at this point
        // since each mode has to occur at least twice
        // the maximum is twice max_modes
        // which transforms sets of modes into std::array<bool, 2 * MAX_MODES>
        // and unordered_maps into std::vector<V, 2 * MAX_MODES>

        std::unordered_map<mode_type, uint32_t> count;
        if( hasDuplicates(modeA, descA->getNumModes(), count) ){
           return ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "Mode A has duplicated values.");
        }
        if( hasDuplicates(modeB, descB->getNumModes(), count) ){
           return ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "Mode B has duplicated values.");
        }
        if( hasDuplicates(modeC, descC->getNumModes(), count) ){
           return ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "Mode C has duplicated values.");
        }

        for(const auto it : count )
        {
            if(it.second == 1)
            {
                return ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "Mode " + std::to_string(it.first) + "only oclwres once.");;
            }
        }

        // why can this not be equal?
        if(alignmentRequirementA <= 0
        || alignmentRequirementA % getDataTypeSize(descA->getDataType()) != 0)
        {
            return ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "Alignment requirement is not met for A.");
        }
        if(alignmentRequirementB <= 0
        || alignmentRequirementB % getDataTypeSize(descB->getDataType()) != 0)
        {
            return ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "Alignment requirement is not met for B.");
        }
        if(alignmentRequirementC <= 0
        || alignmentRequirementC % getDataTypeSize(descC->getDataType()) != 0)
        {
            return ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "Alignment requirement is not met for C.");
        }
        if( alignmentRequirementD != alignmentRequirementC)
        {
            return ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "C and D must have the same alignment.");
        }
        alignmentRequirementA_ = alignmentRequirementA;
        alignmentRequirementB_ = alignmentRequirementB;
        alignmentRequirementC_ = alignmentRequirementC;
        alignmentRequirementD_ = alignmentRequirementD;

        descA_  = *descA;
        descB_  = *descB;
        descC_  = *descC;
        std::copy_n(modeA, descA_.getNumModes(), modeA_.begin());
        std::copy_n(modeB, descB_.getNumModes(), modeB_.begin());
        std::copy_n(modeC, descC_.getNumModes(), modeC_.begin());

        minComputeType_ = minimumComputeType;

        tag_ = 0;

        this->setInitialized();

        return LWTENSOR_STATUS_SUCCESS;
    }

    /**
     * Sets the default parameters
     */
    void initElementwiseParameters(ElementwiseParameters &params)
    {
        ElementwiseParameters defaultValues;
        params = defaultValues;
    }
    /**
     * Sets the default parameters
     */
    void initElementwisePlan(ElementwisePlan &plan)
    {
        ElementwisePlan defaultValues;
        plan.swapAB_ = defaultValues.swapAB_;
        plan.swapAC_ = defaultValues.swapAC_;
        initElementwiseParameters(plan.params_);
    }


lwtensorStatus_t
ContractionDescriptorInternal::initContractionDescriptorInternal_(
                const lwdaDataType_t typeA,
                const lwdaDataType_t typeB,
                const lwdaDataType_t typeC,
                const lwtensorComputeType_t typeCompute,
                const ModeList &modeA,
                const ModeList &modeB,
                const ModeList &modeM, // free modes
                const ModeList &modeN, // free modes
                const ModeList &modeK, // contracted modes
                const ModeList &modeL, // looped/batched modes
                const ExtentMap &extent,
                const lwtensorOperator_t opA,
                const StrideMap &strideA,
                uint32_t alignmentRequirementA,
                const lwtensorOperator_t opB,
                const StrideMap &strideB,
                uint32_t alignmentRequirementB,
                const lwtensorOperator_t opC,
                const StrideMap &strideC,
                uint32_t alignmentRequirementC,
                bool stridedLoadsReqA,
                bool stridedLoadsReqB,
                bool contiguousModeIsBatchedA,
                bool contiguousModeIsBatchedB)
{
    /** If the mode lists exceed the limit, then return with an error. */
    if (modeM.size() > ContractionDescriptorInternal::LWTENSOR_MAX_MODES_M
     || modeN.size() > ContractionDescriptorInternal::LWTENSOR_MAX_MODES_N
     || modeK.size() > ContractionDescriptorInternal::LWTENSOR_MAX_MODES_K
     || modeL.size() > ContractionDescriptorInternal::LWTENSOR_MAX_MODES_L)
    {
        RETURN_STATUS(LWTENSOR_STATUS_NOT_SUPPORTED)
    }

    typeA_       = typeA;
    typeB_       = typeB;
    typeC_       = typeC;
    typeCompute_ = typeCompute;

    assert( modeM.size() >= 1 && modeN.size() >= 1 && modeK.size() >= 1 );

    nmodeM = modeM.size();
    nmodeN = modeN.size();
    nmodeK = modeK.size();
    nmodeL = modeL.size();

    opA_   = opA; 
    opB_   = opB;
    opC_   = opC;

    stridedLoadsReqA_ = stridedLoadsReqA;
    stridedLoadsReqB_ = stridedLoadsReqB;
    contiguousModeIsBatchedA_ = contiguousModeIsBatchedA;
    contiguousModeIsBatchedB_ = contiguousModeIsBatchedB;
    transA_           = (std::find(modeK.begin(), modeK.end(), modeA.front()) != modeK.end());
    transB_           = (std::find(modeK.begin(), modeK.end(), modeB.front()) == modeK.end());

    //partitions_ = 1; // this member variable is already set in ContractionPlan::init()
    if (partitions_ == 0 || partitions_ < -1)
    {
        RETURN_STATUS(LWTENSOR_STATUS_ILWALID_VALUE)
    }

    /** Determine minimal alignment requirements. */
    const auto vecModeA    = transA_ ? modeK.front() : modeM.front();
    alignmentRequirementA_ = std::min(alignmentRequirementA, getAlignmentRequirement(typeA, strideA, extent, vecModeA));
    const auto vecModeB    = transB_ ? modeN.front() : modeK.front();
    alignmentRequirementB_ = std::min(alignmentRequirementB, getAlignmentRequirement(typeB, strideB, extent, vecModeB));
    const auto vecModeC    = modeM.front();
    alignmentRequirementC_ = std::min(alignmentRequirementC, getAlignmentRequirement(typeC, strideC, extent, vecModeC));

    sortArray<extent_type>( modeM, extent, extentM);
    for(int i=nmodeM; i < LWTENSOR_MAX_MODES_M; ++i)
        extentM[i] = 1; // for ND blocking in TC kernel
    sortArray<extent_type>( modeN, extent, extentN);
    for(int i=nmodeN; i < LWTENSOR_MAX_MODES_N; ++i)
        extentN[i] = 1; // for ND blocking in TC kernel
    sortArray<extent_type>( modeL, extent, extentL);
    for(int i=nmodeK; i < LWTENSOR_MAX_MODES_K; ++i)
        extentK[i] = 1; // for parallel K if no second mode exists 

    sortArray<stride_type>( modeM, strideA, strideAm);
    for(int i=nmodeM; i < LWTENSOR_MAX_MODES_M; ++i)
        strideAm[i] = 0; // for ND blocking in TC kernel
    sortArray<stride_type>( modeM, strideC, strideCm);
    for(int i=nmodeM; i < LWTENSOR_MAX_MODES_M; ++i)
        strideCm[i] = 0; // for ND blocking in TC kernel
    sortArray<stride_type>( modeN, strideB, strideBn);
    for(int i=nmodeN; i < LWTENSOR_MAX_MODES_N; ++i)
        strideBn[i] = 0; // for ND blocking in TC kernel
    sortArray<stride_type>( modeN, strideC, strideCn);
    for(int i=nmodeN; i < LWTENSOR_MAX_MODES_N; ++i)
        strideCn[i] = 0; // for 2D blocking in TC kernel
    sortArray<stride_type>( modeL, strideA, strideAl);
    sortArray<stride_type>( modeL, strideB, strideBl);
    sortArray<stride_type>( modeL, strideC, strideCl);

    // INFO: https://jirasw.lwpu.com/browse/CUT-509
    //       When first mode of C has stride > 1, we cannot use fast epilogue.
    if (strideC.at(modeM.storage_[0]) > 1)
    {
         forceUseNaiveEpilogue_ = true;
    }

    #ifdef DEBUG
        printf("M: ");
        for (auto s : modeM) printf("(%c,%d,%d,%d) ", s, extent.at(s), strideA.at(s), strideC.at(s));
        printf("\nN: ");
        for (auto s : modeN) printf("(%c,%d,%d,%d) ", s, extent.at(s), strideB.at(s), strideC.at(s));
        printf("\nK: ");
        for (auto s : modeK) printf("(%c,%d,%d,%d) ", s, extent.at(s), strideA.at(s), strideB.at(s));
        printf("\nL: ");
        for (auto s : modeL) printf("(%c,%d,%d,%d,%d) ", s, extent.at(s), strideA.at(s), strideB.at(s),strideC.at(s));
    #endif
   // we might have to split a mode since the parallelization in k-dim is performed over modeK[1]
//   if( useLargeK )
//   {
//
//       // estimate numCTAs without large-k
//       const auto numCTAs = 1;//getNumCTAs(*this, 32, 32);
//
//       /// ensure that the old mode remains a multiple of vec
//       if( strideA.at(modeK.front()) != 1 && strideB.at(modeK.front()) != 1)
//       {
//           vec = 1;// no stride constraints
//       }
//
//       ModeList modeK_tmp(modeK);
//       ExtentMap extent_tmp(extent);
//       StrideMap strideA_tmp(strideA);
//       StrideMap strideB_tmp(strideB);
//
//       // large-k is allways parallelized across modeK[1] !!!
//       // Hence, we want to keep extent[modeK[1]] to a limited size (say ~ 2 * SM)
//       if( modeK.size() == 1 )
//       {
//           // split modeK: find x such that extentK | x and 2*SM<= x <=4*SM
//           auto modeK0 = modeK.front();
//           auto newExtent = findGoodSplitK(extent.at(modeK0), numCTAs, numSMs, vec);
//
////           printf("BEST: %d\n", newExtent);
//           splitAndInsert(modeK0, RESERVED_K_MODE, newExtent, 1, modeK_tmp, extent_tmp, strideA_tmp, strideB_tmp);
//       }
//       else
//       {
//           // find an existing suitable mode over which we can prallelize (i.e., large enough,
//           // but not too large and not stride-1 mode)
//           extent_type bestModeKExtent = 0;
//           mode_type bestModeK = -1;
//           for( auto k : modeK)
//           {
//               if( extent.at(k) > bestModeKExtent &&
//                   extent.at(k) <= numSMs && strideA.at(k) != 1 && strideB.at(k) != 1 )
//               {
//                   bestModeKExtent = extent.at(k);
//                   bestModeK = k;
//               }
//           }
//           if( bestModeK != -1 )
//           {
//               // move found mode to the second mode
//               modeK_tmp.remove(bestModeK);
//               modeK_tmp.insert(std::next(modeK_tmp.begin()), bestModeK);
//           }
//           else
//           {
//               // no such mode was found. In this case we have to split some of the
//               // existing modes
//               extent_type bestModeKExtent = 0;
//               mode_type bestModeK = -1;
//               for( auto k : modeK)
//               {
//                   if( (stridedLoadsA && strideB.at(k) == 1) || 
//                       (stridedLoadsB && strideA.at(k) == 1) ||
//                       ((!stridedLoadsA  && !stridedLoadsB) && (strideA.at(k) == 1 || strideB.at(k) == 1)) )
//                       continue;
//                   
//                   const auto myVec = (strideA.at(k) == 1 || strideB.at(k) == 1 ) ? vec : 1;
//                   const auto newExtent = findGoodSplitK(extent.at(k), numCTAs, numSMs, myVec);
//                   if( newExtent > bestModeKExtent )
//                   {
//                       bestModeKExtent = newExtent;
//                       bestModeK = k;
//                       if( numSMs*0.25 <= newExtent && newExtent <= numSMs)
//                       {
//                           break; // fast return
//                       }
//                   }
//               }
//               assert( bestModeKExtent != 0 );
////               printf("Split: %d %c %d\n", bestModeK ,bestModeK , bestModeKExtent);
//               splitAndInsert(bestModeK, RESERVED_K_MODE, bestModeKExtent, 1, modeK_tmp, extent_tmp, strideA_tmp, strideB_tmp);
//           }
//       }
////       for(auto k : modeK_tmp)
////           printf("(%d,%c) %d %d %d; ", k,k, extent_tmp.at(k), strideA_tmp.at(k), strideB_tmp.at(k));
////       printf("\n");
//
//       this->nmodeK = modeK_tmp.size();
//
//       sortArray<extent_type>( modeK_tmp, extent_tmp, extentK);
//       sortArray<stride_type>( modeK_tmp, strideA_tmp, strideAk);
//       sortArray<stride_type>( modeK_tmp, strideB_tmp, strideBk);
//   } else {
       sortArray<extent_type>( modeK, extent, extentK);
       sortArray<stride_type>( modeK, strideA, strideAk);
       sortArray<stride_type>( modeK, strideB, strideBk);
//   }

    this->setInitialized();
    return LWTENSOR_STATUS_SUCCESS;
}

lwtensorStatus_t
ContractionDescriptorInternal::initContractionDescriptorInternal(
                const lwdaDataType_t typeA,
                const lwdaDataType_t typeB,
                const lwdaDataType_t typeC,
                const lwtensorComputeType_t typeCompute,
                const ModeList &modeA,
                const ModeList &modeB,
                const ModeList &modeM, // free modes
                const ModeList &modeN, // free modes
                const ModeList &modeK, // contracted modes
                const ModeList &modeL, // looped/batched modes
                const ExtentMap &extent,
                const lwtensorOperator_t opA,
                const StrideMap &strideA,
                uint32_t alignmentRequirementA,
                const lwtensorOperator_t opB,
                const StrideMap &strideB,
                uint32_t alignmentRequirementB,
                const lwtensorOperator_t opC,
                const StrideMap &strideC,
                uint32_t alignmentRequirementC,
                bool stridedLoadsReqA,
                bool stridedLoadsReqB,
                bool contiguousModeIsBatchedA,
                bool contiguousModeIsBatchedB,
                bool &swapAB)
{
    if( stridedLoadsReqA && stridedLoadsReqB )
    {
        RETURN_STATUS(LWTENSOR_STATUS_NOT_SUPPORTED)
    }
    if(stridedLoadsReqA && !stridedLoadsReqB)
    {
        swapAB = !swapAB;
        forceUseNaiveEpilogue_ = true; // required for correctness (since the normal epilogue is row-major)
        return initContractionDescriptorInternal_(
                 typeB, typeA, typeC, typeCompute,
                 modeB, modeA,
                 modeN, modeM,
                 modeK,
                 modeL,
                 extent,
                 opB, strideB, alignmentRequirementB,
                 opA, strideA, alignmentRequirementA,
                 opC, strideC, alignmentRequirementC,
                 stridedLoadsReqB, stridedLoadsReqA, contiguousModeIsBatchedA, contiguousModeIsBatchedB);
    }else{
        forceUseNaiveEpilogue_ = false;
        return initContractionDescriptorInternal_(
                 typeA, typeB, typeC, typeCompute,
                 modeA, modeB,
                 modeM, modeN,
                 modeK,
                 modeL,
                 extent,
                 opA, strideA, alignmentRequirementA,
                 opB, strideB, alignmentRequirementB,
                 opC, strideC, alignmentRequirementC,
                 stridedLoadsReqA, stridedLoadsReqB, contiguousModeIsBatchedA, contiguousModeIsBatchedB);
    }
}

extent_type ContractionDescriptorInternal::getTotalExtentM() const
{
    extent_type extent = 1;
    for(int i=0; i < this->nmodeM; ++i){
        extent *= this->extentM[i];
    }
    return extent;
}
extent_type ContractionDescriptorInternal::getTotalExtentN() const
{
    extent_type extent = 1;
    for(int i=0; i < this->nmodeN; ++i){
        extent *= this->extentN[i];
    }
    return extent;
}
extent_type ContractionDescriptorInternal::getTotalExtentK() const
{
    extent_type extent = 1;
    for(int i=0; i < this->nmodeK; ++i){
        extent *= this->extentK[i];
    }
    return extent;
}
extent_type ContractionDescriptorInternal::getTotalExtentL() const
{
    extent_type extent = 1;
    for(int i=0; i < this->nmodeL; ++i){
        extent *= this->extentL[i];
    }
    return extent;
}

/**
  * This function determines the minimal alignment requirement for the given tensor.
  *
  * \param[in] ptr Pointer to the start of the tensor.
  * \param[in] stride Stride of the mode that ought to be vectorized.
  * \param[in] extent Extent of the mode that ought to be vectorized.
  * \return Alignment requirement in bytes.
  */
uint32_t ContractionDescriptorInternal::getAlignmentRequirement(const lwdaDataType_t dataType,
                                                                const StrideMap& stride,
                                                                const ExtentMap& extent,
                                                                const mode_type vecMode) const
{
    const uint32_t typeSize = getDataTypeSize(dataType);
    uint32_t vectorWidth    = getVectorization(dataType); // in elements
    StrideMap tempStride(stride);

    if(tempStride.at(vecMode) == 1)
    {
        tempStride.erase(vecMode);

        while(vectorWidth > 1)
        {
            if( (extent.at(vecMode) % vectorWidth) == 0 )       // is s multiple of the vector-width
            {
                bool allStridesMultipleVec = true;
                for(const auto& s : tempStride) 
                {
                    if(s.second % vectorWidth != 0)
                    {
                        allStridesMultipleVec = false;
                        break;
                    }
                }

                if( allStridesMultipleVec ) {
                    return vectorWidth * typeSize;
                }
            }

            vectorWidth /= 2;
        }
    }

    return typeSize;
}

TTGTCandidate::TTGTCandidate(const ContractionDynamicParams &params,
                     const bool useHybrid, int32_t candidateIdx) : sizeA_(0), sizeB_(0), sizeC_(0), sizeD_(0)
{
    const lwtensorOperator_t opA = params.opA_;
    const lwtensorOperator_t opB = params.opB_;
    const lwtensorOperator_t opC = params.opC_;
    const lwdaDataType_t typeA = params.typeA_;
    const lwdaDataType_t typeB = params.typeB_;
    const lwdaDataType_t typeC = params.typeC_;
    const ModeList& modeM = params.modeM_;
    const ModeList& modeN = params.modeN_;
    const ModeList& modeK = params.modeK_;
    const ModeList& modeL = params.modeL_;
    const ExtentMap& extent = params.extent_;
    const StrideMap& strideA = params.strideA_;
    const ModeList& modeA = params.modeA_; // modes before transpose
    const StrideMap& strideB = params.strideB_;
    const ModeList& modeB = params.modeB_; // modes before transpose
    const StrideMap& strideC = params.strideC_;
    const ModeList& modeC = params.modeC_; // modes before transpose

    /** Start from the naive case, where all tensors are transposed. */
    ModeList modeM_after;
    ModeList modeN_after;
    ModeList modeK_after;
    ModeList modeL_after;
    transposeA_ = true;
    transposeB_ = true;
    transposeC_ = true;

    const uint64_t totalExtentM = getTotalModeExtent(modeM, extent);
    const uint64_t totalExtentN = getTotalModeExtent(modeN, extent);
    const uint64_t totalExtentK = getTotalModeExtent(modeK, extent);
    const uint64_t totalExtentL = getTotalModeExtent(modeL, extent);

    /** Get contiguous indicies of mode C that appear in modes {M,N,L}. */
    auto m_c_contiguous = getContiguousIndices(modeM, modeC);
    auto n_c_contiguous = getContiguousIndices(modeN, modeC);
    auto l_c_contiguous = getContiguousIndices(modeL, modeC);
    /** Get contiguous indicies of mode A that appear in modes {M,K,L}. */
    auto m_a_contiguous = getContiguousIndices(modeM, modeA);
    auto k_a_contiguous = getContiguousIndices(modeK, modeA);
    auto l_a_contiguous = getContiguousIndices(modeL, modeA);
    /** Get contiguous indicies of mode B that appear in modes {N,K,L}. */
    auto n_b_contiguous = getContiguousIndices(modeN, modeB);
    auto k_b_contiguous = getContiguousIndices(modeK, modeB);
    auto l_b_contiguous = getContiguousIndices(modeL, modeB);

    // printMode( std::string( "modeM" ), modeM );
    // printMode( std::string( "modeN" ), modeN );
    // printMode( std::string( "modeK" ), modeK );
    // printMode( std::string( "modeL" ), modeL );
    // printMode( std::string( "modeC" ), modeC );
    // printMode( std::string( "modeA" ), modeA );
    // printMode( std::string( "modeB" ), modeB );
    // printMode( std::string( "M_C" ), m_c_contiguous );
    // printMode( std::string( "N_C" ), n_c_contiguous );
    // printMode( std::string( "M_A" ), m_a_contiguous );
    // printMode( std::string( "K_A" ), k_a_contiguous );
    // printMode( std::string( "N_B" ), n_b_contiguous );
    // printMode( std::string( "K_B" ), k_b_contiguous );
    // printf( "\n" );

    /** Check if C is contiguous on modes {M,N}. */
    bool is_m_c_contiguous = modeM.size() == m_c_contiguous.size();
    bool is_n_c_contiguous = modeN.size() == n_c_contiguous.size();
    bool is_l_c_contiguous = modeL.size() == l_c_contiguous.size();
    /** Check if A is contiguous on modes {M,K}. */
    bool is_m_a_contiguous = modeM.size() == m_a_contiguous.size();
    bool is_k_a_contiguous = modeK.size() == k_a_contiguous.size();
    bool is_l_a_contiguous = modeL.size() == l_a_contiguous.size();
    /** Check if B is contiguous on modes {N,K}. */
    bool is_n_b_contiguous = modeN.size() == n_b_contiguous.size();
    bool is_k_b_contiguous = modeK.size() == k_b_contiguous.size();
    bool is_l_b_contiguous = modeL.size() == l_b_contiguous.size();

    /** Check if {A,B,C} agree on the order of modes {M,N,K}. */
    const bool c_a_agree_on_m = isTheSameList(m_c_contiguous, m_a_contiguous) && is_m_a_contiguous;
    const bool c_b_agree_on_n = isTheSameList(n_c_contiguous, n_b_contiguous) && is_n_b_contiguous;
    const bool a_b_agree_on_k = isTheSameList(k_a_contiguous, k_b_contiguous) && is_k_b_contiguous;
    const bool a_b_agree_on_l = isTheSameList(l_a_contiguous, l_b_contiguous) && is_l_a_contiguous;
    const bool a_c_agree_on_l = isTheSameList(l_a_contiguous, l_c_contiguous) && is_l_a_contiguous;
    const bool c_b_agree_on_l = isTheSameList(l_b_contiguous, l_c_contiguous) && is_l_b_contiguous;

    // std::cout << "C A agree:" << c_a_agree_on_m << std::endl;
    // std::cout << "C B agree:" << c_b_agree_on_n << std::endl;
    // std::cout << "A B agree:" << a_b_agree_on_k << std::endl;

    /** Check if stride C matches extent C in each mode. */
    bool is_c_stride_and_extent_matching = strideMatchesExtent(modeC, strideC, extent);
    bool is_a_stride_and_extent_matching = strideMatchesExtent(modeA, strideA, extent);
    bool is_b_stride_and_extent_matching = strideMatchesExtent(modeB, strideB, extent);

    // std::cout << "C stride and extent matches:" << is_c_stride_and_extent_matching << std::endl;
    // std::cout << "A stride and extent matches:" << is_a_stride_and_extent_matching << std::endl;
    // std::cout << "B stride and extent matches:" << is_b_stride_and_extent_matching << std::endl;
    // printStrideExtent( std::string( "C" ), modeC, strideC, extent );
    // printStrideExtent( std::string( "A" ), modeA, strideA, extent );
    // printStrideExtent( std::string( "B" ), modeB, strideB, extent );

    /** Check if modeL is the leading mode? */
    bool is_c_l_leading = std::find(modeL.begin(), modeL.end(), modeC.front()) != modeL.end();
    bool is_a_l_leading = std::find(modeL.begin(), modeL.end(), modeA.front()) != modeL.end();
    bool is_b_l_leading = std::find(modeL.begin(), modeL.end(), modeB.front()) != modeL.end();

    /** Get the tensor size. */
    float totalSizeC = (float) totalExtentM * (float) totalExtentN;
    float totalSizeA = (float) totalExtentM * (float) totalExtentK;
    float totalSizeB = (float) totalExtentK * (float) totalExtentN;

    /** Decide whether each tensor must be transposed (s.t. it can be mapped to a batched-GEMM). */
    bool transpose_c
        = !is_l_c_contiguous | !is_m_c_contiguous | !is_n_c_contiguous | !is_c_stride_and_extent_matching | is_c_l_leading;
    bool transpose_a
        = !is_l_a_contiguous | !is_m_a_contiguous | !is_k_a_contiguous | !is_a_stride_and_extent_matching | is_a_l_leading;
    bool transpose_b
        = !is_l_b_contiguous | !is_n_b_contiguous | !is_k_b_contiguous | !is_b_stride_and_extent_matching | is_b_l_leading;

    if (useHybrid)
    {
        // Transpose A or B (which ever seems more reasonable from a perf perspective)
        transpose_c = false;

        const bool req3DTileAC
            = (std::find(modeK.begin(), modeK.end(), modeA.front()) == modeK.end()) && (modeA.front() != modeC.front());

        if (req3DTileAC)
        {
            transpose_a = true;
            transpose_b = false;
        }
        else
        {
            const bool isTransposeUsefulA = transpose_a || !a_b_agree_on_k || !c_a_agree_on_m;
            const bool isTransposeUsefulB = transpose_b || !a_b_agree_on_k || !c_b_agree_on_n;
            // transpose either A or B
            if (isTransposeUsefulA) // we will permute to make the access pattern more regular
            {
                if (isTransposeUsefulB)
                {
                    if (totalSizeA <= totalSizeB) // transpose the smaller of the two
                    {
                        transpose_a = true;
                        transpose_b = false;
                    }
                    else if (isTransposeUsefulB)
                    {
                        transpose_a = false;
                        transpose_b = true;
                    }
                }
                else
                {
                    transpose_a = true;
                    transpose_b = false;
                }
            }
            else if (isTransposeUsefulB)
            {
                transpose_a = false;
                transpose_b = true;
            }
            else if (a_b_agree_on_l) // A and B agree on l
            {
                // check if it would make sense to transpose C
                if (!a_c_agree_on_l)
                {
                    transpose_a = false;
                    transpose_b = false;
                    transpose_c = true;
                }
            }
            else if (totalExtentN < totalExtentM) // transpose the smaller of the two tensors
            {
                transpose_a = false;
                transpose_b = true;
            }
            else
            {
                transpose_a = true;
                transpose_b = false;
            }
        }

        // ensure that only one tensor is transposed
        assert(!(transpose_a && transpose_b));
        assert(!(transpose_a && transpose_c));
        assert(!(transpose_b && transpose_c));

        if (transpose_a)
        {
            // since we transpose A anyway we sort m- and k-modes w.r.t. C and B respectively
            intersect(modeC, modeA, modeM_after);
            intersect(modeB, modeA, modeK_after);
            for (auto l : modeL) {
                modeM_after.remove(l);
                modeK_after.remove(l);
            }
            modeN_after = modeN;
            modeL_after = modeL; // TODO sort w.r.t B
        }
        else if (transpose_b)
        {
            // since we transpose B anyway we sort n- and k-modes w.r.t. C and A respectively
            intersect(modeA, modeB, modeK_after);
            intersect(modeC, modeB, modeN_after);
            for (auto l : modeL) {
                modeN_after.remove(l);
                modeK_after.remove(l);
            }
            modeM_after = modeM;
            modeL_after = modeL;
        }
        else if (transpose_c)
        {
            // since we transpose C anyway we sort m- and n-modes w.r.t. A and B respectively
            intersect(modeA, modeC, modeM_after);
            intersect(modeB, modeC, modeN_after);
            for (auto l : modeL) {
                modeM_after.remove(l);
                modeN_after.remove(l);
            }
            modeL_after = modeL;
            modeK_after = modeK;
        }
        else
        {
            modeM_after = modeM;
            modeN_after = modeN;
            modeK_after = modeK;
            modeL_after = modeL;
        }
    }
    else
    {
        // I've disabled this optimization for now since it might result into problems
        // w.r.t. beta (since we might have to colwert the type (i.e., dereference the
        // pointer)
//        if( getDataTypeSize(typeA) > getDataTypeSize(typeCompute) || // this is just a perf optimization
//                ((typeA != typeCompute) && !isDataTypeSupported) )
//        {
//            transpose_a = true;
//            this->typeA = typeCompute;
//        }else{
//            this->typeA = typeA;
//        }
//        if( getDataTypeSize(typeB) > getDataTypeSize(typeCompute) || // this is just a perf optimization
//                ((typeB != typeCompute) && !isDataTypeSupported) )
//        {
//            transpose_b = true;
//            this->typeB = typeCompute;
//        }else{
//            this->typeB = typeB;
//        }
//        if( getDataTypeSize(typeC) > getDataTypeSize(typeCompute) || // this is just a perf optimization
//                ((typeC != typeCompute) && !isDataTypeSupported) )
//        {
//            transpose_c = true;
//            this->typeC = typeCompute;
//        }else{
//            this->typeC = typeC;
//        }
        transpose_a |= (opA != LWTENSOR_OP_IDENTITY);
        transpose_b |= (opB != LWTENSOR_OP_IDENTITY);
        transpose_c |= (opC != LWTENSOR_OP_IDENTITY);

        /** If C and A disagree on M, then one of them must be transposed. */
        bool transpose_c_or_a = (!c_a_agree_on_m || !a_c_agree_on_l) && !transpose_c && !transpose_a;
        bool transpose_c_or_b = (!c_b_agree_on_n || !c_b_agree_on_l) && !transpose_c && !transpose_b;
        bool transpose_a_or_b = (!a_b_agree_on_k || !a_b_agree_on_l) && !transpose_a && !transpose_b;

        /** Update the decision according to the cost analysis. */
        if (transpose_c_or_a && transpose_c_or_b && transpose_a_or_b)
        {
            transpose_c = true;
            transpose_a = true;
            transpose_b = true;
        }
        else if (transpose_c_or_a && transpose_c_or_b && !transpose_a_or_b)
        {
            if (totalSizeC <= totalSizeA + totalSizeB) {
                transpose_c = true;
            }
            else
            {
                transpose_a = true;
                transpose_b = true;
            }
        }
        else if (transpose_c_or_a && !transpose_c_or_b && transpose_a_or_b)
        {
            if (totalSizeA <= totalSizeC + totalSizeB) {
                transpose_a = true;
            }
            else
            {
                transpose_c = true;
                transpose_b = true;
            }
        }
        else if (!transpose_c_or_a && transpose_c_or_b && transpose_a_or_b)
        {
            if (totalSizeB <= totalSizeC + totalSizeA) {
                transpose_b = true;
            }
            else
            {
                transpose_c = true;
                transpose_a = true;
            }
        }
        else if (transpose_c_or_a && !transpose_c_or_b && !transpose_a_or_b)
        {
            if (totalSizeC <= totalSizeA) {
                transpose_c = true;
            }
            else
            {
                transpose_a = true;
            }
        }
        else if (!transpose_c_or_a && transpose_c_or_b && !transpose_a_or_b)
        {
            if (totalSizeC <= totalSizeB) {
                transpose_c = true;
            }
            else
            {
                transpose_b = true;
            }
        }
        else if (!transpose_c_or_a && !transpose_c_or_b && transpose_a_or_b)
        {
            if (totalSizeA <= totalSizeB) {
                transpose_a = true;
            }
            else
            {
                transpose_b = true;
            }
        }

        /** Return the naive case where all operands are transposed. */
        if (transpose_c && transpose_a && transpose_b)
        {
            modeM_after = modeM;
            modeN_after = modeN;
            modeK_after = modeK;
            modeL_after = modeL;
        }
        else if (!transpose_c && transpose_a && transpose_b)
        {
            /** Assign modes {M,N} according to C. */
            modeM_after = m_c_contiguous;
            modeN_after = n_c_contiguous;
            modeK_after = modeK;
            modeL_after = l_c_contiguous;
        }
        else if (transpose_c && transpose_a && !transpose_b)
        {
            /** Assign modes {N,K} according to B. */
            modeM_after = modeM;
            modeN_after = n_b_contiguous;
            modeK_after = k_b_contiguous;
            modeL_after = l_b_contiguous;
        }
        else if (transpose_c && !transpose_a && transpose_b)
        {
            /** Assign modes {M,K} according to A. */
            modeM_after = m_a_contiguous;
            modeN_after = modeN;
            modeK_after = k_a_contiguous;
            modeL_after = l_a_contiguous;
        }
        else if (transpose_c && !transpose_a && !transpose_b)
        {
            /** Assign modes {M,N,K} according to {A,B}. */
            modeM_after = m_a_contiguous;
            modeN_after = n_b_contiguous;
            modeK_after = k_b_contiguous;
            modeL_after = l_a_contiguous;
        }
        else if (!transpose_c && transpose_a && !transpose_b)
        {
            /** Assign modes {M,N,K} according to {C,B}. */
            modeM_after = m_c_contiguous;
            modeN_after = n_c_contiguous;
            modeK_after = k_b_contiguous;
            modeL_after = l_c_contiguous;
        }
        else if (!transpose_c && !transpose_a && transpose_b)
        {
            /** Assign modes {M,N,K} according to {C,A}. */
            modeM_after = m_c_contiguous;
            modeN_after = n_c_contiguous;
            modeK_after = k_a_contiguous;
            modeL_after = l_a_contiguous;
        }
        else if (!transpose_c && !transpose_a && !transpose_b)
        {
            /** Assign modes {M,N,K} according to {C,A,B}. */
            modeM_after = m_c_contiguous;
            modeN_after = n_c_contiguous;
            modeK_after = k_b_contiguous;
            modeL_after = l_a_contiguous;
        }
        else
        {
            assert(0);
        }
    }
    assert(modeM_after.size() == modeM.size());
    assert(modeN_after.size() == modeN.size());
    assert(modeK_after.size() == modeK.size());
    assert(modeL_after.size() == modeL.size());

    /** Assign transposeC, transposeA, and transposeB. */
    transposeA_ = transpose_a;
    transposeB_ = transpose_b;
    transposeC_ = transpose_c;

    constexpr uint64_t alignmentRequirement = ContractionPlan::alignmentRequirement_; // in bytes
    if (transposeA_)
    {
        sizeA_ = roundUp(totalExtentL * totalExtentM * totalExtentK * getDataTypeSize(typeA), alignmentRequirement);
    }
    if (transposeB_)
    {
        sizeB_ = roundUp(totalExtentL * totalExtentN * totalExtentK * getDataTypeSize(typeB), alignmentRequirement);
    }
    sizeD_ = roundUp(totalExtentL * totalExtentM * totalExtentN * getDataTypeSize(typeC), alignmentRequirement);
    if (transposeC_)
    {
        sizeC_ = sizeD_;
    }

    /**
     * In this version, we always permute A to A_after and B to B_after
     * in the order of {modeM,modeK} and {modeN,modeK}. As a result,
     * We will be calling a GEMM_NT, which has he highest performance.
     * The resulting C will have {modeM,modeN}.
     */
    if (transposeA_)
    {
        bool modeOrderMK = true; // prefer non-transpoase A
        if (candidateIdx == 1 || candidateIdx == 3)
        {
            modeOrderMK = !modeOrderMK; // explore other options
        }
        if (modeOrderMK)
        {
            for (auto m : modeM_after) modeA_.push_back(m);
            for (auto m : modeK_after) modeA_.push_back(m);
            for (auto m : modeL_after) modeA_.push_back(m);
        }
        else
        {
            for (auto m : modeK_after) modeA_.push_back(m);
            for (auto m : modeM_after) modeA_.push_back(m);
            for (auto m : modeL_after) modeA_.push_back(m);
        }
    }else{
        for (auto m : modeA) modeA_.push_back(m);
    }

    if (transposeB_)
    {
        bool modeOrderKN = false; // prefer transpose B
        if (candidateIdx == 2 || candidateIdx == 3)
        {
            modeOrderKN = !modeOrderKN; // explore other options
        }
        if (modeOrderKN)
        {
            for (auto m : modeK_after) modeB_.push_back(m);
            for (auto m : modeN_after) modeB_.push_back(m);
            for (auto m : modeL_after) modeB_.push_back(m);
        }
        else
        {
            for (auto m : modeN_after) modeB_.push_back(m);
            for (auto m : modeK_after) modeB_.push_back(m);
            for (auto m : modeL_after) modeB_.push_back(m);
        }
    }else{
        for (auto m : modeB) modeB_.push_back(m);
    }

    if (transposeC_)
    {
        for (auto m : modeM_after) modeC_.push_back(m);
        for (auto m : modeN_after) modeC_.push_back(m);
        for (auto m : modeL_after) modeC_.push_back(m);
    }else{
        for (auto m : modeC) modeC_.push_back(m);
    }

    assert(modeA_.size() == modeA.size());
    assert(modeB_.size() == modeB.size());
    assert(modeC_.size() == modeC.size());
}

size_t TTGTCandidate::getRequiredWorkspace() const
{
    return this->sizeA_ + this->sizeB_ + this->sizeC_;
}

size_t TTGTCandidate::getRecommendedWorkspace() const
{
    return this->sizeA_ + this->sizeB_ + this->sizeD_;
}

/**
  * Print the stride and extent of each mode in the list.
  */
void printStrideExtent(const std::string& name, const ModeList& mode,
                       const StrideMap& stride,
                       const ExtentMap& extent)
{
    std::cout << name << ":";
    for (auto m : mode) std::cout << "(" << stride.find(m)->second << "," << extent.find(m)->second << ") ";
    std::cout << "\n";
}

/**
  * Print out all candidate information.
  */
void TTGTCandidate::print() const
{
    std::cout << "TTGT candidate: C transposed? " << transposeC_ << "\n";
    std::cout << "                A transposed? " << transposeA_ << "\n";
    std::cout << "                B transposed? " << transposeB_ << "\n";
    std::cout << "                XGEMM TRANSA? " << (std::find(modeB_.begin(), modeB_.end(), modeA_.front()) != modeB_.end()) << "\n";
    std::cout << "                XGEMM TRANSB? " << (std::find(modeA_.begin(), modeA_.end(), modeB_.front()) == modeA_.end()) << "\n";
    std::cout << "                       modeA? ";
    printMode(std::string("modeA"), modeA_);
    std::cout << "                       modeB? ";
    printMode(std::string("modeB"), modeB_);
    std::cout << "                       modeC? ";
    printMode(std::string("modeC"), modeC_);
}

/**
 * \pre All modes of A must also appear in either B or C (or both).
 */
lwtensorStatus_t ReductionParams::init(
        const Context* ctx,
        const lwdaDataType_t typeA,
        const lwdaDataType_t typeB,
        const lwdaDataType_t typeC,
        const lwtensorComputeType_t typeCompute, 
        const lwtensorOperator_t opA,
        const lwtensorOperator_t opB,
        const lwtensorOperator_t opC,
        const lwtensorOperator_t opAB,
        const lwtensorOperator_t opReduce,
        const ModeList &modeA,
        const ModeList &modeB,
        const ModeList &modeC,
        const StrideMap &strideA,
        const StrideMap &strideB,
        const StrideMap &strideC,
        const ExtentMap &extent)
{
    typeA_ = typeA;
    typeB_ = typeB;
    typeC_ = typeC;
    typeCompute_ = typeCompute;
    opA_ = opA;
    opB_ = opB;
    opC_ = opC;
    opAB_ = opAB;
    opReduce_ = opReduce;

    for(int i=0; i < LWTENSOR_MAX_MODES; ++i){
        extentM_[i] = 1;
        extentK_[i] = 1;
        strideAm_[i] = 0;
        strideCm_[i] = 0;
        strideAk_[i] = 0;
        strideBk_[i] = 0;
    }
    // -1 since we might have added one artificial m or n mode
    if( modeA.size() >= LWTENSOR_MAX_MODES - 1 )
    {
        RETURN_STATUS(ctx->logError(LWTENSOR_STATUS_NOT_SUPPORTED, "Too many (non-fusable) modes."))
    }

    ModeList modeK;
    intersect( modeA, modeB, modeK );
    ModeList modeM;
    intersect( modeA, modeC, modeM );
    ModeList modeN;
    intersect( modeB, modeC, modeN );
    ModeList modeL;
    intersect( modeK, modeC, modeL );
    for(auto l : modeL)
    {
        modeK.remove(l);
        modeN.remove(l);
        modeM.remove(l);
    }

    totalExtentM_ = getTotalModeExtent(modeM, extent);
    totalExtentK_ = getTotalModeExtent(modeK, extent);

    const auto totalN = getTotalModeExtent(modeN, extent);
    if( totalN > 1 )
    {
        RETURN_STATUS(LWTENSOR_STATUS_NOT_SUPPORTED);
    }
    const auto totalL = getTotalModeExtent(modeL, extent);
    if( totalL > 1 )
    {
        RETURN_STATUS(ctx->logError(LWTENSOR_STATUS_NOT_SUPPORTED, "Batched reductions are not yet supported."))
    }
    if( modeA.size() == 0 || modeK.size() == 0 )
    {
        RETURN_STATUS(ctx->logError(LWTENSOR_STATUS_NOT_SUPPORTED, "Reduction not possible: Number of modes is invalid (are you missing a contracted mode? You could try to add an 'artificial' mode with an extent of one)"))
    }

    const bool transA = (strideA.at(modeK.front()) == 1);

    const auto extentK0 = modeK.size() > 0 ? extent.at(modeK.front()) : 1;
    const auto extentM0 = modeM.size() > 0 ? extent.at(modeM.front()) : 1;

    const auto typeSizeA = getDataTypeSize(typeA);
    const int preferredVectorWidth
        = (typeSizeA >= cPreferredVectorWidthBytes) ? 1 : cPreferredVectorWidthBytes / typeSizeA;
    const extent_type vectorWidthM = transA ? 1 : ( extentM0 % preferredVectorWidth == 0 ? preferredVectorWidth : 1);
    const extent_type vectorWidthK = !transA ? 1 : ( extentK0 % preferredVectorWidth == 0 ? preferredVectorWidth : 1);

    /********************
     * Split k-dim to increase parallelism.
     ********************/
    StrideMap strideA_(strideA);
    StrideMap strideB_(strideB);
    StrideMap strideC_(strideC);
    ExtentMap extent_(extent);
    extent_type numElementsPerThreadblockM = (transA) ? 1 : NUM_THREADS_M * vectorWidthM; 
    const auto numBlocksM = (totalExtentM_ + numElementsPerThreadblockM - 1) / numElementsPerThreadblockM ;
    if( modeK.size() == 1U && totalExtentK_ > 1 )
    {
        auto mode0 = modeK.front();
        auto newExtent = findGoodSplitK(extentK0, numBlocksM, 300, vectorWidthK);
        splitAndInsert(mode0, RESERVED_K_MODE, newExtent, 1, modeK, extent_, strideA_, strideB_);
    }

    constexpr int minNumBlocks = 160; // we want to have at least this many threadblocks active
    constexpr int targetNumBlocks = 512; // this or more
    // We want to have at least this many elements in the unblocked mode (i.e., consumed by threads within a threadblock) to reduce the index callwlation overhead
    constexpr extent_type minBlockedExtent = 256;
    constexpr extent_type targetExtentBlocked = 2*1024; // this or more

    // the k-dim --per threadblock-- should have at least minBlockedExtent  many elements (since it
    // will reduce the indexing overhead; moreover, if this value is too small we run the
    // risk of having idle threads along this dimension. For instance, if we have two
    // k-modes k1, k2 with extent(k1) = 13, and extent(k2) = 100000 as well as a moderate
    // totalExtentM_, say 1 in this case (i.e., no parallelism in m-dimension), then we
    // still want both 1) increase the blockedExtent and 2) keep the unblockedExtent large
    // enough to extract parallelism out of it. In this case we have to split the k2 mode;
    // for instance into k21 = 100, k22 = 1000 such that k21 * k1 >256 and k22 >=
    // minNumBlocks;
    // In summary, we want:
    //    1) minBlockedExtent <= blockedExtent (or larger) and
    //    2) minNumBlocks <= numBlocksM * unblockedExtent (or larger)
    // We give up, if neither can be fulfilled (this is the case if both m and k is
    // small, i.e., the entire reduction is likely insignificant in time) 
    // To be precise, we want to split the k-modes such that:
    //    1) minBlockedExtent <= sum(extent[modek[i]]), 0 <= i < cNumBlockedModes
    //    1) minNumBlocks <= sum(extent[modek[i]]), cNumBlockedModes <= i < nmodeK
    // We accomplish this by spliting modeK[cNumBlockedModes-1]
    
    extent_type blockedExtent = 1;
    extent_type unblockedExtent = 1;
    uint32_t i = 0;
    for( auto mode : modeK )
    {
        const auto lwrrentExtent = extent_.at(mode);
        if( i < cNumBlockedModes ){
            blockedExtent *= lwrrentExtent; // these are parallelized across threads
        }else{
            unblockedExtent *= lwrrentExtent; // these are parallelized across threadblocks
        }
        i++;
    }
    const auto numBlocks = unblockedExtent * numBlocksM; // available parallelism
    if( numBlocks < minNumBlocks ) // parallelism is insufficient
    {
        if( blockedExtent < minBlockedExtent )
        {
            // We give up, since everything is small (i.e., the reduction will not take a
            // lot of time); let's hope we don't have too many of those. We could do
            // better here.
        }else if( modeK.size() >= 2) { // blocked extent is large enough

            auto secondModeK = *std::next(modeK.begin(), cNumBlockedModes - 1);
            const auto extentSecondModeK = extent_.at(secondModeK);
            extent_type bestNewExtent = 1;
            float bestScore = 0.0f;

            // The new mode will be counted towards parallelism (NOT towards blocking)

            /*
             * Start looking for a good split that favors parallelism
             */
            extent_type newExtent = targetNumBlocks / numBlocks + 1; // start with a large amount (i.e., high parallelismScore)
            assert(newExtent >= 1);
            // find good split
            while( newExtent >= 1)
            {
                newExtent--;
                if( extentSecondModeK % newExtent == 0 ) // multiplicative split is possible
                {
                    const auto newBlockedExtent = blockedExtent / newExtent;
                    const auto newNumBlocks = numBlocks * newExtent;

                    if( newNumBlocks < minNumBlocks ) 
                        break; // we started from a large number (i.e., we won't find a solution any longer)
                    if(newBlockedExtent < minBlockedExtent )
                        continue;

                    const float parallelismScore = ((float)newNumBlocks) / targetNumBlocks; // larger is better
                    const float blockingScore = std::min(1.0f, ((float)newBlockedExtent) / targetExtentBlocked); // cap score
                    const float score = parallelismScore * blockingScore;
                    if( score > bestScore )
                    {
                        bestScore = score;
                        bestNewExtent = newExtent;
                    }
                }
            }
            /*
             * Search for a good split that favors blocking
             */
            // startlooking for extents such that the overall blockedExtent is targetExtentBlocked. Thus, we want that 
            // blockedExtent / newExtent <= targetExtentBlocked; => newExtent = blockedExtent / (targetExtentBlocked)
            auto blockingWithoutSecond = blockedExtent / extentSecondModeK;
            extent_type x = std::min(extentSecondModeK, static_cast<extent_type>(1.5 * targetExtentBlocked / blockingWithoutSecond + 1)); // represents the mode that will be counted towards blocking
            while( x > 1 )
            {
                x--;
                if( extentSecondModeK % x == 0 ) // multiplicative split is possible
                {
                    newExtent = extentSecondModeK / x;
                    const auto newBlockedExtent = blockingWithoutSecond * x;
                    const auto newNumBlocks = numBlocks * newExtent;

                    if( newNumBlocks < minNumBlocks ) 
                        continue;
                    if(newBlockedExtent < minBlockedExtent )
                        break; // we started from a large number (i.e., we won't find a solution any longer)

                    const float parallelismScore = std::min(1.0f, ((float)newNumBlocks) / targetNumBlocks); // larger is better
                    const float blockingScore = std::min(1.0f, ((float)newBlockedExtent) / targetExtentBlocked); // cap score
                    const float score = parallelismScore * blockingScore;
                    if( score > bestScore )
                    {
                        bestScore = score;
                        bestNewExtent = newExtent;
                    }
                }
            }
            if( bestScore == 0 )
            {
                newExtent = 1;
                auto newBlockedExtent = blockedExtent / newExtent;
                auto newNumBlocks = numBlocks * newExtent;
                float parallelismScore = std::min(1.0f, ((float)newNumBlocks) / targetNumBlocks); // larger is better
                float blockingScore = std::min(1.0f, ((float)newBlockedExtent) / targetExtentBlocked); // cap score
                const float score1 = parallelismScore * blockingScore;

                newExtent = extentSecondModeK;
                newBlockedExtent = blockedExtent / newExtent;
                newNumBlocks = numBlocks * newExtent;
                parallelismScore = std::min(1.0f, ((float)newNumBlocks) / targetNumBlocks); // larger is better
                blockingScore = std::min(1.0f, ((float)newBlockedExtent) / targetExtentBlocked); // cap score
                const float score2 = parallelismScore * blockingScore;

                if( score2 > score1 )
                    bestNewExtent = extentSecondModeK;
                else
                    bestNewExtent = 1;
            }
            const int pos = cNumBlockedModes;
            splitAndInsert(secondModeK, RESERVED_M_MODE_PW, bestNewExtent, pos, modeK, extent_, strideA_, strideB_);
        }
    }else if( blockedExtent < minBlockedExtent )
    {
        // enough parallelism but blocked extent is too small
        // e.g., if first two extents are small. In this case we could try to reorder the
        // modes (at the expense of less structured memory accesses in A) or increase
        // cNumBlockedModes (at the expense of more expensive indexing or increase binary
        // size) => At the moment we give up.
    }
//    for( auto mode : modeK )
//        printf("%d: %d %d %d\n", mode, extent_.at(mode), strideA_.at(mode), strideB_.at(mode));
    /********************/

    nmodeK_ = modeK.size();
    nmodeL_ = modeL.size();
    nmodeM_ = modeM.size();

    this->initStrideExtent(false, extentM_, strideAm_, strideCm_, modeM, strideA_, strideC_, extent_);
    this->initStrideExtent(true, extentK_, strideAk_, strideBk_, modeK, strideA_, strideB_, extent_);
    
    for(int i=0; i < LWTENSOR_MAX_MODES; ++i) {
        extentM_divmod[i] = lwtlass::FastDivmod(extentM_[i]);
        extentK_divmod[i] = lwtlass::FastDivmod(extentK_[i]);
    }


    this->setInitialized();
    return LWTENSOR_STATUS_SUCCESS;
//    printf("%d %d, %d; %d\n", numBlocks, blockedExtent_, unblockedExtent_, totalExtentK_);
}

void ReductionParams::initStrideExtent(const bool blockMode,
        extent_type* extentLocal,
        stride_type* strideALocal,
        stride_type* strideBLocal,
        const ModeList &modes,
        const StrideMap &strideA,
        const StrideMap &strideB,
        const ExtentMap &extent)
{
    extent_type blockedExtent = 1;
    extent_type unblockedExtent = 1;
    uint32_t i = 0U;
    for( auto mode : modes )
    {
        const auto lwrrentExtent = extent.at(mode);

        extentLocal[i] = lwrrentExtent;
        strideALocal[i] = strideA.at(mode);
        strideBLocal[i] = strideB.at(mode);
        
        if( blockMode )
        {
            if( i < cNumBlockedModes ){
                blockedExtent *= extentLocal[i]; // these are parallelized across threads
            }else{
                unblockedExtent *= extentLocal[i]; // these are parallelized across threadblocks
            }
        }
        i++;
    }
    if( blockMode )
    {
        unblockedExtent_ = unblockedExtent;
        blockedExtent_ = blockedExtent;
    }
}

void ContractionPlan::init(lwdaDataType_t typeScalar, const int32_t partitionsK)
{
    reductionParams_.unsetInitialized();
    gettParams_.unsetInitialized();
    gettParams_.partitions_ = partitionsK;
    initElementwisePlan(transposePlanA_);
    initElementwisePlan(transposePlanB_);
    initElementwisePlan(transposePlanC_);

    transposeA_          = false;
    transposeB_          = false;
    transposeC_          = false;
    dispatchToReduction_ = false;
    dispatchToTrinary_   = false;
    swapAB_              = false;
    useBLAS_             = false;
    useLwTe_             = false;
    candidateIdx_        = -1;
    containerIdx_        = -1;
    bufferSizeA_         = 0;
    bufferSizeB_         = 0;
    bufferSizeC_         = 0;
    workspaceSize_       = 0;
    this->typeScalar_    = typeScalar;
    this->unsetInitialized();
    requiresMeasurement_ = false;
}

int ContractionPlan::info(char* dst, const size_t sz) const
{
    if(dst == nullptr)
        return 0;
    const int64_t size = sz;
    int64_t bytesWritten = 0;
    if( candidateIdx_ >= 0 && containerIdx_ >= 0 )
    {
        const Candidate<ContractionDescriptorInternal>* candidate = nullptr;
        const ComputeEngineBase<ContractionDescriptorInternal>* contractionEngine = nullptr;

        if (!useLwTe_)
        {
            contractionEngine = getContractionEngineLwtlass();
        }
        else
        {
#ifdef LWTENSOR_ENABLE_LWTE
            contractionEngine = getContractionEngineLwte();
#endif
        }
        if (contractionEngine != nullptr)
        {
            HANDLE_ERROR(contractionEngine->getCandidatePtr(candidateIdx_, containerIdx_, candidate));
        }
        bytesWritten = gettParams_.info(dst, sz);
        if (candidate != nullptr)
        {
            bytesWritten += candidate->info(dst + bytesWritten, std::max(size - bytesWritten, (int64_t)0));
            bytesWritten += candidate->infoWithParam(this, dst + bytesWritten, std::max(size - bytesWritten, (int64_t)0));
        }
    }
    bytesWritten = snprintf(dst + bytesWritten, std::max(size - bytesWritten, (int64_t)0), "d(%d,%d,%d)t(%d,%d,%d)sw(%d)", dispatchToTrinary_, dispatchToReduction_, useBLAS_, transposeA_, transposeB_, transposeC_, swapAB_);
    return bytesWritten;
}

bool hasFreeModes(const ModeList &modeA,
                  const ModeList &modeB,
                  const ModeList &modeC)
{
    for(auto a : modeA){
        // ensure that the mode exists in C
        for(auto c : modeC){
            if( a == c )
            {
                // ensure that the mode does not exists in B (i.e., it's not a batched mode)
                auto found = false;
                for(auto b : modeB){
                    if( a == b )
                    {
                        found = true;
                        break;
                    }
                }
                if( !found )
                {
                    return true;
                }
            }
        }
    }
    return false;
}
ContractionDynamicParams::ContractionDynamicParams( const ContractionDescriptor* desc, bool &swapAB):
            typeA_((lwdaDataType_t)42), // TODO use unknown type
            typeB_((lwdaDataType_t)42),
            typeC_((lwdaDataType_t)42),
            typeCompute_(desc->getComputeType()),
            opA_(LWTENSOR_OP_UNKNOWN),
            opB_(LWTENSOR_OP_UNKNOWN),
            opC_(LWTENSOR_OP_UNKNOWN),
            alignmentReqA_(desc->getAlignmentA()),
            alignmentReqB_(desc->getAlignmentB()),
            alignmentReqC_(desc->getAlignmentC()),
            alignmentReqD_(desc->getAlignmentD()),
            stridedLoadsA_(false),
            stridedLoadsB_(false),
            contiguousModeIsBatchedA_(false),
            contiguousModeIsBatchedB_(false)
{
    swapAB = false;
    const auto *descA = desc->getDescA();
    const auto *descB = desc->getDescB();
    const auto *descC = desc->getDescC();
    const auto *modeA = desc->getModeA();
    const auto *modeB = desc->getModeB();
    const auto *modeC = desc->getModeC();

    typeA_ = descA->getDataType();
    typeB_ = descB->getDataType();
    typeC_ = descC->getDataType();

    opA_ = descA->getOp();
    opB_ = descB->getOp();
    opC_ = descC->getOp();

    strideA_.reserve(24); // avoids multiple mallocs (try to reduce overhead)
    strideB_.reserve(24); // avoids multiple mallocs (try to reduce overhead)
    strideC_.reserve(24); // avoids multiple mallocs (try to reduce overhead)
    extent_.reserve(24);  // avoids multiple mallocs (try to reduce overhead)
    if (initStrideExtentModesSorted(descA, modeA, strideA_, modeA_, extent_) != LWTENSOR_STATUS_SUCCESS)
    {
        throw InternalError("9408\n");
    }
    if (initStrideExtentModesSorted(descB, modeB, strideB_, modeB_, extent_) != LWTENSOR_STATUS_SUCCESS)
    {
        throw InternalError("9409\n");
    }
    if (initStrideExtentModesSorted(descC, modeC, strideC_, modeC_, extent_) != LWTENSOR_STATUS_SUCCESS)
    {
        throw InternalError("9410\n");
    }

    /*
     * Delete all extent-1 modes
     */
    for (auto it = modeC_.begin(); it != modeC_.end();)
    {
        auto mode = *it;
        if (extent_.at(mode) == 1)
        {
            it = modeC_.erase(it);
            strideC_.erase(mode);

            auto itA = std::find(modeA_.begin(), modeA_.end(), mode);
            if (itA != modeA_.end())
            {
                modeA_.erase(itA);
                strideA_.erase(mode);
            }
            auto itB = std::find(modeB_.begin(), modeB_.end(), mode);
            if (itB != modeB_.end())
            {
                modeB_.erase(itB);
                strideB_.erase(mode);
            }
            extent_.erase(mode);
            if (it == modeC_.end())
            {
                break;
            }
            else if (it != modeC_.begin())
            {
                it = std::prev(it); // account for it++ and erase()
            }
            // else (i.e., it == modeC_.egin()): don't advance iterator since we still have to check that mode
        }
        else
        {
            it++;
        }
    }

    // fuses modes
    if( fuseModes(modeA_, strideA_,
                  modeB_, strideB_,
                  modeC_, strideC_, extent_) != LWTENSOR_STATUS_SUCCESS)
    {
        throw InternalError("9411\n");
    }

    /* Ensure that the stride-1 mode of C always comes from A */
    if( modeC_.size() > 0 )
    {
        for (const auto mode : modeB_) {
            if (mode == modeC_.front()) {
                std::swap(modeA_, modeB_);
                std::swap(opA_, opB_);
                std::swap(typeA_, typeB_);
                std::swap(strideA_, strideB_);
                std::swap(alignmentReqA_, alignmentReqB_);
                swapAB = true;
                break;
            }
        }
    }

    bool hasFreeModesA = hasFreeModes(modeA_, modeB_, modeC_);
    bool hasFreeModesB = hasFreeModes(modeB_, modeA_, modeC_);

    // internal representation
    /** If modeM is empty, then insert a reserved mode; adds GEMV-like support */
    if ( !hasFreeModesA ) {
        assert(extent_.find(RESERVED_M_MODE) == extent_.end());
        assert(strideA_.find(RESERVED_M_MODE) == strideA_.end());
        assert(strideC_.find(RESERVED_M_MODE) == strideC_.end());
        extent_[RESERVED_M_MODE] = 1;
        if (!modeA_.empty() && strideA_.find(modeA_.back()) != strideA_.end())
            strideA_[RESERVED_M_MODE] = strideA_[modeA_.back()] * extent_.at(modeA_.back());
        else
            strideA_[RESERVED_M_MODE] = 0;
        if (!modeC_.empty() && strideC_.find(modeC_.back()) != strideC_.end())
            strideC_[RESERVED_M_MODE] = strideC_[modeC_.back()] * extent_.at(modeC_.back());
        else
            strideC_[RESERVED_M_MODE] = 0;
        modeA_.push_back(RESERVED_M_MODE);
        modeC_.push_back(RESERVED_M_MODE);
    }

    // internal representation
    /** If modeN is empty, then insert a reserved mode; adds GEMV-like support */
    if ( !hasFreeModesB ) {
        assert(extent_.find(RESERVED_N_MODE) == extent_.end());
        assert(strideB_.find(RESERVED_N_MODE) == strideB_.end());
        assert(strideC_.find(RESERVED_N_MODE) == strideC_.end());
        extent_[RESERVED_N_MODE] = 1;
        if (!modeB_.empty() && strideB_.find(modeB_.back()) != strideB_.end())
            strideB_[RESERVED_N_MODE] = strideB_[modeB_.back()] * extent_.at(modeB_.back());
        else
            strideB_[RESERVED_N_MODE] = 0;
        if (!modeC_.empty() && strideC_.find(modeC_.back()) != strideC_.end())
            strideC_[RESERVED_N_MODE] = strideC_[modeC_.back()] * extent_.at(modeC_.back());
        else
            strideC_[RESERVED_N_MODE] = 0;
        modeB_.push_back(RESERVED_N_MODE);
        modeC_.push_back(RESERVED_N_MODE);
    }

    if(initModeOrderContraction(modeA_, modeB_, modeC_, extent_,
                                modeM_, modeN_, modeK_, modeL_,
                                stridedLoadsA_, stridedLoadsB_, contiguousModeIsBatchedA_, contiguousModeIsBatchedB_) != LWTENSOR_STATUS_SUCCESS)
    {
        throw InternalError("9412\n");
    }
//    for( auto m : modeC_)
//        printf("%d,", m);
//    printf(" = ");
//    for( auto m : modeA_)
//        printf("%d,", m);
//    printf(" * ");
//    for( auto m : modeB_)
//        printf("%d,", m);
//    printf("\nM: ");
//    for( auto m : modeM_)
//        printf("(%d,%d), ", m, extent_.at(m));
//    printf("\nN: ");
//    for( auto m : modeN_)
//        printf("(%d,%d), ", m, extent_.at(m));
//    printf("\nK: ");
//    for( auto m : modeK_)
//        printf("(%d,%d), ", m, extent_.at(m));
//    printf("\n");
}

lwtensorStatus_t ColwolutionDescriptor::init(
        const Context* ctx,
        const uint32_t numModesActivation, const int32_t modeActivation[],
        const uint32_t numModesFilter, const int32_t modeFilter[],
        const uint32_t numModesOutput, const int32_t modeOutput[],
        const uint32_t numColwolvedModes, const lwtensorColwolvedMode_t colwolvedModes[],
        const uint32_t numGroups,
        const lwtensorComputeType_t typeCompute,
        const lwtensorOperator_t opOut) 
{
    if( numColwolvedModes > kMaxColwolvedModes)
        return ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "Too many colwolved modes");

    if( numGroups == 0 )
    {
        return ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "numGroups is invalid.");
    }


    numModesActivation_ = numModesActivation;
    numModesFilter_ = numModesFilter;
    numModesOutput_ = numModesOutput;
    numColwolvedModes_ = numColwolvedModes;
    numGroups_ = numGroups;

    constexpr uint32_t maxModes = ColwolutionDescriptor::MAX_MODES;
    if (numModesActivation > maxModes || numModesFilter > maxModes || numModesOutput > maxModes)
    {
        return ctx->logError(LWTENSOR_STATUS_ILWALID_VALUE, "Too many modes.");
    }

    for (uint32_t i=0; i < numModesActivation; ++i)
    {
        modeActivation_[i] = modeActivation[i];
    }
    for (uint32_t i=0; i < numModesFilter; ++i)
    {
        modeFilter_[i] = modeFilter[i];
    }
    for (uint32_t i=0; i < numModesOutput; ++i)
    {
        modeOutput_[i] = modeOutput[i];
    }
    
    for (uint32_t i=0U; i < numColwolvedModes_; ++i)
    {
        if (colwolvedModes[i].padding != 0U)
        {
            return ctx->logError(LWTENSOR_STATUS_NOT_SUPPORTED, "Padding is not supported yet.");
        }
        colwolvedModes_[i] = colwolvedModes[i];
    }
    typeCompute_ = typeCompute;
    opOut_ = opOut;

    this->setInitialized();

    return LWTENSOR_STATUS_SUCCESS;
}

int ColwolutionDescriptor::findColwolvedMode(mode_type mode) const
{
    int pos = -1;
    for (uint32_t i=0; i < numColwolvedModes_; ++i)
    {
        if  (mode == colwolvedModes_[i].modeActivation
                || ( mode == colwolvedModes_[i].modeFilter)
                || ( mode == colwolvedModes_[i].modeOutput))
        {
            return i;
        }
    }
    return pos;
}

void ColwolutionDescriptor::getOutputExtent(
        const TensorDescriptor * descActivation,
        const TensorDescriptor * descFilter,
        int64_t extent[]) const
{
    ExtentMap extent_;
    for (uint32_t i=0; i < numModesActivation_; ++i)
    {
        extent_[modeActivation_[i]] = descActivation->getExtent(i);
    }
    for (uint32_t i=0; i < numModesFilter_; ++i)
    {
        extent_[modeFilter_[i]] = descFilter->getExtent(i);
    }

    for (uint32_t i=0; i < numModesOutput_; ++i)
    {
        auto lwrrentMode = modeOutput_[i];
        const int pos = this->findColwolvedMode(lwrrentMode);
        if (pos == -1) // not a colwolved mode (just copy it)
        {
            extent[i] = extent_.at(lwrrentMode);
        }
        else
        {
            auto colwolvedMode = colwolvedModes_[pos];
            auto extentFilter = extent_.at(colwolvedMode.modeFilter);
            auto extentActivation = extent_.at(colwolvedMode.modeActivation);
            extent[i] = 1 + ( extentActivation + 2*colwolvedMode.padding - (((extentFilter-1)*colwolvedMode.dilation)+1) )/colwolvedMode.stride;
        }
    }
}

size_t ContractionDynamicParams::getHash(const uint64_t workspaceSize, const bool swapAB, const lwtensorAutotuneMode_t autotuneMode, uint32_t tag) const
{
    if (swapAB)
    {
        return ContractionDescriptorInternal::getHash(
                modeB_, strideB_, typeB_, opB_, alignmentReqB_,
                modeA_, strideA_, typeA_, opA_, alignmentReqA_,
                modeC_, strideC_, typeC_, opC_, alignmentReqC_,
                extent_, typeCompute_, workspaceSize, static_cast<uint32_t>(autotuneMode), tag);
    }
    else
    {
        return ContractionDescriptorInternal::getHash(
                modeA_, strideA_, typeA_, opA_, alignmentReqA_,
                modeB_, strideB_, typeB_, opB_, alignmentReqB_,
                modeC_, strideC_, typeC_, opC_, alignmentReqC_,
                extent_, typeCompute_, workspaceSize, static_cast<uint32_t>(autotuneMode), tag);
    }
}

lwtensorStatus_t Contractiolwariant::setFromRank(const lwtensorAlgo_t algo, const uint32_t rank)
{
    if (algo == LWTENSOR_ALGO_DEFAULT)
    {
#ifdef LWTENSOR_ENABLE_LWTE
        constexpr int kNumLwTeCandidates = 1;
#else
        constexpr int kNumLwTeCandidates = 0;
#endif

        if (rank == 0) // try best GETT kernel first
        {
            algo_ = LWTENSOR_ALGO_GETT;
            kernel_ = 0;
        }
        else if (rank == 1) // try TTGT next
        {
            algo_ = LWTENSOR_ALGO_TTGT;
            kernel_ = 0;
        }
#ifdef LWTENSOR_ENABLE_LWTE
        else if (rank == 2) // try best LwTe kernel
        {
            algo_ = LWTENSOR_ALGO_LWTE;
            kernel_ = 0;
        }
#endif
        else if (rank == 2 + kNumLwTeCandidates) // TRY second-best GETT kernel
        {
            algo_ = LWTENSOR_ALGO_GETT;
            kernel_ = 1;
        }
        else if (rank == 3 + kNumLwTeCandidates) // try TGETT
        {
            algo_ = LWTENSOR_ALGO_TGETT;
            kernel_ = 0;
        }
        else if (rank == 4) // try second-best TTGT next
        {
            algo_ = LWTENSOR_ALGO_TTGT;
            kernel_ = 1;
        }
        else  // TRY all other GETT kernels
        {
            algo_ = LWTENSOR_ALGO_GETT;
            kernel_ = rank - 3;
        }
    }
    else
    {
        algo_ = algo;
        kernel_ = rank;
    }
    return LWTENSOR_STATUS_SUCCESS;
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
IlwokeLwblasLt::IlwokeLwblasLt() :
            lA_(&lAOpaque_), lB_(&lBOpaque_), lC_(&lCOpaque_), lD_(&lDOpaque_),
            mul_(&mulOpaque_)
{
}

lwtensorStatus_t IlwokeLwblasLt::initLayout(const IlwokeLwblasLt::Params::Matrix &mat, lwblasLtMatrixLayout_t &layout) const
{
    HANDLE_ERROR(lwblasLtMatrixLayoutInit(layout, mat.type_, mat.row_, mat.col_, mat.ld_));
    if (params_.numBatched_ > 1)
    {
        HANDLE_ERROR(lwblasLtMatrixLayoutSetAttribute(layout,
                    LWBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &params_.numBatched_, sizeof(params_.numBatched_)));
        HANDLE_ERROR(lwblasLtMatrixLayoutSetAttribute(layout,
                    LWBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &mat.ldBatched_, sizeof(mat.ldBatched_)));
    }
    return LWTENSOR_STATUS_SUCCESS;
} 

/**
 * \brief Initializes lwBLAS' algo
 * \param[out] algo
 */
lwtensorStatus_t IlwokeLwblasLt::initAlgo(const lwblasLtHandle_t& handle, bool useCasD,
        uint32_t alignmentA, uint32_t alignmentB, uint32_t alignmentC,
        size_t workspaceSize, lwblasLtMatmulAlgo_t* algo, int32_t kernel) const
{
    constexpr int32_t maxRequestedAlgos = 8;
    lwblasLtMatmulHeuristicResult_t heuristic[maxRequestedAlgos];

    if (kernel >= maxRequestedAlgos)
    {
        RETURN_STATUS(LWTENSOR_STATUS_NOT_SUPPORTED);
    }

    kernel = kernel < 0 ? 0 : kernel; // deal with default-kernel (i.e. -1)
    int numReturned = 0;
    const int requestedAlgoCount = std::min(kernel + 1, maxRequestedAlgos);

    lwblasLtMatmulPreferenceOpaque_t preference;

    HANDLE_ERROR(lwblasLtMatmulPreferenceInit(&preference));
    HANDLE_ERROR(lwblasLtMatmulPreferenceSetAttribute(&preference, LWBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
    HANDLE_ERROR(lwblasLtMatmulPreferenceSetAttribute(&preference, LWBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &alignmentA, sizeof(alignmentA)));
    HANDLE_ERROR(lwblasLtMatmulPreferenceSetAttribute(&preference, LWBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &alignmentB, sizeof(alignmentB)));
    HANDLE_ERROR(lwblasLtMatmulPreferenceSetAttribute(&preference, LWBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &alignmentC, sizeof(alignmentC)));
    HANDLE_ERROR(lwblasLtMatmulPreferenceSetAttribute(&preference, LWBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &alignmentC, sizeof(alignmentC)));
    HANDLE_ERROR(lwblasLtMatmulAlgoGetHeuristic(handle, mul_, lA_,
                lB_, useCasD ? lD_ : lC_, useCasD ? lD_ : lC_,
                &preference, requestedAlgoCount, heuristic, &numReturned));
    if (numReturned <= kernel)
    {
        RETURN_STATUS(LWTENSOR_STATUS_NOT_SUPPORTED);
    }
    HANDLE_ERROR(heuristic[kernel].state);
    assert(heuristic[kernel].workspaceSize <= workspaceSize);
    *algo = heuristic[kernel].algo;
    return LWTENSOR_STATUS_SUCCESS;
}

lwtensorStatus_t IlwokeLwblasLt::execute(const lwblasLtHandle_t& handle,
        bool useCasD,
        const void* alpha, const void* A, const void *B,
        const void* beta, void* C, void* workspace, size_t workspaceSize,
        lwdaStream_t stream) const
{
    const lwblasLtMatmulAlgo_t *algo = useCasD ? &this->algoDLt_ : &this->algoCLt_;

    auto lA  = const_cast<lwblasLtMatrixLayout_t>(&lAOpaque_);
    auto lB  = const_cast<lwblasLtMatrixLayout_t>(&lBOpaque_);
    auto lC  = const_cast<lwblasLtMatrixLayout_t>(&lCOpaque_);
    auto lD  = const_cast<lwblasLtMatrixLayout_t>(&lDOpaque_);
    auto mul = const_cast<lwblasLtMatmulDesc_t>(&mulOpaque_);
    HANDLE_ERROR(lwblasLtMatmul(handle, mul,
                alpha, A, lA, B, lB,
                beta, C, useCasD ? lD : lC, C, useCasD ? lD : lC,
                algo, workspace, workspaceSize, stream));
    return LWTENSOR_STATUS_SUCCESS;
}

lwtensorStatus_t ContractionPlan::initLwblasLt(
                          int32_t numBatched,
                          lwblasOperation_t transA, int rowA, int colA, int64_t bdA, int ldA, uint32_t alignmentRequirementA,
                          lwblasOperation_t transB, int rowB, int colB, int64_t bdB, int ldB, uint32_t alignmentRequirementB,
                                                    int rowC, int colC, int64_t bdC, int ldC, uint32_t alignmentRequirementC,
                                                                        int64_t bdD, int ldD,
                          const int32_t deviceId, const DeviceProp *deviceProp, const int32_t kernel)
{
    const lwdaDataType_t &typeA = gettParams_.typeA_;
    const lwdaDataType_t &typeB = gettParams_.typeB_;
    const lwdaDataType_t &typeC = gettParams_.typeC_;
    const lwtensorComputeType_t &typeCompute = gettParams_.typeCompute_;
    const auto typeScalar = getScalarType(typeC, typeCompute);
    const size_t remainingWorkspace = workspaceSize_ - bufferSizeA_ - bufferSizeB_ - bufferSizeC_;

    RETURN_STATUS(lwblasLtIlwoke_.init(typeCompute, typeScalar, numBatched,
            transA, typeA, rowA, colA, bdA, ldA, alignmentRequirementA,
            transB, typeB, rowB, colB, bdB, ldB, alignmentRequirementB,
                    typeC, rowC, colC, bdC, ldC, alignmentRequirementC,
                                       bdD, ldD,
                    remainingWorkspace, deviceId, deviceProp, kernel));
}

/**
 * Updates max_alignment to conform with value
 * \pre max_alignment is a power of two
 * \post ret <= max_alignment, ret is a power of two, value % ret == 0
 */
static uint32_t updateAlignment(uint32_t value, uint32_t max_alignment) {
    while (value % max_alignment != 0)
    {
        max_alignment /= 2;
    }
    return max_alignment;
}

lwtensorStatus_t IlwokeLwblasLt::init(lwtensorComputeType_t typeCompute, lwdaDataType_t typeScalar, int32_t numBatched,
                          lwblasOperation_t transA, const lwdaDataType_t typeA, int rowA, int colA, int64_t bdA, int ldA, uint32_t alignmentRequirementA,
                          lwblasOperation_t transB, const lwdaDataType_t typeB, int rowB, int colB, int64_t bdB, int ldB, uint32_t alignmentRequirementB,
                                                    const lwdaDataType_t typeC, int rowC, int colC, int64_t bdC, int ldC, uint32_t alignmentRequirementC,
                                                                                                    int64_t bdD, int ldD,
                          size_t remainingWorkspace, const int32_t deviceId, const DeviceProp *deviceProp, const int32_t kernel)
{
    this->unsetInitialized();
    this->params_.init(typeCompute,
                       typeScalar,
                       numBatched,
                       transA, typeA, rowA, colA, bdA, ldA,
                       transB, typeB, rowB, colB, bdB, ldB,
                               typeC, rowC, colC, bdC, ldC,
                                                  bdD, ldD);

    lA_ = &lAOpaque_;
    lB_ = &lBOpaque_;
    lC_ = &lCOpaque_;
    lD_ = &lDOpaque_;
    mul_ = &mulOpaque_;

    HANDLE_ERROR(initLayout(params_.A_, lA_));
    HANDLE_ERROR(initLayout(params_.B_, lB_));
    HANDLE_ERROR(initLayout(params_.C_, lC_));
    HANDLE_ERROR(initLayout(params_.D_, lD_));

    HANDLE_ERROR(lwblasLtMatmulDescInit(mul_, params_.typeCompute_, params_.typeScalar_));
    HANDLE_ERROR(lwblasLtMatmulDescSetAttribute(mul_,
                LWBLASLT_MATMUL_DESC_TRANSA, &params_.opA_, sizeof(params_.opA_)));
    HANDLE_ERROR(lwblasLtMatmulDescSetAttribute(mul_,
                LWBLASLT_MATMUL_DESC_TRANSB, &params_.opB_, sizeof(params_.opB_)));

    // WAR for lwbugs 3046503: check bdA, bdB, bdC
    if (numBatched > 0)
    {
        if (bdA != 0)
        {
            alignmentRequirementA = updateAlignment(static_cast<uint32_t>(bdA * getDataTypeSize(typeA)), alignmentRequirementA);
        }
        if (bdB != 0)
        {
            alignmentRequirementB = updateAlignment(static_cast<uint32_t>(bdB * getDataTypeSize(typeB)), alignmentRequirementB);
        }
        if (bdC != 0)
        {
            alignmentRequirementC = updateAlignment(static_cast<uint32_t>(bdC * getDataTypeSize(typeC)), alignmentRequirementC);
        }
    }

    const int alignmentD = 128; // TODO is this really correct?
    HANDLE_ERROR(initAlgo(globalLwblasLtHandles[deviceId], true, alignmentRequirementA, alignmentRequirementB, alignmentD, remainingWorkspace, &algoDLt_, kernel));
    HANDLE_ERROR(initAlgo(globalLwblasLtHandles[deviceId], false, alignmentRequirementA, alignmentRequirementB, alignmentRequirementC, remainingWorkspace, &algoCLt_, kernel));

    // WAR for lwbugs 3113021: initialization could succeed but lwblasLtMatmul might still fail
    if (numBatched > deviceProp->maxGridSize[2]) // TODO Andrzej, should this be dependend on algoDLt_ and algoCLt_ and limited to fp16?
    {
        RETURN_STATUS(LWTENSOR_STATUS_NOT_SUPPORTED);
    }

    this->setInitialized();
    return LWTENSOR_STATUS_SUCCESS;
}
#endif // LWTENSOR_LWDA_VERSION_MAJOR >= 11

}
