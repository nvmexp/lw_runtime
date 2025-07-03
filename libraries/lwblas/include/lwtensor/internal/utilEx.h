#pragma once

#include <unordered_map>

#include <lwda_runtime.h>
#include <lwblas_v2.h>

#include <lwtensor/types.h>
#include <lwtensor/internal/util.h>
#include <lwtensor/internal/operators.h>

#include<lwtensor/internal/defines.h>
#include<lwtensor/internal/types.h>

namespace LWTENSOR_NAMESPACE
{
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
    static const BFloat16 lwtensorOneBF16__ = BFloat16(1.0f);
#endif
    static const half lwtensorOneFP16__ = half(1.0f);
    static const float lwtensorOneFP32__ = 1.0f;
    static const double lwtensorOneFP64__ = 1.0;
    static const int8_t lwtensorOneI8__ = 1;
    static const uint8_t lwtensorOneU8__ = 1;
    static const int8_t lwtensorOneI32__ = 1;
    static const lwComplex lwtensorOneC32__ = make_lwFloatComplex(1.0f, 0.0f);
    static const lwDoubleComplex lwtensorOneC64__ = make_lwDoubleComplex(1.0, 0.0);

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
    static const BFloat16 lwtensorZeroBF16__ = BFloat16(0.0f);
#endif
    static const half lwtensorZeroFP16__ = half(0.0f);
    static const float lwtensorZeroFP32__ = 0.0f;
    static const double lwtensorZeroFP64__ = 0.0;
    static const int8_t lwtensorZeroI8__ = 0;
    static const uint8_t lwtensorZeroU8__ = 0;
    static const int8_t lwtensorZeroI32__ = 0;
    static const lwComplex lwtensorZeroC32__ = make_lwFloatComplex(0.0f, 0.0f);
    static const lwDoubleComplex lwtensorZeroC64__ = make_lwDoubleComplex(0.0, 0.0);

    const void* lwtensorGetOnePtr(lwdaDataType_t type);
    const void* lwtensorGetZeroPtr(lwdaDataType_t type);

    constexpr __device__ bool isPower2(const uint32_t n) { return (n & (n - 1)) == 0; }

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
    lwblasComputeType_t getLwblasComputeType(lwtensorComputeType_t typeCompute);
#endif // LWTENSOR_LWDA_VERSION_MAJOR >= 11

    lwdaDataType_t getScalarType(const lwdaDataType_t typeOutput, const lwtensorComputeType_t typeCompute);

    /**
     * Translates lwTENSOR's compute type to lwdaDataType_t
     * \param[in] isComplexValued This flag determines if the underlying computation is complex-valued.
     * \param[in] typeCompute
     */
    lwdaDataType_t computeTypeToLwda(const lwtensorComputeType_t typeCompute, const bool isComplexValued);

    /// checks if typeA is at least as accurate as the typeB
    bool lwdaTypeAsAclwrateAs(const lwdaDataType_t typeA, const lwdaDataType_t typeB);

    bool isValidComputeType(const lwtensorComputeType_t typeCompute) noexcept;
    
    lwdaDataType_t lwdaDataTypeToReal(const lwdaDataType_t computeType);

    /**
     * colwerts legacy values of lwtensorComputeType_t to the new values (which no longer
     * distinguish between real an complex)
     * \param[in] typeCompte
     */
    lwtensorComputeType_t normalizeComputeType(const lwtensorComputeType_t typeCompute) noexcept;
    
    /**
     * Colwerts from datatype typeSrc to dataType typeDst
     */
    lwtensorStatus_t colwert(const void* src, lwdaDataType_t typeSrc,
                                   void* dst, lwdaDataType_t typeDst);

    /**
     * Returns the neutral element w.r.t. the specified operator.
     */
    template <typename typeCompute>
    __host__ __device__ static typeCompute getNeutralElement(
            const lwtensorOperator_t opReduce)
    {
        typeCompute initValue = lwGet<typeCompute>(0);
        if (opReduce == LWTENSOR_OP_ADD)
        {
            initValue = lwGet<typeCompute>(0);
        }
        else if (opReduce == LWTENSOR_OP_MUL)
        {
            initValue = lwGet<typeCompute>(1);
        }
        else if (opReduce == LWTENSOR_OP_MIN)
        {
            initValue = lwMaxOfType<typeCompute>();
        }
        else if (opReduce == LWTENSOR_OP_MAX)
        {
            initValue = lwLowestOfType<typeCompute>();
        }
        //    else if (opReduce == LWTENSOR_OP_NORM1)
        //    {
        //        initValue = lwGet<typeCompute>(0);
        //    }
        //    else if (opReduce == LWTENSOR_OP_NORM2)
        //    {
        //        initValue = lwGet<typeCompute>(0);
        //    }
        return initValue;
    }

    template<typename T, typename S>
    inline void sortArray(const ModeList& mode,
            const S& tosort,
            T* sorted)
    {
        uint32_t counter = 0;
        for (auto idx : mode)
        {
            if (tosort.find(idx) == tosort.end())
                throw InternalError("Error: Mode not found 87.\n");
            sorted[counter] = tosort.at(idx);
            counter++;
        }
    }

    template<typename T>
    static T lwtensorMin__(T a, T b)
    {
        return (a<b)?a:b;
    }
    template<typename T>
    static T lwtensorMax__(T a, T b)
    {
        return (a>b)?a:b;
    }

    /**
     * \brief This function tries to split a mode with an 'extent' via an "multiplicative"-split.
     *
     * This function tries to split a mode with an 'extent' via an "multiplicative"-split
     * such that the resulting number of CTAs is roughly 4 * numSMs.
     *
     * \param[in] extent Extent of the mode that should be split.
     * \param[in] numCTAs number of CTAs that are already available (i.e., without
     * splitting this mode; e.g., those CTAs resulting from blocking along m- and n-
     * modes).
     * \param[in] vec The vectorization used along the mode that should be split.
     * \pre extent must be divisible by vec
     *
     * \return Extent of the new mode ('newExtent'). To be precise: extent = newExtent * (extent/newExtent).
     */
    extent_type findGoodSplitK(const extent_type extent,
                               const int numCTAs,
                               const int numSMs,
                               const extent_type vec);
    /** 
     * \brief Split mode into two modes (mode and newMode) of size (oldExtent / newExtent) and newExtent, respectively. 
     *
     * Split 'mode' into two modes of size (oldExtent / newExtent) and newExtent, respectively.
     *
     * \param[in] mode The name of the mode that should be split
     * \param[in] newMode The name of the new mode
     * \param[in] newExtent The extent of newMode
     * \param[in] pos the position to insert the new mode w.r.t. the moderOrder
     * \param[in,out] modeOrder This list could correspond to modeK (or modeM, modeN, modeL).
     * \param[in,out] extent Map that keeps track of the extent of each mode
     * \param[in,out] strideA Map that keeps track of the stride of each mode w.r.t. A
     * \param[in,out] strideB Map that keeps track of the stride of each mode w.r.t. B
     *
     * \pre mode must be part of modeOrder
     * \pre newMode must not be part of modeOrder
     *
     * \return newMode will be added to modeOrder, extent, strideA, and strideB. The extent of mode will be updated.
     */
    void splitAndInsert(const mode_type mode,
                        const mode_type newMode,
                        const extent_type newExtent, 
                        const int pos,
                        ModeList &modeOrder,
                        ExtentMap &extent,
                        StrideMap &strideA,
                        StrideMap &strideB);

    extent_type getTotalModeExtent(const ModeList &modes,
            const ExtentMap &extent);

    /**
     * Return floap multiplier for equivalient GEMM (i.e., 8 for complex- and 2 for
     * real-valued contractions).
     */
    float getFlopMultiplier(bool useComplex);

    /**
     * Checks if the provided data type is a complex-valued type
     */
    bool isComplex(lwdaDataType_t type);

    lwtensorStatus_t isHostPtr(const void *ptr, bool *isHost);

    int getVectorization(lwdaDataType_t floatType);

    void intersect(const mode_type *a, uint32_t na, const mode_type *b, uint32_t nb, 
            ModeList &intersection);

    void intersect(const ModeList &a, 
            const ModeList &b, 
            ModeList &intersection);

    /** return the mode with the largest extent */
    mode_type getMaxMode(const ModeList &modes, const ExtentMap &extent);

    bool hasDuplicates(const mode_type *a, const uint32_t na, std::unordered_map<mode_type, uint32_t> &count);

    lwtensorStatus_t validateModes(const mode_type *a, uint32_t na, 
            const mode_type *b, uint32_t nb, 
            const mode_type *c, uint32_t nc,
            const bool isReduction = false);

    lwtensorStatus_t getBlockingAutotuning(lwdaDataType_t typeA, lwdaDataType_t typeB, lwdaDataType_t typeC, lwdaDataType_t typeCompute,
            int nmodeM, int nmodeN, int nmodeK, 
            bool transA, bool transB, bool stridedLoadsA, bool stridedLoadsB, bool multiDimBlockingRequiredM, 
            int variantId, int &mc0, int &mc1, int &nc0, int &nc1, int &kc, bool &useLargeK, 
            float *dummyA, float *dummyB, float *dummyC, float *dummyCompute);

    template<typename T>
    T roundUp(T value, T multiple)
    {
        auto remainder = value % multiple;
        if( remainder == 0 )
            return value;
        else
            return value + (multiple - remainder);
    }

    template<typename T>
    T roundDown(T value, T multiple)
    {
        auto remainder = value % multiple;
        if( remainder == 0 )
            return value;
        else
            return roundUp(value, multiple) - multiple;
    }

    template <int X, int Y> struct min_t {
        constexpr static int value = X < Y ? X : Y;
    };
    template <int X, int Y> struct max_t {
        constexpr static int value = X > Y ? X : Y;
    };


    template<uint32_t N, class T1, class T2, typename typeCompute> 
    static __inline__ __device__ T2 colwert( const T1 in, const lwtensorOperator_t op,
            const ElementwiseParameters::ActivationContext *ctx = nullptr )
    {
        T2 out;
        using NolwecT2 = typename std::remove_reference<decltype(out.a[0])>::type;
        for(uint32_t i=0; i < N; ++i)
        {
            out.a[i] = lwGet<NolwecT2>( lwtensorUnaryOp(lwGet<typeCompute>(in.a[i]), op, ctx) );
        }
        return out;
    }

    float getPeakGFlops(lwtensorComputeType_t typeCompute);

    /**
     * \return a list of contiguous indices in 'src' that appear in 'tar'.
     */
    inline ModeList getContiguousIndices( const ModeList& tar, const ModeList& src )
    {
        ModeList ret;
        ModeMap<bool> targetSet;
        for( auto a : tar )
        {
            targetSet[a] = true;
        }
        /** Loop over each index in source in order. */
        for( auto a : src )
        {
            /** Insert a into the retrun list if a is in target. */
            if (targetSet.find(a) != targetSet.end()) ret.push_back(a);
            // if( std::find( tar.begin(), tar.end(), a ) != tar.end() ) ret.push_back( a );
            /** Otherwise, exit and return. */
            else if (! ret.empty()) break;
        }
        return ret;
    }

    /**
     * \return true if lhs[:] == rhs[:].
     */
    template<typename T>
    bool isTheSameList( const T& lhs, const T& rhs )
    {
        /** Early return if the sizes are different. */
        if ( lhs.size() != rhs.size() ) return false;
        /** Loop over all elements and check. */
        for ( auto itl  = lhs.begin(), itr  = rhs.begin();
                (itl != lhs.end()) &&  (itr != rhs.end());
                itl ++,             itr ++ )
        {
            if ( *itl != *itr ) return false;
        }
        /** All elements are in the same order. Return true. */
        return true;
    }

    /**
     * \return true iff stride[ 0 ] == 1 and stride[ i ] == stride[ i - 1 ] * extent[ i - 1 ].
     */
    inline bool strideMatchesExtent( const ModeList& mode,
            const StrideMap& stride,
            const ExtentMap& extent )
    {
        /** Early return if the mode is empty. */
        if ( mode.empty() ) return true;

        /** Now we use the iterator to loop over each mode. */
        auto it = mode.begin();

        /** Check for if the first mode has stride 1. */
        //if ( stride[ *it ] != 1 ) return false;
        if ( stride.find( *it )->second != 1 ) return false;

        /** Remember the current mode as prev. */
        auto prev = *it;

        /** Move the iterator forward. */
        it ++;

        /** If there is only one mode, then return true. */
        if ( it == mode.end() ) return true;

        /** Now loop over the rest. */
        for (; it != mode.end(); it ++ )
        {
            /** Check if stride[ i ] == stride[ i - 1 ] * extent[ i - 1 ]? */
            auto lwrr_stride = stride.find( *it )->second;
            auto prev_stride = stride.find( prev )->second;
            auto prev_extent = extent.find( prev )->second;
            if ( lwrr_stride != prev_stride * prev_extent ) return false;
            /** Update prev with the current mode. */
            prev = *it;
        }
        /** All strides matches. Return true. */
        return true;
    }


    typedef enum { LWTENSOR_ROUTINE_EW = 0,
                   LWTENSOR_ROUTINE_TC = 1,
                   LWTENSOR_ROUTINE_REDUCTION = 2,
                   LWTENSOR_ROUTINE_UNKNOWN = 64} lwtensorRoutine_t;

    std::string colwertToCommandline(lwdaDataType_t typeA);

    std::string reproduceCommand(
        const TensorDescriptor *descA, const int* modeA, const uint32_t alignmentA,
        const TensorDescriptor *descB, const int* modeB, const uint32_t alignmentB,
        const TensorDescriptor *descC, const int* modeC, const uint32_t alignmentC,
        int typeCompute, lwtensorAlgo_t algo, const uint64_t workspaceSize, const int32_t partitionsK,
        lwtensorRoutine_t routine);

/**
 * Create mode sets modeM, modeN, modeK, and modeL.
 *
 * Create modes sets modeM, modeN, modeK, and modeL for the free indices of A, free indices of B,
 * contracted modes as well as batched modes, respectively. The order in which the modes
 * appear in modeM, modeN, modeK reflect the order in which they will be traversed inside
 * of the tensor contraction kernel; thus, the order is critical for perfomance.
 *
 * \param[in] modeA modes of A
 * \param[in] modeB modes of B
 * \param[in] modeC modes of C
 * \param[in] extent Tracks the extents for every mode.
 * \param[out] modeM Free modes of A (i.e., intersection of modes of A and C)
 * \param[out] modeN Free modes of B (i.e., intersection of modes of B and C)
 * \param[out] modeK Contracted modes (i.e., intersection of modes of A and B)
 * \param[out] modeL Batched modes (i.e., intersection of modes of A and B and C)
 */
    lwtensorStatus_t initModeOrderContraction(const ModeList& modeA,
                               const ModeList& modeB,
                               const ModeList& modeC,
                               const ExtentMap& extent,
                               ModeList& modeM,
                               ModeList& modeN,
                               ModeList& modeK,
                               ModeList& modeL,
                               bool& stridedLoadsA, bool& stridedLoadsB,
                               bool &contiguousModeIsBatchedA,
                               bool &contiguousModeIsBatchedB);

}
