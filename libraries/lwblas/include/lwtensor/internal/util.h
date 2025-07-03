#pragma once

#include <algorithm>
#include <string>
#include <iostream>

#include <lwda_fp16.h>
#include <lwda_runtime.h>
#include <lwblasLt.h>


#include <lwtensor.h>
#include <lwtensor/internal/types.h> // for mode_type, ...
#include <lwtensor/internal/exceptions.h>

#include <lwtensor/internal/defines.h>
namespace LWTENSOR_NAMESPACE
{
    template<int N, class T> struct __align__(sizeof(T) * N) vec_t { T a[N]; };

    template<int N> struct size2type     { using type = char[N]; };
    template<>      struct size2type<1>  { using type = char;    };
    template<>      struct size2type<2>  { using type = short;   };
    template<>      struct size2type<4>  { using type = int;     };
    template<>      struct size2type<8>  { using type = int2;    };
    template<>      struct size2type<16> { using type = int4;    };

    inline uint32_t getMaximalAlignmentPtr(const void *ptr)
    {
        const uint64_t ptrAddr = (uint64_t) ptr;
        uint64_t maximalAlignment = 256;
        while(ptrAddr % maximalAlignment != 0)
        {
            maximalAlignment /= 2;
        }
        return maximalAlignment;
    }

    /**
     * Verifies that the strides are sorted in ascending order w.r.t. modes
     *
     * \param[in] strides Maps a mode to its stride.
     * \param[in] modes Sorted list of modes.
     * \changes_elw None.
     * \behavior blocking, reentrant, thread safe
     */
    lwtensorStatus_t validateStride(const StrideMap &strides,
                                    const ModeList &modes) noexcept;
    /**
     * Print all modes in the list.
     */
    void printMode(const std::string& name, const ModeList& mode); 

    /**
     * Initialize strides assumimg a generalized column-major (dense) memory layout (i.e.,
     * strideA.front() corresponds to the stride-1 mode).
     *
     * \param[in] extent extent of each mode
     * \param[in] modeA List of modes
     * \param[out] strideA
     * \pre Each mode in modeA must be present in extent.
     * \changes_elw None.
     * \behavior blocking, reentrant, thread safe
     */
    lwtensorStatus_t  initStride(
            const ExtentMap &extent,
            const ModeList &modeA,
            StrideMap &strideA) noexcept;

    /**
     * Tries to find a value x in an array 'array' with a total of n elements.
     * \param[in] x value to search for
     * \param[in] array array of size n to search through
     * \param[in] n number of elements
     * \pre array Must not be nullptr.
     * \pre array Must have suffucient storage for at least numElements many entries.
     * \returns position of x in array
     * \retval position of x in array if x was found.
     * \retval -1 if x was not found.
     * \changes_elw None.
     * \behavior blocking, reentrant, thread safe
     */
    template<typename T>
    int32_t find(const T x, const T *const array, const uint32_t numElements) noexcept
    {
        for(uint32_t i = 0U; i < numElements; ++i)
        {
            if( array[i] == x )
            {
                return static_cast<int32_t>(i);
            }
        }
        return -1;
    }

    /**
     * Checks if a pointer to an element of type 'type' is zero.
     * \param[in] ptr Pointer to the element in question.
     * \param[in] type Data type of the element in question.
     * \pre ptr Must not be nullptr.
     * \returns Return true iff the element in question is zero, false otherwise.
     * \throws lwtensor::NotSupported if data type is not supported.
     * \changes_elw None.
     * \behavior blocking, reentrant, thread safe
     */
    bool isZero(const void* const ptr, const lwdaDataType_t type);

    /**
     * \param[in] type Data type
     * \returns Size of data type (in bytes).
     * \throws lwtensor::NotSupported if data type is not supported.
     * \changes_elw None.
     * \behavior blocking, reentrant, thread safe
     */
    size_t getDataTypeSize(const lwdaDataType_t type);

    /**
     * Colwerts lwdaError_t into lwtensorStatus_t
     * \param[in] err lwca error code.
     * \returns Corresponding lwtensorStatus_t
     * \changes_elw None.
     * \behavior blocking, reentrant, thread safe
     */
    lwtensorStatus_t handleError( const lwdaError_t err) noexcept;

    /**
     * Colwerts lwblasStatus_t into lwtensorStatus_t
     * \param[in] err lwblas error code.
     * \returns Corresponding lwtensorStatus_t
     * \changes_elw None.
     * \behavior blocking, reentrant, thread safe
     */
    lwtensorStatus_t handleError( const lwblasStatus_t err) noexcept;

    /**
     * No-op; just passes error through.
     * \param[in] err lwtensor error code.
     * \returns err
     * \changes_elw None.
     * \behavior blocking, reentrant, thread safe
     */
    lwtensorStatus_t handleError( const lwtensorStatus_t err ) noexcept;

    /**
     * Passes error through, used when LWTENSOR_LOGINFO_DBG = 0.
     * \param[in] err lwtensor error code.
     * \param[in] desc Description of error.
     * \returns err
     * \changes_elw None.
     * \behavior blocking, reentrant, thread safe
     */
    lwtensorStatus_t handleError( const lwtensorStatus_t err, const std::string &desc ) noexcept;

    /**
     * Passes error through and prints the provided description to std::cerr.
     * \param[in] err lwtensor error code.
     * \param[in] desc Description of error.
     * \returns err
     * \changes_elw None.
     * \behavior blocking, reentrant, thread safe
     */
    lwtensorStatus_t handleError_log( const lwtensorStatus_t err, const std::string &desc ) noexcept;

#ifdef DEBUG
#define RETURN_STATUS( err_expr ) { const lwtensorStatus_t err = err_expr; printf("%s:%d : %s\n",__FILE__, __LINE__, lwtensorGetErrorString(err)); return err; }
#else
#define RETURN_STATUS( err_expr ) { return err_expr; }
#endif

#define LWTENSOR_LOG_API( ctx, kLogLevel, desc ) \
                if (ctx->logLevel_ >= kLogLevel)\
                {\
                    std::cerr << desc;\
                }

#define HANDLE_ERROR_THROW( err_expr ) {const auto err2 = LWTENSOR_NAMESPACE::handleError(err_expr); if( err2 != lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS ) { throw NotSupported("4123");}} 
#define HANDLE_ERROR( err_expr ) {const auto err2 =  LWTENSOR_NAMESPACE::handleError(err_expr); if( err2 != lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS ) {RETURN_STATUS(err2)}}
    /**
     * This function fuses multiple modes that are conselwtive in all tensors (or in two if
     * they only appear in two).
     *
     * \param[in,out] modeA Modes of A; all modes must be sorted w.r.t. their strides
     * \param[in,out] strideA Strides of A for each mode
     * \param[in]     vectorModeA vector mode of A (must be LWTENSOR_ILWALID_MODE if not vectorized)
     * \param[in,out] modeB Modes of B; all modes must be sorted w.r.t. their strides
     * \param[in,out] strideB Strides of B for each mode
     * \param[in]     vectorModeB vector mode of A (must be LWTENSOR_ILWALID_MODE if not vectorized)
     * \param[in,out] modeC Modes of C; all modes must be sorted w.r.t. their strides
     * \param[in,out] strideC Strides of C for each mode
     * \param[in]     vectorModeC vector mode of A (must be LWTENSOR_ILWALID_MODE if not vectorized)
     * \param[in,out] extent Extents after fusing
     * \returns lwtensorStatus_t::LWTENSOR_STATUS_SUCCESS on success, otherwise a corresponding error code is returned.
     * \pre modeA all musts must be sorted in ascending order w.r.t. their strides.
     * \pre modeB all musts must be sorted in ascending order w.r.t. their strides.
     * \pre modeC all musts must be sorted in ascending order w.r.t. their strides.
     * \pre each mode of B must appear in either A or C.
     * \pre each mode must be present in extent
     * \throws May throw std::bad_alloc if insertion into std::unordered_set fails.
     * \changes_elw None.
     * \behavior blocking, reentrant, thread safe
     */
    lwtensorStatus_t fuseModes(
            ModeList &modeA, StrideMap &strideA,
            ModeList &modeB, StrideMap &strideB,
            ModeList &modeC, StrideMap &strideC,
            ExtentMap &extent );

    /**
     * Checks if the provided mode array has any duplicate entries.
     *
     * \param[in] modes Mode array of size (at least) numModes.
     * \param[in] numModes Number of modes in 'modes'.
     * \returns true iff modes has any duplicate entries.
     * \pre modes Must have suffucient storage for at least numModes many entries.
     * \throws May throw std::bad_alloc if insertion into std::unordered_set fails.
     * \changes_elw None.
     * \behavior blocking, reentrant, thread safe
     */
    bool hasDuplicates(const mode_type *const modes, const uint32_t numModes);

    /**
     * Initializes strides, modes and extent.
     *
     * The mode of modesSorted will be sorted w.r.t. asc strides
     * \param[in] desc tensor descriptor corresponding to modesUnsorted
     * \param[in] modesUnsorted Array of size (at least) desc->getNumModes()
     * \param[out] strides Strides of each mode
     * \param[out] modesSorted Modes in asc. order w.r.t. strides
     * \param[out] extent Extents of each mode
     * \pre strides, modesSorted, and extent must be of size 0
     * \throws May throw std::bad_alloc if insertion into std::unordered_set fails.
     * \changes_elw None.
     * \behavior blocking, reentrant, thread safe
     */
    lwtensorStatus_t initStrideExtentModesSorted(
            const TensorDescriptor *const desc,
            const mode_type *const modesUnsorted,
            StrideMap& strides,
            ModeList& modesSorted,
            ExtentMap &extent );

    bool isValidLwdaDataType(const lwdaDataType_t computeType) noexcept;

    /**
     * Checks if the provided operator is a valid unary operator.
     *
     * \param[in] op Operator in question.
     * \param[in] computeType Compute-type for which the operator 'op' will be used.
     * \returns checks if the provided operator is a valid unary operator for the given
     * compute type
     * \changes_elw None.
     * \behavior blocking, reentrant, thread safe
     */
    bool isValidUnaryOperator( const lwtensorOperator_t op, const lwdaDataType_t computeType ) noexcept;

    /**
     * Checks if the provided operator is a valid binary operator.
     *
     * \param[in] op Operator in question.
     * \param[in] computeType Compute-type for which the operator 'op' will be used.
     * \returns checks if the provided operator is a valid binary operator for the given
     * compute type
     * \changes_elw None.
     * \behavior blocking, reentrant, thread safe
     */
    bool isValidBinaryOperator( const lwtensorOperator_t op, const lwdaDataType_t computeType ) noexcept;

    /**
     * Handles the exception and translates it into a lwtensorStatus_t
     * 
     * \param[in] e Exception
     * \changes_elw None.
     * \behavior blocking, reentrant, thread safe
     */
    lwtensorStatus_t handleException(const std::exception& e);

    /**
     * Returns true if the autotuneMode is valid, false otherwise.
     */
    bool isValidAutotuneMode(const lwtensorAutotuneMode_t &autotuneMode);

    /**
     * Returns true if the cacheMode is valid, false otherwise.
     */
    bool isValidCacheMode( const lwtensorCacheMode_t &cacheMode);
}
