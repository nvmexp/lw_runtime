#pragma once

#include <lwda_runtime.h>
#include <lwblasLt.h>

#include <lwtensor/types.h>
#include <lwtensor/internal/types.h>
#include <lwtensor/internal/utilEx.h>
#include <lwtensor/internal/contractionDescriptor.h>
#include <lwtensor/internal/deviceProp.h>
#include <lwtensor/internal/defines.h>

#include "lwtlass/fast_math.h"


namespace LWTENSOR_NAMESPACE
{
    class ContractionDynamicParams;

    /** The TTGT candidate return from getBestTTGTCandidate. */
    class TTGTCandidate
    {
        public:
            TTGTCandidate(const ContractionDynamicParams &params, const bool useHybrid, int32_t candidateIdx);

            static const int32_t kNumCandidates = 4; // NN, TN, NT, TT

        ModeList modeA_; // modes after transpose
        ModeList modeB_; // modes after transpose
        ModeList modeC_; // modes after transpose
        bool transposeA_;
        bool transposeB_;
        bool transposeC_;
        uint64_t sizeA_;
        uint64_t sizeB_;
        uint64_t sizeC_;
        uint64_t sizeD_;

        size_t getRequiredWorkspace() const;
        size_t getRecommendedWorkspace() const;

        void print() const;
    };

    /**
     * \brief This data structure describes a contraction operation between tensors.
     *
     * \details This data structure describes the (physical) layout of a tensor, it encapsulates
     *          information such as the number of modes, the extent and stride of each mode as well
     *          as information about the vectorization of the tensor (e.g., vector index,
     *          vector-width).
     * \req None
     * \Ilwariants None
     */
    class ContractionDescriptor : public Initializable<44>
    {
        public:
            /**
            */
            lwtensorStatus_t initContractionDescriptor(
                                  const Context* ctx,
                                  const TensorDescriptor* descA, const int32_t modeA[], const uint32_t alignmentRequirementA,
                                  const TensorDescriptor* descB, const int32_t modeB[], const uint32_t alignmentRequirementB,
                                  const TensorDescriptor* descC, const int32_t modeC[], const uint32_t alignmentRequirementC,
                                  const TensorDescriptor* descD, const int32_t modeD[], const uint32_t alignmentRequirementD,
                                  lwtensorComputeType_t minComputeType);

            inline TensorDescriptor const* getDescA() const
            {
                return &descA_;
            };

            inline TensorDescriptor const * getDescB() const
            {
                return &descB_;
            };

            inline TensorDescriptor const* getDescC() const
            {
                return &descC_;
            };

            inline mode_type const* getModeA() const
            {
                return modeA_.data();
            };

            inline mode_type const * getModeB() const
            {
                return modeB_.data();
            };

            inline mode_type const* getModeC() const
            {
                return modeC_.data();
            };

            inline uint32_t getAlignmentA() const
            {
                return alignmentRequirementA_;
            }

            inline uint32_t getAlignmentB() const
            {
                return alignmentRequirementB_;
            }

            inline uint32_t getAlignmentC() const
            {
                return alignmentRequirementC_;
            }

            inline uint32_t getAlignmentD() const
            {
                return alignmentRequirementD_;
            }

            inline lwtensorComputeType_t getComputeType() const
            {
                return minComputeType_;
            }

            /**
             * Returns a string that uniquely identifies the given tensor contraction;
             * this is mostly used to identify the contraction within the plan cache.
             * This function canonicalizes the tensor contraction:
             *    - Modes/extents are sorted w.r.t. their strides
             *    - Modes/extents are fused (if possible)
             *    - Modes are canonicalized
             *
             *   TODO: WARNING: the key is a compressed representation of the problem
             *   description (i.e., theoretically there could be clashes; in practice
             *   --even after 1M different problems-- we did not encounter it)
             */
            size_t getHash(const uint64_t workspaceSize, const lwtensorAutotuneMode_t autotuneMode) const
            {
                // canonicalize tensor contraction:
                bool swapAB = false;
                ContractionDynamicParams tcParams(this, swapAB); // sort modes (w.r.t. strides) and fuses modes

                return tcParams.getHash(workspaceSize, swapAB, autotuneMode, tag_);
            }

            void info(char* dst, size_t sz) const
            {
                if( sz <= 0 )
                {
                    return;
                }
                std::stringstream s;
                s << "desc:";
                s << "A(" << std::to_string(static_cast<int32_t>(descA_.getDataType())) << "," << std::to_string(static_cast<int32_t>(descA_.getOp())) << "," << std::to_string(alignmentRequirementA_) << ")";
                for (uint32_t i = 0 ; i < descA_.getNumModes(); i++)
                {
                    s << modeA_[i] << "(" << descA_.getExtent(i) << ":" << descA_.getStride(i) << ")";
                }
                s << "B(" << std::to_string(static_cast<int32_t>(descB_.getDataType())) << "," << std::to_string(static_cast<int32_t>(descB_.getOp())) << "," << std::to_string(alignmentRequirementB_) << ")";
                for (uint32_t i = 0 ; i < descB_.getNumModes(); i++)
                {
                    s << modeB_[i] << "(" << descB_.getExtent(i) << ":" << descB_.getStride(i) << ")";
                }
                s << "C(" << std::to_string(static_cast<int32_t>(descC_.getDataType())) << "," << std::to_string(static_cast<int32_t>(descC_.getOp())) << "," << std::to_string(alignmentRequirementC_) << ")";
                for (uint32_t i = 0 ; i < descC_.getNumModes(); i++)
                {
                    s << modeC_[i] << "(" << descC_.getExtent(i) << ":" << descC_.getStride(i) << ")";
                }
                s << std::to_string(static_cast<int32_t>(minComputeType_));
                strncpy(dst, s.str().c_str(), sz - 1);
                dst[sz - 1] = '\0';
            }

            void setTag(const uint32_t tag) noexcept { tag_ = tag; }

        protected:

            // Descriptors
            TensorDescriptor descA_;
            TensorDescriptor descB_;
            TensorDescriptor descC_;

            // Modes (as defined by the user)
            std::array<mode_type,TensorDescriptor::LWTENSOR_MAX_MODES> modeA_;
            std::array<mode_type,TensorDescriptor::LWTENSOR_MAX_MODES> modeB_;
            std::array<mode_type,TensorDescriptor::LWTENSOR_MAX_MODES> modeC_;

            // Aligment requirement
            uint32_t alignmentRequirementA_;
            uint32_t alignmentRequirementB_;
            uint32_t alignmentRequirementC_;
            uint32_t alignmentRequirementD_;

            // Data type
            lwtensorComputeType_t minComputeType_;

            uint32_t tag_; ///< used to distinguish two --otherwise identical-- tensor contractions w.r.t. the cache (i.e., affects key)
    };

    // Check that ContractionDescriptor fits in lwtensorContractionDescriptor_t
    static_assert(sizeof(ContractionDescriptor) <= sizeof(lwtensorContractionDescriptor_t),
                  "Size of ContractionDescriptor greater than lwtensorContractionDescriptor_t");

    struct ReductionParams : public Initializable<45>
    {
        ReductionParams() {};

        lwtensorStatus_t init(
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
                const ExtentMap &extent);

        void initStrideExtent(const bool blockMode,
                extent_type* extentLocal,
                stride_type* strideALocal,
                stride_type* strideBLocal,
                const ModeList &modes,
                const StrideMap &strideA,
                const StrideMap &strideB,
                const ExtentMap &extents);

        /// maximum number of modes (aka dimensions) supported by this data structure.
        static constexpr int LWTENSOR_MAX_MODES = LWTENSOR_NAMESPACE::kMaxNumModes + 4;

        /// number of threads in m-dimension (only applicable if non-transA case!!!)
        static constexpr int NUM_THREADS_M = 16;
        static constexpr uint32_t targetNumBlocks = 512U; // we want to have roughly this many threadblocks active
        static constexpr uint32_t cNumBlockedModes = 2; // This parameter enables us to fuse !!UP TO!! two modes. We could easily set this to 1 (or templatize different parameters, at the expense of binary size).
        static constexpr int cPreferredVectorWidthBytes = 16; // each thread should load 16 conselwtive bytes

        uint32_t nmodeM_;
        uint32_t nmodeK_;
        uint32_t nmodeL_;

        // The total extent along the k-modes is split along blockedExtentK_ and unblockedExtentK_
        extent_type blockedExtent_; //< maps to a threads
        extent_type unblockedExtent_; //< mapst to threadblocks

        extent_type extentM_[LWTENSOR_MAX_MODES];
        extent_type extentK_[LWTENSOR_MAX_MODES];
        stride_type strideAm_[LWTENSOR_MAX_MODES];
        stride_type strideCm_[LWTENSOR_MAX_MODES];
        stride_type strideAk_[LWTENSOR_MAX_MODES];
        stride_type strideBk_[LWTENSOR_MAX_MODES];
        
        lwtlass::FastDivmod extentK_divmod[LWTENSOR_MAX_MODES];       
        lwtlass::FastDivmod extentM_divmod[LWTENSOR_MAX_MODES];
        
        
        

        extent_type totalExtentM_;
        extent_type totalExtentK_;

        lwdaDataType_t typeA_;
        lwdaDataType_t typeB_;
        lwdaDataType_t typeC_;
        lwtensorComputeType_t typeCompute_;
        lwtensorOperator_t opA_;
        lwtensorOperator_t opB_;
        lwtensorOperator_t opC_;
        lwtensorOperator_t opAB_;
        lwtensorOperator_t opReduce_;
    };

    template<typename Params_>
    class Candidate : public Initializable<139>
    {
        public:
            using Params = Params_;

            virtual void init() = 0;

            /// exelwtes the tensor contraction
            virtual lwtensorStatus_t operator() (const Context *ctx,
                            const Params &params,
                            const void* alpha, const void *A,
                            const void* beta,  const void *B,
                            const void* gamma, const void *C, void *D,
                            void* workspace, uint64_t workspaceSize, lwdaStream_t stream) const = 0;


            /// checks if the candidate is applicable to the problem
            virtual bool isApplicable(const Context *ctx,
                                      const Params &params,
                                      const size_t workspaceSize) const = 0;

            /// prints kernel infos to std::cout
            virtual void print() const = 0;

            /// prints kernel infos to dst
            virtual int info(char* dst, size_t sz) const = 0;

            /// prints kernel infos to dst
            virtual int infoWithParam(const void* param, char* dst, size_t sz) const { (void)param; (void) dst; (void) sz; return 0; }
#ifdef LWTENSOR_SPLIT_K_SWEEP
            virtual int getNumThreadblocks(const void* param) const {(void) param; return 0;}
#endif
    };

    template<typename Params_, typename CandidateInfo_>
    class CandidateTyped : public Candidate<Params_>
    {
        public:
            using CandidateInfo = CandidateInfo_;

            /**
             * Compute the features of this candidate, later used by heuristic to select best candidate.
             */
            virtual CandidateInfo getCandidateInfo() const = 0;
    };

    template <typename Params_, typename CandidateInfo_>
    class Heuristic
    {
        public:
            using Params = Params_;
            using CandidateInfo = CandidateInfo_;

            virtual std::string getName() const
            {
                return "Heuristic";
            }

            /**
             * Returns number of features this heuristc uses, it later assumes that
             *      features vector will have at least this length.
             */
            virtual int numberOfFeatures() const = 0;

            /**
             * Compute the features for given candidate.
             * Assumes that features is at least numberOfFeatures() long.
             * \param[out] features
             */
            virtual void computeFeatures(const Params &params,
                                         const CandidateInfo &candidateInfo,
                                         const DeviceProp* deviceProp,
                                         float* features) const = 0;

            /**
             * Compute scores for all candidate (in a batched fashion).
             * Assumes that scores is at least numberOfFeatures() long.
             * Assumes that features is at least numberOfCandidates * numberOfFeatures() long.
             * \param[out] scores
             */
            virtual void evaluate(int numberOfCandidates,
                                  const float* features,
                                  float* scores) const = 0;
    };

template<typename ElementA,
         typename ElementB,
         typename ElementC,
         typename ElementScalar,
         typename ElementCompute, ///< the minimal compute type encountered in the computation (e.g., for hmma this would be FP16)
         typename MinimalElementCompute, ///< denotes the least precise compute type for the given trait (e.g., TC + SM70 + ElementCompute => F16)
         typename OpClass,
         typename MathOperatorTag,
         int ccTarget, ///< specifies the compute-capability that the kernels will be compiled for
         int ccTargetMin,
         int ccTargetMax
         >
struct ContractionTraits
{
    using ElementA_ = ElementA;
    using ElementB_ = ElementB;
    using ElementC_ = ElementC;
    using ElementScalar_ = ElementScalar;
    using ElementCompute_= ElementCompute;
    using MinimalElementCompute_= MinimalElementCompute;
    using OpClass_ = OpClass;
    using MathOperatorTag_ = MathOperatorTag;
    static const int ccTarget_ = ccTarget;
    static const int ccTargetMin_ = ccTargetMin;
    static const int ccTargetMax_ = ccTargetMax;

    static lwtensorStatus_t isApplicable(const DeviceProp* deviceProp, const ContractionDescriptorInternal &params)
    {
        const auto typeCompute = params.typeCompute_;
        const auto typeScalar = getScalarType(params.typeC_, typeCompute);
        const int computeCapability = deviceProp->major * 10 + deviceProp->minor;
        const auto computeTypeImpl = lwdaDataTypeToReal(toLwdaDataType<MinimalElementCompute_>());

        return (params.typeA_ == toLwdaDataType<ElementA_>() &&
                params.typeB_ == toLwdaDataType<ElementB_>() &&
                params.typeC_ == toLwdaDataType<ElementC_>() &&
                typeScalar    == toLwdaDataType<ElementScalar_>()&&
                lwdaTypeAsAclwrateAs(computeTypeImpl, computeTypeToLwda(typeCompute, false))&&
                computeCapability >= ccTargetMin_ &&
                computeCapability <= ccTargetMax_)
            ? LWTENSOR_STATUS_SUCCESS : LWTENSOR_STATUS_NOT_SUPPORTED;
    }
};

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
    class IlwokeLwblasLt : public Initializable<1233>
    {
        public:

        struct Params
        {
            lwdaDataType_t typeScalar_;
            lwblasComputeType_t typeCompute_;
            int32_t numBatched_;
            lwblasOperation_t opA_, opB_;

            struct Matrix
            {
                lwdaDataType_t type_;
                uint64_t row_, col_;
                int64_t ld_, ldBatched_;

                void init(lwdaDataType_t type, uint64_t row, uint64_t col, int64_t ld, int64_t ldBatched)
                {
                    type_ = type;
                    row_ = row;
                    col_ = col;
                    ld_ = ld;
                    ldBatched_ = ldBatched;
                }
            } A_, B_, C_, D_;

            void init(lwtensorComputeType_t typeCompute, lwdaDataType_t typeScalar, int32_t numBatched,
                      lwblasOperation_t opA, const lwdaDataType_t typeA, int rowA, int colA, int64_t bdA, int ldA,
                      lwblasOperation_t opB, const lwdaDataType_t typeB, int rowB, int colB, int64_t bdB, int ldB,
                                             const lwdaDataType_t typeC, int rowC, int colC, int64_t bdC, int ldC,
                                                                                             int64_t bdD, int ldD)
            {
                typeScalar_ = typeScalar;
                typeCompute_ = getLwblasComputeType(typeCompute);
                numBatched_ = numBatched;
                opA_ = opA;
                opB_ = opB;

                A_.init(typeA, rowA, colA, ldA, bdA);
                B_.init(typeB, rowB, colB, ldB, bdB);
                C_.init(typeC, rowC, colC, ldC, bdC);
                D_.init(typeC, rowC, colC, ldD, bdD);
            }
        };

        public:

        /**
         * \brief Initializes all lwblasLt descriptors.
         */
        lwtensorStatus_t init(lwtensorComputeType_t typeCompute,
                          lwdaDataType_t typeScalar,
                          int32_t numBatched,
                          lwblasOperation_t transA, const lwdaDataType_t typeA, int rowA, int colA, int64_t bdA, int ldA, uint32_t alignmentRequirementA,
                          lwblasOperation_t transB, const lwdaDataType_t typeB, int rowB, int colB, int64_t bdB, int ldB, uint32_t alignmentRequirementB,
                                                    const lwdaDataType_t typeC, int rowC, int colC, int64_t bdC, int ldC, uint32_t alignmentRequirementC,
                                                                                                    int64_t bdD, int ldD,
                          size_t remainingWorkspace, const int32_t deviceId, const DeviceProp* deviceProp, const int32_t kernel);

        IlwokeLwblasLt();

        /**
         * \brief LwblasLt exelwtion stage.
         * \param[in] useCasD: If true, use the D (non-strided) layout.
         */
        lwtensorStatus_t execute(const lwblasLtHandle_t& handle,
                bool useCasD,
                const void* alpha, const void* A, const void *B,
                const void* beta, void* C, void* workspace, size_t workspaceSize,
                lwdaStream_t stream) const;

        private:
        /**
         * \brief LwblasLt planning stage.
         * \param[in] useCasD: If true, use the D (non-strided) layout.
         * \param[out] algo
         */
        lwtensorStatus_t initAlgo(const lwblasLtHandle_t& handle, bool useCasD,
                uint32_t alignmentA, uint32_t alignmentB, uint32_t alignmentC,
                size_t workspaceSize, lwblasLtMatmulAlgo_t* algo, int32_t kernel) const;

        /**
         * \brief Initialize a matrix descriptor.
         * \param[in] mat
         * \param[out] layout
         */
        lwtensorStatus_t initLayout(const Params::Matrix &mat, lwblasLtMatrixLayout_t &layout) const;

        Params params_;
        lwblasLtMatrixLayout_t lA_, lB_, lC_, lD_;
        lwblasLtMatrixLayoutOpaque_t lAOpaque_, lBOpaque_, lCOpaque_, lDOpaque_;
        lwblasLtMatmulDesc_t mul_;
        lwblasLtMatmulDescOpaque_t mulOpaque_;
        lwblasLtMatmulAlgo_t algoCLt_; ///< algo used if C!=D
        lwblasLtMatmulAlgo_t algoDLt_; ///< algo used if C==D
    };
#endif // LWTENSOR_LWDA_VERSION_MAJOR >= 11

    class ContractionPlan : public Initializable<45>
    {
        private:
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
            IlwokeLwblasLt lwblasLtIlwoke_;
#endif // LWTENSOR_LWDA_VERSION_MAJOR >= 11

        public:
            ContractionPlan() {}
            void init(lwdaDataType_t typeScalar, const int32_t partitionsK);

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
            lwtensorStatus_t initLwblasLt(
                          int32_t numBatched,
                          lwblasOperation_t transA, int rowA, int colA, int64_t bdA, int ldA, uint32_t alignmentRequirementA,
                          lwblasOperation_t transB, int rowB, int colB, int64_t bdB, int ldB, uint32_t alignmentRequirementB,
                                                    int rowC, int colC, int64_t bdC, int ldC, uint32_t alignmentRequirementC,
                                                                        int64_t bdD, int ldD,
                          const int32_t deviceId, const DeviceProp *deviceProp, const int32_t kernel);
#endif

        struct Params
        {
            Params(const Context* ctx,
                   const void* alpha, const void *A, const void *B,
                   const void* beta,  const void *C, void *D,
                   void* workspace, const uint64_t workspaceSize, lwdaStream_t stream)
                : ctx_(ctx), alpha_(alpha), A_(A), B_(B), beta_(beta), C_(C), D_(D),
                  workspace_(workspace), workspaceSize_(workspaceSize), stream_(stream)
            {
            }

            const Context* ctx_;
            const void* alpha_;
            const void *A_;
            const void *B_;
            const void* beta_;
            const void *C_;
            void *D_;
            void* workspace_;
            const uint64_t workspaceSize_;
            lwdaStream_t stream_;
        };

        ContractionDescriptorInternal gettParams_;
        ElementwisePlan planForAlphaIsZero_; ///< Dedicated EW plan for the case alpha == 0
        ElementwisePlan transposePlanA_; ///< Transpose plans used for TTGT or TGETT
        ElementwisePlan transposePlanB_; ///< Transpose plans used for TTGT or TGETT
        ElementwisePlan transposePlanC_; ///< Transpose plans used for TTGT or TGETT
        ReductionParams reductionParams_;
        bool transposeA_;  ///< indicates if A has to be transposed
        bool transposeB_;  ///< indicates if B has to be transposed
        bool transposeC_;  ///< indicates if C has to be transposed
        bool dispatchToTrinary_;
        bool dispatchToReduction_;
        bool swapAB_; ///< indicates if A and B must be swapped
        uint64_t bufferSizeA_; ///< buffer size required for transpose A (for TTGT)
        uint64_t bufferSizeB_; ///< buffer size required for transpose B (for TTGT)
        uint64_t bufferSizeC_; ///< buffer size required for transpose C (for TTGT)
        uint64_t workspaceSize_; ///< available workspace size at the time of creating the plan
        lwdaDataType_t typeScalar_;

        bool useBLAS_; ///< indicates that BLAS should be used to execute
        bool useLwTe_; ///< indicates that LwTe should be used to execute
        int32_t candidateIdx_;
        int32_t containerIdx_;
        ElementwisePlan blasElementwisePlan_; ///< Update plan for C != D if not transposed

        size_t getKey() const { return key_; }
        void setKey(size_t key) { key_ = key; }

        /**
         * Checks if this plan can be cached (lwrrenlty only GEMM-like contractions can be
         * cached--mainly due to how we store the key)
         */
        bool canBeCached() const
        {
            return !dispatchToTrinary_ && !dispatchToReduction_ && gettParams_.isInitialized();
        }

        lwtensorStatus_t operator() (Params &params) const;

        /**
         * Writes information about this plan to dst
         *
         * \param[out] dst This char array will contain the information of this kernel
         * \param[in] sz size of the dst in bytes.
         * \return Returns the number of characters that has been written.
         */
        int info(char* dst, const size_t sz) const;

        bool getRequiresMeasurement() const noexcept
        {
            return requiresMeasurement_;
        }
        void setRequiresMeasurement(const bool requiresMeasurement) noexcept
        {
            requiresMeasurement_ = requiresMeasurement;
        }

        static const int alignmentRequirement_ = 256; // bytes (for workspace)
        private:
            size_t key_; ///< used to cache the plan
            bool requiresMeasurement_; ///< determines if the plan requires a measurement at exelwtion time
    };

    // Check that ContractionPlan fits in lwtensorContractionPlan_t
    static_assert(sizeof(ContractionPlan) <= sizeof(lwtensorContractionPlan_t),
                  "Size of ContractionPlan greater than lwtensorContractionPlan_t");
}
