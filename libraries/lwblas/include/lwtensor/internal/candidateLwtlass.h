#pragma once

#include <stdio.h>
#include <lwtensor/types.h>
#include <lwtensor/internal/typesEx.h>
#include <lwtensor/internal/contractionDescriptor.h>
#include <lwtensor/internal/tensorContractionLwtlass.h>
#include <lwtensor/internal/context.h>
#include <lwtensor/internal/featuresUtils.h>

#include <lwtensor/internal/defines.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

namespace LWTENSOR_NAMESPACE
{

    template<typename UnaryTransform>
    inline lwtensorOperator_t lwtlassTransformToOperator();
    template<>
    inline lwtensorOperator_t lwtlassTransformToOperator<lwtlass::transform::thread::UnaryTransform::Identity>(){ return LWTENSOR_OP_IDENTITY; }
    template<>
    inline lwtensorOperator_t lwtlassTransformToOperator<lwtlass::transform::thread::UnaryTransform::Conjugate>(){ return LWTENSOR_OP_CONJ; }

    template<typename T> struct LwdaTypeToChar;
    template<> struct LwdaTypeToChar<int32_t> { static const char kValue = 'i'; };
    template<> struct LwdaTypeToChar<uint32_t> { static const char kValue = 'u'; };
    template<> struct LwdaTypeToChar<lwtlass::half_t> { static const char kValue = 'h'; };
    template<> struct LwdaTypeToChar<half> { static const char kValue = 'h'; };
    template<> struct LwdaTypeToChar<float> { static const char kValue = 's'; };
    template<> struct LwdaTypeToChar<double> { static const char kValue = 'd'; };
    template<> struct LwdaTypeToChar<lwtlass::complex<lwtlass::tfloat32_t>> { static const char kValue = 'r'; };
    template<> struct LwdaTypeToChar<lwtlass::complex<float>> { static const char kValue = 'c'; };
    template<> struct LwdaTypeToChar<lwComplex> { static const char kValue = 'c'; };
    template<> struct LwdaTypeToChar<lwtlass::complex<double>> { static const char kValue = 'z'; };
    template<> struct LwdaTypeToChar<lwDoubleComplex> { static const char kValue = 'z'; };
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
    template<> struct LwdaTypeToChar<lwtlass::bfloat16_t> { static const char kValue = 'b'; };
    template<> struct LwdaTypeToChar<BFloat16> { static const char kValue = 'b'; };
#endif
    template<> struct LwdaTypeToChar<lwtlass::tfloat32_t> { static const char kValue = 't'; };

    template<typename T>
    struct LwtlassArchToInt;
    template<> struct LwtlassArchToInt<lwtlass::arch::Sm50> { static const int kValue = 0; };
    template<> struct LwtlassArchToInt<lwtlass::arch::Sm70> { static const int kValue = 1; };
    template<> struct LwtlassArchToInt<lwtlass::arch::Sm75> { static const int kValue = 2; };
    template<> struct LwtlassArchToInt<lwtlass::arch::Sm80> { static const int kValue = 3; };

    template<typename T>
    struct LwtlassMathTagToInt;
    template<> struct LwtlassMathTagToInt<lwtlass::arch::OpMultiplyAdd> { static const int kValue = 0; };
    template<> struct LwtlassMathTagToInt<lwtlass::arch::OpMultiplyAddComplex> { static const int kValue = 1; };
    template<> struct LwtlassMathTagToInt<lwtlass::arch::OpMultiplyAddFastF16> { static const int kValue = 2; };
    template<> struct LwtlassMathTagToInt<lwtlass::arch::OpMultiplyAddFastBF16> { static const int kValue = 3; };

    template<typename T>
    struct LwtlassOpClassToInt;
    template<> struct LwtlassOpClassToInt<lwtlass::arch::OpClassTensorOp> { static const int kValue = 0; };
    template<> struct LwtlassOpClassToInt<lwtlass::arch::OpClassSimt> { static const int kValue = 1; };

    template<
        typename Traits,
        typename TransformA, int kElementsPerAccessA, int kBlockedModesM, bool transA, bool kStridedLoadsA,
        typename TransformB, int kElementsPerAccessB, int kBlockedModesN, bool transB, bool kStridedLoadsB,
                             int kElementsPerAccessC,
        typename ThreadblockShape,
        /// Threadblock-level tile size in k-dimension (concept: IntTuple)
        typename ShapeK,
        typename WarpShape, typename InstructionShape,
        typename ArchTag, int kNumThreads,
        int kLocalMemoryUsage=0,
        int kWaitSchedule=0,
        int kAvgLDS=0,
        int kAvgLDG=0,
        int kAvgAntidep=0>
    class CandidateLwtlass : public CandidateTyped<ContractionDescriptorInternal, CandidateInfoLwtlass>
    {
        /*
         * This class represents all the (engineered) features w.r.t to a specific CandidateLwtlass and a
         * specific tensor contraction
         */
        public :
            static constexpr int kMaxModes_ = LWTENSOR_NAMESPACE::kMaxNumModes;

            using ElementA_ = typename Traits::ElementA_;
            using TransformA_ = TransformA;
            static const  int kElementsPerAccessA_ = kElementsPerAccessA;
            static const  int kBlockedModesM_ = kBlockedModesM;
            static const  bool transA_ = transA;
            static const  bool kStridedLoadsA_ = kStridedLoadsA;

            using ElementB_ = typename Traits::ElementB_;
            using TransformB_ = TransformB;
            static const int kElementsPerAccessB_ = kElementsPerAccessB;
            static const int kBlockedModesN_ = kBlockedModesN;
            static const bool transB_ = transB;
            static const bool kStridedLoadsB_ = kStridedLoadsB;

            using ElementC_ = typename Traits::ElementC_;
            static const int kElementsPerAccessC_ = kElementsPerAccessC;

            using ElementScalar_ = typename Traits::ElementScalar_;
            using ElementAclwmulator_ = typename Traits::ElementCompute_;

            using ThreadblockShape_ = ThreadblockShape;
            using ShapeK_ = ShapeK;
            using WarpShape_ = WarpShape;
            using InstructionShape_ = InstructionShape;
            using OpClass_ = typename Traits::OpClass_;
            using ArchTag_ = ArchTag;
            using MathOperatorTag_ = typename Traits::MathOperatorTag_;
            static const int ccTarget_ = Traits::ccTarget_;
            static const int kNumThreads_ = kNumThreads;

            static const int kLocalMemoryUsage_ = kLocalMemoryUsage;
            static const int kWaitSchedule_ = kWaitSchedule;
            static const int kAvgLDS_ = kAvgLDS;
            static const int kAvgLDG_ = kAvgLDG;
            static const int kAvgAntidep_ = kAvgAntidep;

            CandidateLwtlass() {}

            void init()
            {
                contraction_.init();
                this->setInitialized();
            }

            using Contraction = typename ::LWTENSOR_NAMESPACE::ContractionLwtlass<
                ElementA_, TransformA_, kElementsPerAccessA_, kBlockedModesM_, transA_, kStridedLoadsA_,
                ElementB_, TransformB_, kElementsPerAccessB_, kBlockedModesN_, transB_, kStridedLoadsB_,
                ElementC_, kElementsPerAccessC_, ElementScalar_, ElementAclwmulator_,
                ThreadblockShape_, ShapeK_, WarpShape_, InstructionShape_, OpClass_, ArchTag_, ccTarget_, MathOperatorTag_ >;
            static_assert(std::is_same<typename Contraction::Params, Params>::value, "The params data structure must be identical");

//            virtual void computeFeatures(const Params &params,
//                                         const DeviceProp* deviceProp,
//                                         Features* featuresAlias) const
//            {
//                auto features = static_cast<FeaturesLwtlass*>(featuresAlias);
//
//                features->init(ThreadblockShape_::kM,
//                               ThreadblockShape_::kN, kNumThreads,
//                               contraction_.getMaxActiveBlocksPerMultiprocessor(),
//                               kElementsPerAccessA_ + kElementsPerAccessB_ + kElementsPerAccessC_);
// 
//                features->callwlate(params, deviceProp);
//            }

            CandidateInfoLwtlass getCandidateInfo() const
            {
                CandidateInfoLwtlass params;
                params.threadblockM = ThreadblockShape_::kM;
                params.threadblockN = ThreadblockShape_::kN;
                params.threadblockK = ThreadblockShape_::kK;
                params.shapeK0 = lwtlass::contraction::At<0,ShapeK>::value;
                params.shapeK1 = lwtlass::contraction::Count<ShapeK>::value > 1 ? lwtlass::contraction::At<1,ShapeK>::value : 1;
                params.numModesContracted = ShapeK::kRank;
                params.numBlockedModesContracted = lwtlass::contraction::Count<ShapeK>::value;
                params.warpM = WarpShape_::kM;
                params.warpN = WarpShape_::kN;
                params.warpK = WarpShape_::kK;
                params.elementsPerAccessA = kElementsPerAccessA_;
                params.elementsPerAccessB = kElementsPerAccessB_;
                params.elementsPerAccessC = kElementsPerAccessC_;
                params.blockedModesM = kBlockedModesM_;
                params.blockedModesN = kBlockedModesN_;
                params.numThreads = kNumThreads;
                params.maxCTAsPerSM = contraction_.getMaxActiveBlocksPerMultiprocessor();
                params.localMemoryUsage = kLocalMemoryUsage_;
                params.waitSchedule = kWaitSchedule_;
                params.avgLDS = kAvgLDS_;
                params.avgLDG = kAvgLDG_;
                params.avgAntidep = kAvgAntidep_;
                return params;
            }

            virtual bool isApplicable(const Context *ctx,
                    const Params &params,
                    const size_t workspaceSize) const
            {
                (void) workspaceSize; // avoid compiler warning (this variable is not needed)

                constexpr auto alignmentRequirementA = kElementsPerAccessA_ * sizeof(ElementA_);
                constexpr auto alignmentRequirementB = kElementsPerAccessB_ * sizeof(ElementB_);
                constexpr auto alignmentRequirementC = kElementsPerAccessC_ * sizeof(ElementC_);
                const DeviceProp* deviceProp = ctx->getDeviceProp();
                if (contraction_.getMaxActiveBlocksPerMultiprocessor() == 0)
                {
                    return false;
                }
                if (static_cast<size_t>(deviceProp->sharedMemPerMultiprocessor) < sizeof(typename Contraction::Gett::GettKernel::SharedStorage))
                {
                    return false;
                }
                return (params.transA_ == transA_) &&
                       (params.transB_ == transB_) &&
                       (!params.contiguousModeIsBatchedA_) &&
                       (!params.contiguousModeIsBatchedB_) &&
                       (params.opA_ == lwtlassTransformToOperator<TransformA>()) &&
                       (params.opB_ == lwtlassTransformToOperator<TransformB>()) &&
                       (params.opC_ == LWTENSOR_OP_IDENTITY) &&
                       (params.alignmentRequirementA_ % alignmentRequirementA == 0) &&
                       (params.alignmentRequirementB_ % alignmentRequirementB == 0) &&
                       (params.alignmentRequirementC_ % alignmentRequirementC == 0) &&
                       (params.nmodeM <= kMaxModes_) &&
                       (params.nmodeN <= kMaxModes_) &&
                       (params.nmodeL <= kMaxModes_) &&
                       (params.nmodeK <= ShapeK_::kRank);
            }

            virtual lwtensorStatus_t operator() (const Context *ctx,
                            const Params &params,
                            const void* alpha, const void *A,
                            const void* unused, const void *B,
                            const void* beta,  const void *C, void *D,
                            void* workspace, uint64_t workspaceSize, lwdaStream_t stream) const
            {
                (void) unused;
                return contraction_(params,
                            (ElementScalar_*) alpha, (ElementA_*)A, (ElementB_*)B,
                            (ElementScalar_*) beta, (ElementC_*) C, (ElementC_*)D, workspace, workspaceSize, stream);
            }

            virtual void print() const
            {
                printf("LwtlassCandidate: %d x %d, %d %d %d, %d %d, %d %d\n", ThreadblockShape_::kM, ThreadblockShape_::kN,
                        kElementsPerAccessA_, kElementsPerAccessB_, kElementsPerAccessC_,
                        kStridedLoadsA_, kStridedLoadsB_, transA_, transB_);
            }
            virtual int info(char* dst, size_t sz) const
            {
                constexpr int kBufferSize = 32;
                char strShapeK[kBufferSize];
                lwtlass::contraction::Tuple<ShapeK_> shapeK;
                int pos = 0;
                for(int i=0; i < ShapeK_::kRank; ++i){
                    pos += sprintf(strShapeK+pos, "%d,",shapeK[i]);
                }
                assert(pos < kBufferSize);
                strShapeK[pos-1] = '\0'; // replace last ','

                return snprintf(dst, sz, "kernel:tb:%d,%d,%d;k:%s;w:%d,%d,%d;is:%d,%d,%d;a:%d,%d,%d;s:%d,%d;t:%d,%d;bf:%d,%d;op:%d,%d;cc:%d,%d,%d;ar:%d;fm:%d;oc:%d;tp:%c,%c,%c,%c,%c;reg:%d;lmem:%d;ac:%d;wa:%d;ls:%d;lg:%d;la:%d;",
                        ThreadblockShape_::kM, ThreadblockShape_::kN, ThreadblockShape_::kK,
                        strShapeK,
                        WarpShape_::kM, WarpShape_::kN, WarpShape_::kK,
                        InstructionShape_::kM, InstructionShape_::kN, InstructionShape_::kK,
                        kElementsPerAccessA_, kElementsPerAccessB_, kElementsPerAccessC_,
                        kStridedLoadsA_, kStridedLoadsB_,
                        transA_, transB_,
                        kBlockedModesM_, kBlockedModesN_,
                        lwtlassTransformToOperator<TransformA>(), lwtlassTransformToOperator<TransformB>(),
                        Traits::ccTarget_, Traits::ccTargetMin_, Traits::ccTargetMax_,
                        LwtlassArchToInt<ArchTag_>::kValue,
                        LwtlassMathTagToInt<MathOperatorTag_>::kValue,
                        LwtlassOpClassToInt<OpClass_>::kValue,
                        LwdaTypeToChar<typename Traits::ElementA_>::kValue,
                        LwdaTypeToChar<typename Traits::ElementB_>::kValue,
                        LwdaTypeToChar<typename Traits::ElementC_>::kValue,
                        LwdaTypeToChar<typename Traits::ElementScalar_>::kValue,
                        LwdaTypeToChar<typename Traits::ElementCompute_>::kValue,
                        contraction_.getNumRegisters(),
                        kLocalMemoryUsage_,
                        contraction_.getMaxActiveBlocksPerMultiprocessor(),
                        kWaitSchedule_,
                        kAvgLDS_,
                        kAvgLDG_,
                        kAvgAntidep_);
            }

            virtual int infoWithParam(const void* rawParam, char* dst, size_t sz) const
            {
                const ContractionPlan* param = (const ContractionPlan*) rawParam;
                return snprintf(dst, sz, "gs:%d;", contraction_.getGridSize(param->gettParams_, param->workspaceSize_));
            }

#ifdef LWTENSOR_SPLIT_K_SWEEP
            virtual int getNumThreadblocks(const void*rawParam) const
            {
                const ContractionPlan* param = (const ContractionPlan*) rawParam;
                return contraction_.getGridSize(param->gettParams_, param->workspaceSize_);
            }
#endif
        private:
            //     int nCTAs = 1;
            //     constexpr int MC = ThreadblockShape_::kM;
            //     constexpr int NC = ThreadblockShape_::kN;

            //     // blocked modes
            //     int m = 1;
            //     for ( int i = 0; i < kBlockedModesM_ && i < params.nmodeM; i ++ )
            //     {
            //         m *= params.extentM[i];
            //     }
            //     nCTAs *= (m + MC - 1) / MC;
            //     // unblocked modes
            //     for ( int i = kBlockedModesM_; i < params.nmodeM; i ++ )
            //     {
            //         nCTAs *= params.extentM[ i ];
            //     }

            //     // blocked modes
            //     int n = 1;
            //     for ( int i = 0; i < kBlockedModesN_ && i < params.nmodeN; i ++ )
            //     {
            //         n *= params.extentN[i];
            //     }
            //     nCTAs *= (n + NC - 1) / NC;
            //     // unblocked modes
            //     for ( int i = kBlockedModesN_; i < params.nmodeN; i ++ )
            //     {
            //         nCTAs *= params.extentN[ i ];
            //     }

            //     for ( int i = 0; i < params.nmodeL; ++ i )
            //     {
            //         nCTAs *= params.extentL[ i ];
            //     }

            //     return nCTAs;
            // }

            // extent is blocked using a blocking of 'blocking'
            float getUtilization(const extent_type extent, const extent_type blocking) const
            {
                const float totalwork = std::max(blocking, ((extent + blocking - 1) / blocking) * blocking);
                return ((float) extent) / totalwork; // usefulwork / totalwork
            }

            Contraction contraction_;
    };
}

#pragma GCC diagnostic pop
