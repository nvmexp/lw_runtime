#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <lwtensor/internal/contractionDescriptor.h>
#include <lwtensor/internal/utilEx.h>
#include <lwtensor/internal/defines.h>

#include <lwtlass/contraction/device/gett.h>

namespace LWTENSOR_NAMESPACE {

template<typename T, typename U>
T make_coord(const U &from) {
    typename T::Index arr[T::kRank];
    for (int i = 0; i < T::kRank; i++) {
        arr[i] = from[i];
    }
    return T{arr};
}

inline lwtensorStatus_t handleError(lwtlass::Status status) {
  if (status == lwtlass::Status::kSuccess) {
    return LWTENSOR_STATUS_SUCCESS;
  } else if (status == lwtlass::Status::kErrorArchMismatch) {
    RETURN_STATUS(LWTENSOR_STATUS_ARCH_MISMATCH)
  } else if (status == lwtlass::Status::kErrorInsufficientDriver) {
    RETURN_STATUS(LWTENSOR_STATUS_INSUFFICIENT_DRIVER)
  } else {
    RETURN_STATUS(LWTENSOR_STATUS_INTERNAL_ERROR)
  }
}

template<
    typename ElementA_, typename TransformA_, int kElementsPerAccessA_, int kBlockedModesM, bool transA_, bool kStridedLoadsA_,
    typename ElementB_, typename TransformB_, int kElementsPerAccessB_, int kBlockedModesN, bool transB_, bool kStridedLoadsB_,
    typename ElementC,  int kElementsPerAccessC,
    typename ElementScalar, typename ElementAclwmulator,
    typename ThreadblockShape,
    /// Threadblock-level tile size in k-dimension (concept: IntTuple)
    typename ShapeK,
    typename WarpShape, typename InstructionShape,
    typename OpClass, typename ArchTag, int ccTarget,
    typename MathOperatorTag = typename lwtlass::arch::OpMultiplyAdd>
class ContractionLwtlass
{
    public:

        using Params = ContractionDescriptorInternal;
        /*
         * WARNING: This kernel swaps the notion of A/B, m/n
         */

        static constexpr int kMaxModes_ = LWTENSOR_NAMESPACE::kMaxNumModes;
        static constexpr bool transA = transA_ == transB_ ? !transA_ : transA_; // TODO why is this fix necessary
        static constexpr bool transB = transA_ == transB_ ? !transB_ : transB_; // TODO why is this fix necessary

        using ElementA = ElementB_;
        using ElementB = ElementA_;
        static constexpr int kElementsPerAccessA = kElementsPerAccessB_;
        static constexpr int kElementsPerAccessB = kElementsPerAccessA_;
        static constexpr bool kStridedLoadsA = kStridedLoadsB_;
        static constexpr bool kStridedLoadsB = kStridedLoadsA_;

        using TransformA = TransformB_;
        using TransformB = TransformA_;

        using ElementOutput = ElementC;
        using LayoutAclwmulator = lwtlass::layout::RowMajor;
        // this value could be changed to facilitate non-vectorized loads too

        using Gett = typename lwtlass::contraction::device::Gett<
            ElementA, kElementsPerAccessA, transA, kStridedLoadsA,
            ElementB, kElementsPerAccessB, transB, kStridedLoadsB,
            ElementOutput, kElementsPerAccessC,
            ElementScalar,
            ElementAclwmulator,
            kBlockedModesM, kBlockedModesN,
            OpClass, ArchTag,
            ThreadblockShape, ShapeK, WarpShape, InstructionShape, ccTarget,
            kMaxModes_,
            TransformA, TransformB, MathOperatorTag
                >;

        static_assert( Gett::kMajorVersion == 0 &&
                       Gett::kMinorVersion == 3 &&
                       Gett::kPatchVersion == 0, "Wrong LWTLASS version.");

        void init()
        {
            gett_.init();
        }

        inline int getMaxActiveBlocksPerMultiprocessor() const { return gett_.getMaxActiveBlocksPerMultiprocessor(); }

        inline int getNumRegisters() const { return gett_.getNumRegisters(); }

        inline size_t getLocalMemorySize() const { return gett_.getLocalMemorySize(); }

        typename Gett::GettKernel::Params getParams(
                                 const Params &params,
                                 const ElementScalar alpha, ElementA_* A, ElementB_ *B,
                                 const ElementScalar beta, ElementC *C, ElementC *D,
                                 void* workspace, size_t workspace_size) const
        {
            using TensorCoord = typename Gett::TensorCoord;
            using TensorStrideCoord = typename Gett::TensorStrideCoord;
            // IDEA we could reduce const memory by 2x by using a single coord for m and n strides

            // NOTICE: useA is ilwerse here to implicitly swap A and B, since lwTENSOR expects ColumnMajor C, but LWTLASS only provided rowMajor C for now TODO
            int numModesM = params.nmodeN;
            int numModesN = params.nmodeM;
            int numModesK = params.nmodeK;
            int numModesL = params.nmodeL;

            assert(numModesK <= ShapeK::kRank);

            using TensorStrideCoordK = typename Gett::GettKernel::Mma::IteratorA::CoordStridesContracted;
            using TensorCoordK = typename Gett::GettKernel::Mma::IteratorA::CoordExtentsContracted;

            TensorStrideCoordK strideAk;
            TensorStrideCoordK strideBk;
            TensorCoordK extentK;
            int total_extent_k = 1;
            int blocked_extent_m = 1;
            int blocked_extent_n = 1;
            for(int i=0; i < numModesK; ++i){
                strideAk[i] = params.strideBk[i]; // TODO swap, once we have a generic epilog
                strideBk[i] = params.strideAk[i]; // TODO swap, once we have a generic epilog
                extentK[i] = params.extentK[i];
                total_extent_k *= extentK[i];
            }
            for(int i=numModesK; i < TensorCoordK::kRank; ++i){
                strideAk[i] = 0; // this is a trick to use a compile-time loop count within the iterator (~+20% perf)
                strideBk[i] = 0;
                extentK[i] = 1;
            }

            TensorCoord extentM;
            TensorStrideCoord strideAm;
            TensorStrideCoord strideCm;
            for(int i=0; i < numModesM; ++i)
            {
                strideAm[i] = params.strideBn[i]; // TODO
                strideCm[i] = params.strideCn[i]; // TODO
                extentM[i] = params.extentN[i]; // TODO
                if( i < kBlockedModesM )
                    blocked_extent_m *= extentM[i];
            }
            // add artificial mode to improve kernel code
            for(int i=numModesM; i < TensorCoord::kRank; ++i)
            {
                strideAm[i] = 0;
                strideCm[i] = 0;
                extentM[i] = 1;
            }
            numModesM = (numModesM < kBlockedModesM) ? kBlockedModesM : numModesM;

            TensorCoord extentN;
            TensorStrideCoord strideBn;
            TensorStrideCoord strideCn;
            for(int i=0; i < numModesN; ++i){
                strideBn[i] = params.strideAm[i]; // TODO
                strideCn[i] = params.strideCm[i]; // TODO
                extentN[i] = params.extentM[i]; // TODO
                if( i < kBlockedModesN )
                    blocked_extent_n *= extentN[i];
            }
            for(int i=numModesN; i < TensorCoord::kRank; ++i)
            {
                strideBn[i] = 0;
                strideCn[i] = 0;
                extentN[i] = 1;
            }
            numModesN = (numModesN < kBlockedModesN) ? kBlockedModesN : numModesN;

            TensorStrideCoord strideAl;
            TensorStrideCoord strideBl;
            TensorStrideCoord strideCl;
            TensorCoord extentL;
            for(int i=0; i < numModesL; ++i)
            {
                extentL[i] = params.extentL[i];
                strideAl[i] = params.strideBl[i];// TODO
                strideBl[i] = params.strideAl[i];// TODO
                strideCl[i] = params.strideCl[i];
            }
            for(int i=numModesL; i < TensorCoord::kRank; ++i)
            {
                strideAl[i] = 0;
                strideBl[i] = 0;
                strideCl[i] = 0;
                extentL[i] = 1;
            }

            //
            // Launch the kernel
            //

            const bool useNaiveEpilogue = (params.extentM[1] > 1) || (params.extentN[1] > 1) || params.forceUseNaiveEpilogue_;

            static_assert(kBlockedModesM == kBlockedModesN, "invalid blocking");

            using AffineLayout = lwtlass::layout::AffineRankN<kBlockedModesM + kBlockedModesN>;
            using AffineTensorView = lwtlass::TensorView<ElementC, AffineLayout>;

            auto stride_cm = make_coord<lwtlass::Coord<kBlockedModesM, int64_t>>(strideCm);
            auto stride_cn = make_coord<lwtlass::Coord<kBlockedModesN, int64_t>>(strideCn);

            typename AffineLayout::TensorCoord::Index affine_extent_arr[kBlockedModesM + kBlockedModesN];
            for (int i = 0; i < kBlockedModesM; i++) {
                affine_extent_arr[i] = extentM[i];
            }
            for (int i = 0; i < kBlockedModesN; i++) {
                affine_extent_arr[i + kBlockedModesM] = extentN[i];
            }

            typename AffineLayout::TensorCoord affine_extent(affine_extent_arr);
            AffineLayout affine_layout(stride_cm, stride_cn);


            using LinearLayout = lwtlass::layout::RowMajor;

            LinearLayout tensorC_linear_layout = LinearLayout::packed({blocked_extent_m, blocked_extent_n});
            LinearLayout tensorD_linear_layout = LinearLayout::packed({blocked_extent_m, blocked_extent_n});
            tensorC_linear_layout.stride(0) = strideCm[0];
            tensorD_linear_layout.stride(0) = strideCm[0];

//            typename Gett::GettKernel::Params paramsLwtlass{
//                {gemm_k_iterations_inner, gemm_k_iterations_outer}, // mma
//                {blocked_extent_m, blocked_extent_n, total_extent_k},
//                {extentK, strideAk, numModesK, extentM, strideAm, numModesM}, // params_A
//                B,
//                {extentK, strideBk, numModesK, extentN, strideBn, numModesN}, // params_B
//                A,
//                affine_extent,
//                affine_layout,
//                { C, tensorC_linear_layout },
//                { D, tensorD_linear_layout },
//                {alpha, beta},
//                strideCm,
//                strideCn, 
//                strideAl, 
//                strideBl, 
//                strideCl, 
//                extentL, 
//                numModesL,
//                useNaiveEpilogue
//            };
            
//            using OutputLayout = lwtlass::layout::RowMajor;
//            using OutputType = lwtlass::TensorRef<ElementC, OutputLayout>;
//            auto layoutC = OutputLayout::packed({blocked_extent_m, blocked_extent_n});
//            layoutC.stride(0) = strideCm[0];
//            OutputType tensorC(C, layoutC);
//            OutputType tensorD(D, layoutC);
//            static_assert(kBlockedModesM == 2 && kBlockedModesN == 2, "invalid blocking");
//
//
//            using AffineLayout = lwtlass::layout::AffineRankN<kBlockedModesM + kBlockedModesN>;
//            using AffineTensorView = lwtlass::TensorView<ElementC, AffineLayout>;
//
//            lwtlass::Coord<kBlockedModesM, int64_t> stride_cm({(int64_t)strideCm[0], (int64_t)strideCm[1]});
//            lwtlass::Coord<kBlockedModesN, int64_t> stride_cn({(int64_t)strideCn[0], (int64_t)strideCn[1]});
//
//            typename AffineLayout::TensorCoord affine_extent({extentM[0], extentM[1], extentN[0], extentN[1]});
//            AffineLayout affine_layout(stride_cm, stride_cn);
//
            using TensorCoordBlockedM = typename Gett::GettKernel::Mma::IteratorA::CoordExtentsFree;
            using TensorStrideCoordBlockedM = typename Gett::GettKernel::Mma::IteratorA::CoordStridesFree;
            using TensorCoordBlockedN = typename Gett::GettKernel::Mma::IteratorB::CoordExtentsFree;
            using TensorStrideCoordBlockedN = typename Gett::GettKernel::Mma::IteratorB::CoordStridesFree;
            using ParamsA = typename Gett::GettKernel::Mma::IteratorA::Params;
            using ParamsB = typename Gett::GettKernel::Mma::IteratorB::Params;
            using ParamsContracted = typename Gett::GettKernel::Mma::IteratorB::ParamsContracted;
//
//            typename OutputLayout::TensorCoord extentC({extentM[0], extentM[1], extentN[0], extentN[1]});
//            OutputLayout layoutC(stride_cm, stride_cn);
//            OutputType tensorC(C, layoutC, extentC);
//            OutputType tensorD(D, layoutC, extentC);
//
//            typename Gett::GettKernel::Mma::Params mma_params{gemm_k_iterations_inner, gemm_k_iterations_outer};
            lwtlass::gemm::GemmCoord gemm_problem_size{blocked_extent_m, blocked_extent_n, total_extent_k};

            while (reinterpret_cast<intptr_t>(workspace) % sizeof(int) != 0) {
                workspace = static_cast<void*>(static_cast<char*>(workspace) + 1);
                if (workspace_size > 0) {
                    workspace_size -= 1;
                }
            }

            if (workspace == nullptr && workspace_size > 0) {
                workspace_size -= std::min(workspace_size, sizeof(int));
            }

            for (int partitions = params.partitions_; partitions > 1; partitions--)
            {
                typename Gett::GettKernel::Params paramsLwtlass(
                    gemm_problem_size,
                    /*params_A*/ ParamsA{ make_coord<TensorCoordBlockedM>(extentM),
                                   make_coord<TensorStrideCoordBlockedM>(strideAm),
                                   extentK, strideAk,
                               },
                    B,
                    /*params_B*/ ParamsB{ make_coord<TensorCoordBlockedN>(extentN),
                                   make_coord<TensorStrideCoordBlockedN>(strideBn),
                                   extentK, strideBk
                    },
                    A,
                    ParamsContracted{ extentK },
                    affine_extent,
                    affine_layout,
                    { C, tensorC_linear_layout },
                    { D, tensorD_linear_layout },
                    {alpha, beta},
                    strideAm,
                    strideCm,
                    strideBn,
                    strideCn,
                    strideAl,
                    strideBl,
                    strideCl,
                    extentM,
                    extentN,
                    extentK,
                    extentL,
                    numModesM,
                    numModesN,
                    numModesK,
                    numModesL,
                    useNaiveEpilogue,
                    partitions);

                auto required_workspace_size = gett_.get_workspace_size(paramsLwtlass);
                if (required_workspace_size > workspace_size)
                {
                    continue;
                }
    
                paramsLwtlass.semaphore = static_cast<int*>(workspace);
                return paramsLwtlass;
            }

            typename Gett::GettKernel::Params paramsLwtlass(
                gemm_problem_size,
                /*params_A*/ ParamsA{ make_coord<TensorCoordBlockedM>(extentM),
                               make_coord<TensorStrideCoordBlockedM>(strideAm),
                               extentK, strideAk,
                           },
                B,
                /*params_B*/ ParamsB{ make_coord<TensorCoordBlockedN>(extentN),
                               make_coord<TensorStrideCoordBlockedN>(strideBn),
                               extentK, strideBk
                },
                A,
                ParamsContracted{ extentK },
                affine_extent,
                affine_layout,
                { C, tensorC_linear_layout },
                { D, tensorD_linear_layout },
                {alpha, beta},
                strideAm,
                strideCm,
                strideBn,
                strideCn,
                strideAl,
                strideBl,
                strideCl,
                extentM,
                extentN,
                extentK,
                extentL,
                numModesM,
                numModesN,
                numModesK,
                numModesL,
                useNaiveEpilogue,
                1);
            return paramsLwtlass;
        }


        inline int getGridSize(const Params& params, size_t workspace_size) const
        {
            auto ret = getParams(params, 1, nullptr, nullptr, 1, nullptr, nullptr, nullptr, workspace_size);
            return ret.gridSize;
        }

        lwtensorStatus_t operator()(const Params &params,
                                 const ElementScalar *alpha_, ElementA_* A, ElementB_ *B,
                                 const ElementScalar *beta_, ElementC *C, ElementC *D,
                                 void* workspace, size_t workspace_size,
                                 lwdaStream_t stream) const
        {
            typename Gett::GettKernel::Params paramsLwtlass = getParams(params, *alpha_, A, B, *beta_, C, D, workspace, workspace_size);
            HANDLE_ERROR(gett_.run(paramsLwtlass, stream));
            return LWTENSOR_STATUS_SUCCESS;
        }

        Gett gett_;
};
}
#pragma GCC diagnostic pop
