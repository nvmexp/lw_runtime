#pragma once

#include <lwtensor/internal/elementwise.h>
#include <lwtensor/internal/typesEx.h>
#include <lwtensor/internal/defines.h>
#include <lwtensor/internal/heuristicEW.h>

namespace LWTENSOR_NAMESPACE
{

    template<typename Traits, // specialization of ElementwiseTraits
             typename Config>
    class CandidateEW : public CandidateTyped<ElementwiseParameters, CandidateInfoEW>
    {

        using TypeA = typename Traits::TypeA_;
        using TypeB = typename Traits::TypeB_;
        using TypeC = typename Traits::TypeC_;
        using TypeCompute = typename Traits::TypeCompute_;

    public:

        void init() override
        {
            const void* ptr = lookupElementwiseKernel<Config, TypeA, TypeB, TypeC, TypeCompute>();
            // Similar to with lwtlass GETT, this implementation is only correct if there's one type of device in the machine
            auto ret = lwdaOclwpancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM_, ptr, Config::NUM_THREADS, 0);
            if (ret != lwdaSuccess)
            {
                lwdaGetLastError(); // mask errors from lwdaOclwpancyMaxActiveBlocksPerMultiprocessor
            }
            this->setInitialized();
        }

        /// exelwtes the tensor contraction
        lwtensorStatus_t operator() (const Context *ctx,
                const Params &params,
                const void* alpha, const void *A,
                const void* beta,  const void *B,
                const void* gamma, const void *C, void *D,
                void* workspace, uint64_t workspaceSize, lwdaStream_t stream) const override
        {
            (void)workspace; //unused
            (void)workspaceSize; //unused

            // optimization for stride-1 cases that get mapped to 2D kernel: This
            // ensures that loads of A are still coalesced (see elementwise.h)
            const bool isSquare = Config::BLOCKING[0] == Config::BLOCKING[1];
            const bool noTransposedReadForAB = isSquare && params.strideA_[0] == 1; // we'll swap strideA/B[0] with strideA/B[1] inside of the kernel

            const bool isVectorizedA = checkVectorizable(params.nmodeC_, A, sizeof(TypeA), params.strideA_, noTransposedReadForAB ? 0 : 1);
            const bool isVectorizedB = checkVectorizable(params.nmodeC_, B, sizeof(TypeB), params.strideB_, noTransposedReadForAB ? 0 : 1);
            const bool isVectorizedC = checkVectorizable(params.nmodeC_, C, sizeof(TypeC), params.strideC_, 0);
            const bool isVectorizedD = checkVectorizable(params.nmodeC_, D, sizeof(TypeC), params.strideC_, 0);

            auto zero = lwGet<TypeCompute>(0);
            alpha = (alpha == nullptr) ? &zero : alpha;
            beta  = (beta  == nullptr) ? &zero : beta;
            gamma = (gamma == nullptr) ? &zero : gamma;

            launchElementwise<Config, TypeA, TypeB, TypeC, TypeCompute>(
                ctx, params, numBlocksPerSM_,
                *(const TypeCompute*)alpha, (const TypeA *)A, isVectorizedA,
                *(const TypeCompute*)beta,  (const TypeB *)B, isVectorizedB,
                *(const TypeCompute*)gamma, (const TypeC *)C, isVectorizedC,
                                                  (TypeC *)D, isVectorizedD, stream);

            HANDLE_ERROR(lwdaGetLastError());
            RETURN_STATUS(LWTENSOR_STATUS_SUCCESS);
        }

        virtual CandidateInfoEW getCandidateInfo() const
        {
            CandidateInfoEW params;
            params.vectorWidth = Config::VEC;
            params.ndimTile = Config::NDIM_TILE;
            params.numThreads = Config::NUM_THREADS;
            params.opPack = Config::OpPack_::toElementwiseOpPack();
            for(int i=0; i < 3; ++i)
            {
                params.blocking[i] = Config::BLOCKING[i];
            }
            return params;
        }

//         /**
//          * Compute the features (e.g., estimated runtime) of this candidate w.r.t. the provided parameters (i.e., a specific contraction)
//          * \param[out] features
//          */
//         void computeFeatures(const Params &params,
//                 const DeviceProp* deviceProp,
//                 Features* featuresAlias) const
//         {
//             auto features = static_cast<FeaturesEW*>(featuresAlias);
//             features->init(Config::VEC, Config::NDIM_TILE, Config::NUM_THREADS, Config::BLOCKING, Config::OpPack_::toElementwiseOpPack());
//             features->callwlate(params, deviceProp);
//         }

        /// checks if the candidate is applicable to the problem
        bool isApplicable(const Context *ctx,
                const Params &params,
                const size_t workspaceSize) const
        {
            (void) workspaceSize; // unused
            (void) ctx; // not used for now, but maybe later

            constexpr bool isStrideOneKernel = Config::NDIM_TILE == 1;

            constexpr bool usesGenericOps = std::is_same<OpPackGeneric, typename Config::OpPack_>::value;
            const bool paramsMatchOps = typename Config::OpPack_() == params.opPack_;
            const bool hasFittingOps = (usesGenericOps && (! paramsMatchOps)) || ((! usesGenericOps) && paramsMatchOps);

            return ((!isStrideOneKernel) || ((params.nmodeC_ ==1) || params.isStrideOne_)) && // 1D is only applicable if ndim ==1 or isStrideOne
                   ((isStrideOneKernel) || params.nmodeC_ > 1) && // 2D is only applicable if ndim > 1
                   hasFittingOps &&
                   (params.nmodeC_ >= Config::NDIM_TILE) &&
                   (Config::TRANSPOSE || params.isStrideOne_ || (isStrideOneKernel && params.nmodeC_ == 1));
        }

        /// prints kernel infos to std::cout
        void print() const
        {
            char buffer[1024];
            info(buffer, 1024);
            printf("%s", buffer);
        }

        /// prints kernel infos to dst
        int info(char* dst, size_t sz) const
        {
            constexpr bool usesGenericOps = std::is_same<OpPackGeneric, typename Config::OpPack_>::value;
            return snprintf(dst, sz, "kernel:%d;b:%d,%d,%d;op:%d;v:%d;t:%d;cc:%d;", Config::NDIM_TILE, Config::BLOCKING[0], Config::BLOCKING[1], Config::BLOCKING[2], usesGenericOps, Config::VEC, Config::NUM_THREADS, Traits::targetCC_);
        }

        private:

        static bool checkVectorizable(const int nmodeC, const void* ptr, int typeSize, const stride_type* stride, int vectorizedModeId)
        {
            bool result = ( (uint64_t)ptr % (typeSize * Config::VEC) ) == 0;
            bool isStrideOneKernel = Config::NDIM_TILE == 1;
            vectorizedModeId = isStrideOneKernel ? 0 : vectorizedModeId;

            for(int i=0; i < (int)nmodeC; ++i)
            {
                result = result && ( i == vectorizedModeId ? stride[i] == 1 : stride[i] % Config::VEC == 0 );
            }

            return result;
        }

        int numBlocksPerSM_ = 1;
    };

}
