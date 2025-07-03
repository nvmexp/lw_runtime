/**
* @file
* @brief This file contains none-vectorized elementwise lwca kernels.
*/
#pragma once

#include <stdio.h>
#include <assert.h>

#include <lwda_runtime.h>

#include <lwtensor/types.h>
#include <lwtensor/internal/types.h>
#include <lwtensor/internal/util.h>
#include <lwtensor/internal/operators.h>
#include <lwtensor/internal/context.h>
#include <lwtensor/internal/defines.h>
#include <lwtensor/internal/elementwiseModern.h>

#include "lwtlass/fast_math.h"

namespace LWTENSOR_NAMESPACE
{
    template<int N, class T>
    static __inline__ __device__ vec_t<N, T> loadVec(const T* address)
    {
        constexpr int totalSize = sizeof(T) * N;
        using LoadType = typename size2type<totalSize>::type;

        #ifdef DEBUG
        assert( ((ptr_t)address) % (totalSize) == 0 )
        #endif

        const LoadType val = *reinterpret_cast<const LoadType*>(address);
        vec_t<N, T> result;
        *reinterpret_cast<LoadType*>(&result) = val;
        
        return result;
    }

    template<int N, class T>
    static __inline__ __device__ void storeVec(const vec_t<N, T> vec, T* address)
    {
        constexpr int totalSize = sizeof(T) * N;
        using StoreType = typename size2type<totalSize>::type;

        #ifdef DEBUG
        assert( ((ptr_t)address) % (totalSize) == 0 )
        #endif

        *reinterpret_cast<StoreType*>(address) = *reinterpret_cast<const StoreType*>(&vec);
    }

    /**
     * \brief Load VEC elements at once from global memory, if possible in one instruction
     * \param[in] data pointer to the elements
     * \param[in] isVectorized can we do vectorized load at all?
     * \param[in] isUnlimited false if we don't need to read some of the last elements
     * \param[in] limit max number of elements that we have to read
     * \param[in] stride element stride in data array
     * \result vector or elements
     */
    template<int VEC, typename T, typename C>
    __inline__ __device__ vec_t<VEC, C> loadTransformedVecOrSeq(
                const T *data, const bool isVectorized, const bool isUnlimited,
                const int limit, const stride_type stride, const modern::ScaleOperator<C> scale)
    {
        #ifdef DEBUG
        assert( isUnlimited == false && limit < VEC );
        assert( isVectorized && stride == 1 );
        assert( isVectorized && ( (int64_t)data % (sizeof(T) * VEC) == 0 ) );
        #endif

        vec_t<VEC, T> result;
        if (isVectorized && isUnlimited && ! scale.needLoad())
        {
            result = loadVec<VEC, T>(data);
        }
        else
        {
            for(int v = 0; v < VEC; ++v)
            {
                result.a[v] = (isUnlimited || v < limit) && ! scale.needLoad() ? data[v * stride] : lwGet<T>(0);
            }
        }

        vec_t<VEC, C> transform;
        for(uint32_t v = 0; v < VEC; ++v)
        {
            transform.a[v] = scale.unary(lwGet<C>(result.a[v]));
        }

        return transform;
    }

    /**
     * \brief Store VEC elements at once from global memory, if possible in one instruction
     * \param[in] data pointer to the elements
     * \param[in] isVectorized can we do vectorized store at all?
     * \param[in] isUnlimited false if we don't need to read some of the last elements
     * \param[in] limit max number of elements that we have to read
     * \param[in] stride element stride in data array
     * \result vector or elements
     */
    template<int VEC, typename T>
    __inline__ __device__ void storeVecOrSeq(
                const vec_t<VEC, T> vec, T *data,
                const bool isVectorized, const bool isUnlimited,
                const int limit, const int stride)
    {
        #ifdef DEBUG
        assert( isUnlimited == false && limit < VEC );
        assert( isVectorized && stride == 1 );
        assert( isVectorized && ( (int64_t)data % (sizeof(T) * VEC) == 0 ) );
        #endif

        if (isVectorized && isUnlimited)
        {
            storeVec<VEC, T>(vec, data);
        }
        else
        {
            for(int v=0; v < VEC; ++v)
            {
                if ( isUnlimited || v < limit )
                {
                    data[v * stride] = vec.a[v];
                }
            }
        }
    }

    /**
     * \brief Load hyper-cubic tile from global memory to shared memory
     * \param[in] alpha scalar of tensor 
     * \param[in] A raw pointer of tensor 
     * \param[out] shm three-dimensional shared memory that stores a tile of the tensor
     * Loads in transposed manner from gmem (along logical dim 1) and stores straight to smem
     * (which is transposed, i.e. its dim 0 corresponds to logical dim 1).
     */
    template<typename TypeIn, typename TypeOut, int kDims, typename Tile, int kVec, int kNumThreads,
        typename Full, bool kUseBinaryOperator, int kVecAlong = 1, typename Smem>
    __inline__ __device__ void loadTile(
            modern::Tensor<const TypeIn, kDims, stride_type> tA,
            Smem& shm,
            modern::ScaleOperator<TypeOut> scale,
            modern::Operator combine,
            bool useTensor, bool isVectorized,
            modern::Tuple<extent_type, kDims> limit)
    {
        if (! useTensor)
        {
            return;
        }

        modern::ThreadLayout<Tile, kNumThreads, kVec, Full, kVecAlong> layout{threadIdx.x, limit};

        // Tell the compiler it should not unroll this outer loop. After unswitching the inner-loop
        // conditions and versioning the loops, we may end up with large sequences of LDGs that
        // cause LG throttling on the SM.
        #pragma unroll 1
        for (int jj = 0; jj < layout.count() && layout.predicate(jj); jj++)
        {
            if (layout.skip(jj)) continue;
            auto pos = layout.at(jj).transpose(0, kVecAlong);
            auto x = loadTransformedVecOrSeq<kVec, TypeIn>(
                        &tA.index(pos), isVectorized, modern::At<kVecAlong, Full>::value, limit.values[kVecAlong] - pos.values[kVecAlong], tA.ld_[kVecAlong], scale);

            #pragma unroll
            for (int v = 0; v < kVec; v++)
            {
                auto& val = shm.index(pos.add(kVecAlong, v).transpose(0, 1));
                // auto& val = shm.index({0, 0});
                val = kUseBinaryOperator ? combine.binary(val, x.a[v]) : x.a[v];
            }
        }
    }

    /**
     * \brief Store hyper-cubic tile from shared memory to global memory
     * Loads and stores along logical dim "0", which is logical dim "1" in shared memory (since smem is transposed)
     */
    template<typename TypeIn, typename TypeOut, int kDims, typename Tile, int kVec, int kNumThreads, typename Full, typename Smem>
    __inline__ __device__ void storeTile(
                Smem shm,
                modern::Tensor<const TypeOut, kDims, stride_type> tC,
                modern::Tensor<TypeOut, kDims, stride_type> tD,
                modern::ScaleOperator<TypeIn> scale,
                modern::Operator combine,
                bool useTensor, bool isVectorizedLoad, bool isVectorizedStore,
                modern::Tuple<extent_type, kDims> limit,
                bool isLeadingModeUnitStrideA
                )
    {
        using  ilwec_t = vec_t<kVec, TypeIn>;
        using outvec_t = vec_t<kVec, TypeOut>;

        constexpr int kVecAlong = 0;
        modern::ThreadLayout<Tile, kNumThreads, kVec, Full, kVecAlong> layout{threadIdx.x, limit};

        #pragma unroll
        for (int jj = 0; jj < layout.count() && layout.predicate(jj); jj++)
        {
            if (layout.skip(jj)) continue;
            auto pos = layout.at(jj);

            ilwec_t AB;
            outvec_t v_out;

            for(int v=0; v < kVec; ++v)
            {
                AB.a[v] = isLeadingModeUnitStrideA ? shm.index(pos.add(kVecAlong, v)) : shm.index(pos.add(kVecAlong, v).transpose(0, 1));
            }

            // Prepare the output vector
            if (! useTensor)
            // There is no C tensor, don't try to load it
            {
                for (int v = 0; v < kVec; v++)
                {
                    v_out.a[v] = lwGet<TypeOut>(combine.binary(AB.a[v], lwGet<TypeIn>(0)));
                }
            }
            else
            // C tensor is present
            {
                // Load C, vectorized or not
                // !! Function call increases register pressure,
                // !!  which MAY slightly decrease the performance in SOME cases (~5%)
                ilwec_t v_in = loadTransformedVecOrSeq<kVec, TypeOut>(
                            &tC.index(pos), isVectorizedLoad, modern::At<kVecAlong, Full>::value, limit.values[kVecAlong] - pos.values[kVecAlong], tC.ld_[kVecAlong], scale);
                // Apply compute operations
                for (int v = 0; v < kVec; v++)
                {
                    v_out.a[v] = lwGet<TypeOut>(combine.binary(AB.a[v], v_in.a[v]));
                }
            }

            // Write the output vector, vectorized or not
            storeVecOrSeq<kVec, TypeOut>(
                    v_out, &tD.index(pos), isVectorizedStore, modern::At<kVecAlong, Full>::value, limit.values[kVecAlong] - pos.values[kVecAlong], tD.ld_[kVecAlong]);
        }
    }

    struct FastDivmodParams
    {
        lwtlass::FastDivmod divmod_[kMaxNumModes];
    };


    template<int NUM_THREADS>
    struct LaunchBoundsSandbox { static const int MIN_NCTA_PER_SM = 8; };

#if __LWDA_ARCH__ >= 800
    template<>
    struct LaunchBoundsSandbox<32> { static const int MIN_NCTA_PER_SM = 14; };
    template<>
    struct LaunchBoundsSandbox<64> { static const int MIN_NCTA_PER_SM = 9; };
    template<>
    struct LaunchBoundsSandbox<128> { static const int MIN_NCTA_PER_SM = 8; };
    template<>
    struct LaunchBoundsSandbox<256> { static const int MIN_NCTA_PER_SM = 6; };
    template<>
    struct LaunchBoundsSandbox<512> { static const int MIN_NCTA_PER_SM = 2; };
#else
    template<>
    struct LaunchBoundsSandbox<32> { static const int MIN_NCTA_PER_SM = 12; };
    template<>
    struct LaunchBoundsSandbox<64> { static const int MIN_NCTA_PER_SM = 9; };
    template<>
    struct LaunchBoundsSandbox<128> { static const int MIN_NCTA_PER_SM = 8; };
    template<>
    struct LaunchBoundsSandbox<256> { static const int MIN_NCTA_PER_SM = 4; };
    template<>
    struct LaunchBoundsSandbox<512> { static const int MIN_NCTA_PER_SM = 2; };
#endif

    /**
     */
    template<
        class Config,
        typename TypeA, 
        typename TypeB, 
        typename TypeC, 
        typename TypeCompute>
    __launch_bounds__(Config::NUM_THREADS, Config::LAUNCH_BOUNDS_MIN_CTA == 0 ? LaunchBoundsSandbox<Config::NUM_THREADS>::MIN_NCTA_PER_SM : Config::LAUNCH_BOUNDS_MIN_CTA)
    __global__ void tensor_elementwise_kernel(
                    const ElementwiseParameters params,
                    const FastDivmodParams fastDivmod,
                    const int32_t totalTiles,
                    const int32_t numTilesPerCTA,
                    const TypeCompute alpha, const TypeA *const A,
                    const TypeCompute beta,  const TypeB *const B,
                    const TypeCompute gamma, const TypeC *const C,
                    TypeC * const __restrict__ D,
                    const bool isVectorizedLoadA, const bool isVectorizedLoadB,
                    const bool isVectorizedLoadC, const bool isVectorizedStoreD)
    {
        constexpr bool isStrideOneKernel = Config::NDIM_TILE == 1;
        // optimization for stride-1 cases that get mapped to 2D kernel: This
        // ensures that loads of A are still coalesced
        const bool swapLd = !isStrideOneKernel && Config::BLOCKING[0] == Config::BLOCKING[1] && params.strideA_[0] == 1;

        constexpr uint32_t NDIM_TILE = Config::NDIM_TILE;
        using Tile = typename modern::Take<NDIM_TILE, modern::MetaTuple<int, Config::BLOCKING[0], Config::BLOCKING[1], Config::BLOCKING[2]>>::type;
        using FullTrue = typename modern::Repeat<bool, NDIM_TILE, true>::type;
        using FullFalse = typename modern::Repeat<bool, NDIM_TILE, false>::type;

        constexpr int kNumThreads = static_cast<int>(Config::NUM_THREADS);
        constexpr int kVec = static_cast<int>(Config::VEC);

        bool use_b_ = (B != nullptr);
        bool use_c_ = (C != nullptr);

        modern::Tuple<stride_type, NDIM_TILE> lda{params.strideA_};
        modern::Tuple<stride_type, NDIM_TILE> ldb{params.strideB_};
        modern::Tuple<stride_type, NDIM_TILE> ldc{params.strideC_};

        if (swapLd)
        {
            lda = lda.transpose(0, 1);
            ldb = ldb.transpose(0, 1);
        }

        const ElementwiseParameters::ActivationContext *ctx = &params.activationContext;

        for (int32_t workId = 0; workId < numTilesPerCTA; workId++)
        {
            using SmemType = modern::Smem<TypeCompute, Tile, ((! isStrideOneKernel) && Config::TRANSPOSE)>;
            //__shared__ __align__(16) TypeCompute shm[M1shm][M0shm+1];
            __shared__ __align__(16) TypeCompute shm[SmemType::kSize];
            SmemType shm_modern{&shm[0]};

            // describes the virtual ctaId
            int32_t ctaId = blockIdx.x + workId * gridDim.x;

            if(ctaId >= totalTiles ) return;

            const TypeA *myA = A;
            const TypeB *myB = B;
            const TypeC *myC = C;
            TypeC *myD = D;
            extent_type limits[NDIM_TILE] = {0};
            /* *****************************************
             *   Mapping CTAs
             * *****************************************/
            #pragma unroll
            for (int i = 0; i < NDIM_TILE; i++)
            {
                extent_type idx = 0;
                fastDivmod.divmod_[i](ctaId, idx, ctaId);
                idx *= modern::at<Tile>(i);
                const extent_type offset = idx;
                limits[i] = params.extent_[i] - offset;
                myA += params.strideA_[i] * idx;
                myB += params.strideB_[i] * idx;
                myC += params.strideC_[i] * idx;
                myD += params.strideC_[i] * idx;
            }
            for (int i = NDIM_TILE; i < params.nmodeC_; i++)
            {
                extent_type idx = 0;
                fastDivmod.divmod_[i](ctaId, idx, ctaId);
                myA += params.strideA_[i] * idx;
                myB += params.strideB_[i] * idx;
                myC += params.strideC_[i] * idx;
                myD += params.strideC_[i] * idx;
            }
            /******************************************
             * Mapping threads
             ******************************************/
            constexpr uint32_t VEC = Config::VEC;

            using staticOpPack = typename Config::OpPack_;
            auto &opPack = params.opPack_;
            modern::Operator operatorA{staticOpPack::opA_, opPack.opA_, ctx};
            modern::Operator operatorB{staticOpPack::opB_, opPack.opB_, ctx};
            modern::Operator operatorC{staticOpPack::opC_, opPack.opC_, ctx};
            modern::ScaleOperator<TypeCompute> scaleA{operatorA, alpha};
            modern::ScaleOperator<TypeCompute> scaleB{operatorB, beta};
            modern::ScaleOperator<TypeCompute> scaleC{operatorC, gamma};
            modern::Operator operatorAB {staticOpPack::opAB_,  opPack.opAB_,  ctx, staticOpPack::opUnaryAfterBinary_, opPack.opUnaryAfterBinary_};
            modern::Operator operatorABC{staticOpPack::opABC_, opPack.opABC_, ctx};

            modern::Tensor<const TypeA, NDIM_TILE, stride_type> tA(myA, lda);
            modern::Tensor<const TypeB, NDIM_TILE, stride_type> tB(myB, ldb);
            modern::Tensor<const TypeC, NDIM_TILE, stride_type> tC(myC, ldc);
            modern::Tensor<TypeC, NDIM_TILE, stride_type> tD(myD, ldc);

            modern::Tuple<extent_type, NDIM_TILE> limit{limits};

            if (isStrideOneKernel)
            {
                using Full = FullTrue;
                modern::ThreadLayout<Tile, kNumThreads, kVec, Full, 0> layout{threadIdx.x, limit};
                for (int ii = 0; ii < layout.count() && layout.predicate(ii); ii++)
                {
                    if (layout.skip(ii)) continue;
                    auto pos = layout.at(ii);
                    vec_t<VEC, TypeCompute> x = loadTransformedVecOrSeq<VEC, TypeA>(
                        &tA.index(pos), isVectorizedLoadA, pos.values[0]+VEC < limit.values[0], limit.values[0] - pos.values[0], tA.ld_[0], scaleA);

                    if (use_b_)
                    {
                        vec_t<VEC, TypeCompute> y = loadTransformedVecOrSeq<VEC, TypeB>(
                            &tB.index(pos), isVectorizedLoadB, pos.values[0]+VEC < limit.values[0], limit.values[0] - pos.values[0], tB.ld_[0], scaleB);

                        for(uint32_t v=0; v < VEC; ++v)
                        {
                            x.a[v] = operatorAB.binary(x.a[v], y.a[v]);
                        }
                    }

                    if (use_c_)
                    {
                        vec_t<VEC, TypeCompute> y = loadTransformedVecOrSeq<VEC, TypeC>(
                            &tC.index(pos), isVectorizedLoadC, pos.values[0]+VEC < limit.values[0], limit.values[0] - pos.values[0], tC.ld_[0], scaleC);

                        for(uint32_t v=0; v < VEC; ++v)
                        {
                            x.a[v] = operatorABC.binary(x.a[v], y.a[v]);
                        }
                    }

                    vec_t<VEC, TypeC> x_c;

                    #pragma unroll
                    for(uint32_t v=0; v < VEC; ++v)
                    {
                        x_c.a[v] = lwGet<TypeC>(x.a[v]);
                    }

                    // Write the output vector, vectorized or not
                    storeVecOrSeq<VEC, TypeC>(
                        x_c, &tD.index(pos), isVectorizedStoreD, pos.values[0]+VEC < limit.values[0], limit.values[0] - pos.values[0], tD.ld_[0]);
                }
            }
            else if (! Config::TRANSPOSE)
            {

                if (modern::full_blocking<Tile>(limit))
                {
                    using Full = FullTrue;
                    modern::ThreadLayout<Tile, kNumThreads, kVec, Full, 0> layout{threadIdx.x, limit};
                    for(int ii = 0; ii < layout.count() && layout.predicate(ii); ii++)
                    {
                        if (layout.skip(ii)) continue;
                        auto pos = layout.at(ii);
                        vec_t<VEC, TypeCompute> x = loadTransformedVecOrSeq<VEC, TypeA>(
                            &tA.index(pos), isVectorizedLoadA, true, limit.values[0] - pos.values[0], tA.ld_[0], scaleA);

                        if (use_b_)
                        {
                            vec_t<VEC, TypeCompute> y = loadTransformedVecOrSeq<VEC, TypeB>(
                                &tB.index(pos), isVectorizedLoadB, true, limit.values[0] - pos.values[0], tB.ld_[0], scaleB);

                            for(uint32_t v=0; v < VEC; ++v)
                            {
                                x.a[v] = operatorAB.binary(x.a[v], y.a[v]);
                            }
                        }

                        if (use_c_)
                        {
                            vec_t<VEC, TypeCompute> y = loadTransformedVecOrSeq<VEC, TypeC>(
                                &tC.index(pos), isVectorizedLoadC, true, limit.values[0] - pos.values[0], tC.ld_[0], scaleC);

                            for(uint32_t v=0; v < VEC; ++v)
                            {
                                x.a[v] = operatorABC.binary(x.a[v], y.a[v]);
                            }
                        }

                        vec_t<VEC, TypeC> x_c;

                        #pragma unroll
                        for(uint32_t v=0; v < VEC; ++v)
                        {
                            x_c.a[v] = lwGet<TypeC>(x.a[v]);
                        }

                        // Write the output vector, vectorized or not
                        storeVecOrSeq<VEC, TypeC>(
                            x_c, &tD.index(pos), isVectorizedStoreD, true, limit.values[0] - pos.values[0], tD.ld_[0]);
                    }
                }
                else
                {
                    using Full = FullFalse;
                    modern::ThreadLayout<Tile, kNumThreads, kVec, Full, 0> layout{threadIdx.x, limit};
                    for(int ii = 0; ii < layout.count() && layout.predicate(ii); ii++)
                    {
                        if (layout.skip(ii)) continue;
                        auto pos = layout.at(ii);
                        vec_t<VEC, TypeCompute> x = loadTransformedVecOrSeq<VEC, TypeA>(
                            &tA.index(pos), isVectorizedLoadA, pos.values[0]+VEC < limit.values[0], limit.values[0] - pos.values[0], tA.ld_[0], scaleA);

                        if (use_b_)
                        {
                            vec_t<VEC, TypeCompute> y = loadTransformedVecOrSeq<VEC, TypeB>(
                                &tB.index(pos), isVectorizedLoadB, pos.values[0]+VEC < limit.values[0], limit.values[0] - pos.values[0], tB.ld_[0], scaleB);

                            for(uint32_t v=0; v < VEC; ++v)
                            {
                                x.a[v] = operatorAB.binary(x.a[v], y.a[v]);
                            }
                        }

                        if (use_c_)
                        {
                            vec_t<VEC, TypeCompute> y = loadTransformedVecOrSeq<VEC, TypeC>(
                                &tC.index(pos), isVectorizedLoadC, pos.values[0]+VEC < limit.values[0], limit.values[0] - pos.values[0], tC.ld_[0], scaleC);

                            for(uint32_t v=0; v < VEC; ++v)
                            {
                                x.a[v] = operatorABC.binary(x.a[v], y.a[v]);
                            }
                        }

                        vec_t<VEC, TypeC> x_c;

                        #pragma unroll
                        for(uint32_t v=0; v < VEC; ++v)
                        {
                            x_c.a[v] = lwGet<TypeC>(x.a[v]);
                        }

                        // Write the output vector, vectorized or not
                        storeVecOrSeq<VEC, TypeC>(
                            x_c, &tD.index(pos), isVectorizedStoreD, pos.values[0]+VEC < limit.values[0], limit.values[0] - pos.values[0], tD.ld_[0]);
                    }
                }
            }
            else
            {
                if (modern::full_blocking<Tile>(limit))
                {
                    using Full = FullTrue;

                    // load A
                    loadTile<TypeA, TypeCompute, NDIM_TILE, Tile,
                        kVec, kNumThreads, Full, false, Config::TRANSPOSE ? 1 : 0>(tA, shm_modern,
                            scaleA, operatorAB,
                            true, isVectorizedLoadA,
                            swapLd ? limit.transpose(0, 1) : limit);
                    __syncthreads();

                    // load B
                    loadTile<TypeB, TypeCompute, NDIM_TILE, Tile,
                        kVec, kNumThreads, Full, true, Config::TRANSPOSE ? 1 : 0>(tB, shm_modern,
                            scaleB, operatorAB,
                            use_b_, isVectorizedLoadB,
                            swapLd ? limit.transpose(0, 1) : limit);
                    __syncthreads();

                    // store C
                    storeTile<TypeCompute, TypeC, NDIM_TILE, Tile,
                        kVec, kNumThreads, Full>(
                            shm_modern, tC, tD,
                            scaleC, operatorABC,
                            use_c_, isVectorizedLoadC, isVectorizedStoreD,
                            limit, swapLd);
                }
                else
                {
                    using Full = FullFalse;

                    // load A
                    loadTile<TypeA, TypeCompute, NDIM_TILE, Tile,
                        kVec, kNumThreads, Full, false, Config::TRANSPOSE ? 1 : 0>(tA, shm_modern,
                            scaleA, operatorAB,
                            true, isVectorizedLoadA,
                            swapLd ? limit.transpose(0, 1) : limit);
                    __syncthreads();

                    // load B
                    loadTile<TypeB, TypeCompute, NDIM_TILE, Tile,
                        kVec, kNumThreads, Full, true, Config::TRANSPOSE ? 1 : 0>(tB, shm_modern,
                            scaleB, operatorAB,
                            use_b_, isVectorizedLoadB,
                            swapLd ? limit.transpose(0, 1) : limit);
                    __syncthreads();

                    // store C
                    storeTile<TypeCompute, TypeC, NDIM_TILE, Tile, kVec, kNumThreads, Full>(
                            shm_modern, tC, tD,
                            scaleC, operatorABC,
                            use_c_, isVectorizedLoadC, isVectorizedStoreD,
                            limit, swapLd);
                }
                if( (workId+1) < numTilesPerCTA)
                {
                    __syncthreads();
                }
            }
        }
    }

    inline void elwOverride(uint32_t& value, const char* name)
    {
        #ifndef LWTENSOR_EXPOSE_INTERNAL
        return;
        #endif
        if (! getelw(name)) return;
        int parsed_value = atoi(getelw(name));
        assert(parsed_value >= 0);
        value = parsed_value;
    }

    inline uint32_t computeElementwiseGrid(
            const ElementwiseParameters& params,
            const int ndimTile, const uint32_t* blocking,
            const uint32_t numSMs, const uint32_t numBlocksPerSM,
            const uint32_t totalTiles)
    {
        uint32_t grid = 0;

        uint32_t use_old_grid = 0;
        elwOverride(use_old_grid, "LWTENSOR_EW_GRID_OLD");

        if (use_old_grid)
        {
            const uint32_t factor = 10;
            const auto blocksPerSM = std::max(1U, totalTiles / (factor * numSMs));
            grid = std::min(totalTiles, std::max(20U, std::min(blocksPerSM * numSMs, static_cast<decltype(blocksPerSM)>(INT_MAX))));
            return grid;
        }

        bool hasRemainder = false;
        for (int i = 0; i < params.nmodeC_; i++)
        {
            if (i < ndimTile)
            {
                if (params.extent_[i] & (blocking[i] - 1) != 0)
                {
                    hasRemainder = true;
                    break;
                }
            }
        }

        const uint32_t blocksPerWave = static_cast<uint32_t>(numBlocksPerSM * numSMs);
        if (! hasRemainder)
        {
            // huh this is wrong... should be just blocksPerWave
            // grid = numSMs * blocksPerWave;
            // the factor below yields good performance for non-remainder cases
            uint32_t factor = 21;
            elwOverride(factor, "LWTENSOR_EW_FACTOR");
            grid = factor * blocksPerWave;
        }
        else
        {
            const uint32_t factor = max(1, min(20, totalTiles / blocksPerWave / 4));
    
            uint32_t previousStride = 1;
            uint32_t stride = 1;
            for (int i = 0; i < params.nmodeC_; i++)
            {
                extent_type dim = params.extent_[i];
                if (i < ndimTile)
                {
                    dim = (params.extent_[i] + blocking[i] - static_cast<extent_type>(1)) / blocking[i];
                }
                if (dim == 1)
                {
                    continue;
                }
                if (grid + stride > factor * blocksPerWave)
                {
                    break;
                }
                grid += stride;
                previousStride = stride;
                stride *= dim;
            }
            while (grid < factor * blocksPerWave)
            {
                grid += previousStride;
            }
    
        }
        grid = min(totalTiles, grid);

        elwOverride(grid, "LWTENSOR_EW_GRID");
        return grid;
    }

    template<typename Config, typename TypeA, typename TypeB, typename TypeC, typename TypeCompute>
    const void* lookupElementwiseKernel()
    {
        const void* ptr = (void*) tensor_elementwise_kernel<Config, TypeA, TypeB, TypeC, TypeCompute>;
        return ptr;
    }

    /**
     * \brief This function launch the elementwise LWCA kernel.
     * \param[in] cfg
     * \param[in] TypeA
     * \param[in] TypeB
     * \param[in] TypeC
     * \param[in] TypeCompute
     *
     */ 
    template<typename Config, typename TypeA, typename TypeB, typename TypeC, typename TypeCompute>
    void launchElementwise(
                const Context *ctx,
                const ElementwiseParameters &params,
                const int numBlocksPerSM,
                const TypeCompute alpha, const TypeA *A, const bool isVectorizedLoadA,
                const TypeCompute  beta, const TypeB *B, const bool isVectorizedLoadB,
                const TypeCompute gamma, const TypeC *C, const bool isVectorizedLoadC,
                TypeC * D, const bool isVectorizedStoreD,
                lwdaStream_t stream)
    {
        const uint32_t blocking[] = {(uint32_t)Config::BLOCKING[0],(uint32_t)Config::BLOCKING[1],(uint32_t)Config::BLOCKING[2]};

        const uint32_t totalTiles = getTotalTiles(params, 3U, blocking);

        const uint32_t numSMs = ctx->getDeviceProp()->multiProcessorCount;

        uint32_t grid = computeElementwiseGrid(params, Config::NDIM_TILE, blocking, numSMs, numBlocksPerSM, totalTiles);

        dim3 block (Config::NUM_THREADS);
    
        FastDivmodParams fastDivmod;
        for (int i = 0; i < params.nmodeC_; i++)
        {
            extent_type dim = params.extent_[i];
            if (i < Config::NDIM_TILE)
            {
                dim = (params.extent_[i] + blocking[i] - static_cast<extent_type>(1)) / blocking[i];
            }
            fastDivmod.divmod_[i] = lwtlass::FastDivmod(dim);
        }

        assert(params.nmodeC_ >= Config::NDIM_TILE);

        const int32_t numTilesPerCTA = (totalTiles + grid - 1) / grid;

        #ifdef DEBUG
        printf("Grid parameters: %d, %d %d %d\n", grid, block.x,block.y,block.z);
        #endif

        tensor_elementwise_kernel<Config>
            <<<grid, block, 0, stream>>>(
                    params,
                    fastDivmod,
                    totalTiles,
                    numTilesPerCTA,
                    alpha, A,
                    beta,  B,
                    gamma, C, D,
                    isVectorizedLoadA, isVectorizedLoadB,
                    isVectorizedLoadC, isVectorizedStoreD);
    }
}
