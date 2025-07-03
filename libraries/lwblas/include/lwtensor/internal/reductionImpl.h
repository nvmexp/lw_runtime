#pragma once

#include <lwda_runtime.h>

#include <lwtensor/internal/operators.h>
#include <lwtensor/internal/typesEx.h>
#include <lwtensor/internal/utilEx.h>
#include <lwtensor/internal/defines.h>

namespace LWTENSOR_NAMESPACE
{
constexpr uint32_t WARP_SIZE = 32;

template <typename T>
struct WarpShuffle
{
    static __device__ T down(unsigned mask, T var, int srcLane) { return __shfl_down_sync(mask, var, srcLane); }
};
template <>
struct WarpShuffle<lwComplex>
{
    static __device__ lwComplex down(unsigned mask, lwComplex var, int srcLane)
    {
        union foo {
            double d;
            lwComplex c;
        };
        foo tmp;
        tmp.c = var;
        tmp.d = __shfl_down_sync(mask, tmp.d, srcLane);
        return tmp.c;
    }
};
template <>
struct WarpShuffle<lwDoubleComplex>
{
    static __device__ lwDoubleComplex down(unsigned mask, lwDoubleComplex var, int srcLane)
    {
        lwDoubleComplex tmp;
        tmp.x = __shfl_down_sync(mask, var.x, srcLane);
        tmp.y = __shfl_down_sync(mask, var.y, srcLane);
        return tmp;
    }
};

/**
 * Reduces thread-local data (i.e., 'reduced') within a thread using the
 * opReduce operator.
 */
template <typename typeCompute, int START, int END = 0>
__device__ typeCompute warpReduce(typeCompute reduced, const lwtensorOperator_t opReduce)
{
    constexpr uint32_t FULL_MASK = 0xffffffff;
#pragma unroll
    for (int offset = START; offset > END; offset /= 2) {
        reduced = lwtensorBinaryOp<typeCompute>(reduced, WarpShuffle<typeCompute>::down(FULL_MASK, reduced, offset),
                                                opReduce);
    }
    return reduced;
}

__device__ static stride_type getOffset(extent_type idx, const extent_type extent[], const stride_type stride[],
                                        const int numModes)
{
    stride_type offset = 0;
#pragma unroll
    for (int i = 0; i < numModes; ++i) {
        extent_type offsetDim = idx % extent[i];
        idx /= extent[i];

        offset += offsetDim * stride[i];
    }
    return offset;
}

/**
 * \brief Computes t = opAB( alpha * opA(A), beta * opB(B) ); C = reduceOp(t[:]);
 *
 * This kernel traverses A in a linear fashion! Rationale: A is the largest tensor (since
 * B has not any free modes (otherwise it would be tensor contraction); we are neglecting
 * the case of a tensor-dot-product where A and B have different layouts for now.
 *
 * Each threadblock has a total of NUM_THREADS_M * NUM_THREADS_K many threads; as the name
 * suggests, NUM_THREADS_M many threads are oriented along the (logical) m-dimension,
 * whilewhile NUM_THREADS_K are oriented along the (logical) k-dimension. The reduction
 * along the k-dimension is first performed via warp-shuffle instructions (i.e.,
 * warpReduce()) and only then via shared memory. It is important to notice that the
 * threadthreads are layed out in two different settings w.r.t., transA:
 * 1) transA == true: In this case the k-dimension is the leading dimension such that the
 *                    threads are first layed-out along k-dim and then m-dim.
 * 2) transA == false: In this case the m-dimension is the leading dimension such that the
 *                     threads are first layed-out along m-dim and then k-dim.
 * Thus, as long as NUM_THREADS_M < WARP_SIZE we first must perform the reduction via
 * WarpShuffle (otherwise it can be skipped and the reduction is solely performed via
 * shared memory).
 */
template <bool transA, int VECTOR_WIDTH, int NUM_MODES, int NUM_THREADS, int NUM_THREADS_M, int NUM_THREADS_K,
          bool useB, lwtensorOperator_t opAconst, lwtensorOperator_t opBconst, lwtensorOperator_t opCconst,
          lwtensorOperator_t opABconst, lwtensorOperator_t opReduceConst, typename typeA, typename typeB,
          typename typeC, typename typeCompute, typename typeScalar>
__launch_bounds__(NUM_THREADS) __global__
    void reduction_kernel(const typeScalar alpha, const typeA* A, const typeB* B, const typeScalar beta,
                          const typeC* C, typeC* D, lwtensorOperator_t opArun, lwtensorOperator_t opBrun,
                          lwtensorOperator_t opCrun, lwtensorOperator_t opABrun, lwtensorOperator_t opReducerun,
                          const ReductionParams params)
{
    static_assert(isPower2(NUM_THREADS_M), "");
    static_assert(NUM_THREADS_M * NUM_THREADS_K == NUM_THREADS, "");
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    static_assert(NUM_WARPS <= WARP_SIZE,
                  "Ensures that the remaining elements can "
                  "again be reduced via warpReduce().");

    constexpr int VECTOR_WIDTH_M = !transA ? VECTOR_WIDTH : 1;
    constexpr int VECTOR_WIDTH_K = transA ? VECTOR_WIDTH : 1;
    constexpr int VECTOR_WIDTH_A = VECTOR_WIDTH_M > VECTOR_WIDTH_K ? VECTOR_WIDTH_M : VECTOR_WIDTH_K;
    using veca_t = vec_t<VECTOR_WIDTH_A, typeA>;
    using vecb_t = vec_t<VECTOR_WIDTH_K, typeB>;
    using vec_comp_t = vec_t<VECTOR_WIDTH, typeCompute>;
    using vec_comp_k_t = vec_t<VECTOR_WIDTH_K, typeCompute>;

    // this is just a trick to enable both compile-time as well as runtime
    // operators in a single kernel
    const auto opA = (opAconst != LWTENSOR_OP_UNKNOWN) ? opAconst : opArun;
    const auto opB = (opBconst != LWTENSOR_OP_UNKNOWN) ? opBconst : opBrun;
    const auto opC = (opCconst != LWTENSOR_OP_UNKNOWN) ? opCconst : opCrun;
    const auto opAB = (opABconst != LWTENSOR_OP_UNKNOWN) ? opABconst : opABrun;
    const auto opReduce = (opReduceConst != LWTENSOR_OP_UNKNOWN) ? opReduceConst : opReducerun;
    const uint32_t maxNumModesK = NUM_MODES != -1 ? (uint32_t)NUM_MODES : params.nmodeK_;
    const uint32_t maxNumModesM = NUM_MODES != -1 ? (uint32_t)NUM_MODES : params.nmodeM_;

    const typeCompute neutralElement = getNeutralElement<typeCompute>(opReduce);
    vec_comp_t neutralElement_vec;
    for (int i = 0; i < VECTOR_WIDTH; ++i) {
        neutralElement_vec.a[i] = neutralElement;
    }

    /* Threadblock mapping */
    // change the mapping of the threadblocks s.t. it matches the data layout of A
    const extent_type bid_k = !transA ? blockIdx.y : blockIdx.x; // blockId along k-dim
    const extent_type bid_m = !transA ? blockIdx.x : blockIdx.y; // blockId along m-dim
    const extent_type gridDim_m = !transA ? gridDim.x : gridDim.y;
    const extent_type gridDim_k = !transA ? gridDim.y : gridDim.x;
    auto const tid = threadIdx.x % NUM_THREADS;
    /* Thread mapping */
    // if transA: maps threads to M-dim first, then to K-dim, vice versa otherwise
    const extent_type tid_m = !transA ? (tid % NUM_THREADS_M) : (tid / NUM_THREADS_K); // threadId along m-dim
    const extent_type tid_k = transA ? (tid % NUM_THREADS_K) : (tid / NUM_THREADS_M);  // threadId along k-dim

    for (extent_type km = bid_m * VECTOR_WIDTH_M * NUM_THREADS_M; km < params.totalExtentM_;
         km += gridDim_m * VECTOR_WIDTH_M * NUM_THREADS_M) // unblocked free modes
    {
        // this vector corresponds to elements along the m-mode if transA == false, or
        // along the k-mode otherwise
        vec_comp_t reduced_vec = neutralElement_vec;

        extent_type offsetM = km + tid_m * VECTOR_WIDTH_M;
        const typeA* myAm = A;

        // this if statement must be within the loop (km), such that all warps can still
        // participate in the subsequent reduction
        if (NUM_THREADS_M == 1 || (offsetM < params.totalExtentM_)) { // TODO KM => offsetM????
            // TODO this could be templatized (only required for partial reductions): ~+5% perf, if skipped
            {
#pragma unroll
                for (auto i = 0; i < maxNumModesM; ++i) {
                    int div;
                    int mod;
                    
                    params.extentM_divmod[i](div, mod, offsetM);

                    myAm += params.strideAm_[i] * mod;
                    offsetM = div;
                }
            }

            // 1) Thread-local reduction
            /* Map threadblocks to unblocked modes */
            for (extent_type ku = bid_k; ku < params.unblockedExtent_; ku += gridDim_k) // unblocked contracted modes
            {
                extent_type offsetK = ku;
                const typeA* myAk = myAm;
                const typeB* myBk = B;
#pragma unroll
                for (auto i = ReductionParams::cNumBlockedModes; i < maxNumModesK; ++i) {
                    int div;
                    int mod;
                    
                    params.extentK_divmod[i](div, mod, offsetK);

                    myAk += params.strideAk_[i] * mod;
                    myBk += params.strideBk_[i] * mod;
                    offsetK = div;

                }

                // The two k-loops (i.e., ku, and idxK) are split to reduce the overhead corresponding to index
                // callwlations.
                
                /* Map threads to blocked modes */
                for (int idxK = tid_k * VECTOR_WIDTH_K; idxK < params.blockedExtent_;
                     idxK += NUM_THREADS_K * VECTOR_WIDTH_K)
                {
                    int offsetK = idxK;
                    const typeA* myAkk = myAk;
                    const typeB* myBkk = myBk;

// This loop is entirely unrolled at compile time. The value of ReductionParams::cNumBlockedModesK could actually be
// templatized; to be precise, this loop costs ~10% performance for cases where no modes must be fused (e.g., if the
// stride-1 mode is large enough)
#pragma unroll
                    for (auto i = 0; i < ReductionParams::cNumBlockedModes; ++i) {
                        int div;
                        int mod;
                        
                        params.extentK_divmod[i](div, mod, offsetK);
                        
                        myAkk += params.strideAk_[i] * mod;
                        myBkk += params.strideBk_[i] * mod;
                        offsetK = div;
                    }
                    // (vectorized) load element from A
                    vec_comp_t tmp
                        = colwert<VECTOR_WIDTH, veca_t, vec_comp_t, typeCompute>(*((const veca_t*) myAkk), opA);
                    // (vectorized) load element from B
                    if (useB) {
                        vec_comp_k_t tmpB
                            = colwert<VECTOR_WIDTH, vecb_t, vec_comp_k_t, typeCompute>(*((const vecb_t*) myBkk), opB);
#pragma unroll
                        for (uint32_t i = 0; i < VECTOR_WIDTH; ++i)
                        {
                            if( transA )
                            {
                                tmp.a[i] = lwtensorBinaryOp<typeCompute>(tmp.a[i], tmpB.a[i], opAB);
                            }
                            else
                            {
                                tmp.a[i] = lwtensorBinaryOp<typeCompute>(tmp.a[i], tmpB.a[0], opAB);
                            }
                        }
                    }

#pragma unroll
                    for (int i = 0; i < VECTOR_WIDTH; ++i)
                        reduced_vec.a[i] = lwtensorBinaryOp<typeCompute>(reduced_vec.a[i], tmp.a[i], opReduce);
                }
            }
        }

        // reduce vector locally
        vec_comp_t m_vec;
        for(int i=0; i < VECTOR_WIDTH_M; ++i){
            m_vec.a[i] = reduced_vec.a[i];
        }


        // this (local) reduction is only required in the case of transA, since in the
        // case of non-transA the vector will house elements along the m-dimension
#pragma unroll
        for(int j=0; j < VECTOR_WIDTH_M; ++j){
#pragma unroll
            for (uint32_t i = 1; i < VECTOR_WIDTH_K; ++i) {
                m_vec.a[j] = lwtensorBinaryOp(static_cast<typeCompute>(reduced_vec.a[i]), m_vec.a[j], opReduce);
            }
        }

        const auto laneId = tid % WARP_SIZE;
        const auto warpId = tid / WARP_SIZE;
        static_assert(WARP_SIZE % NUM_WARPS == 0, "");
        /*
         * TransA : threads within a threadblock are mapped to the k-dim first, followed by m-dim.
         * Non TransA : threads within a threadblock are mapped to the m-dim first, followed by k-dim.
         */
        constexpr bool WARP_REDUCTION_REQUIRED = transA || // always 
                    (NUM_THREADS_M < WARP_SIZE); // do threads --of the same warp-- correspond to different k-indices? 
        // 2) Warp reduction
        if (WARP_REDUCTION_REQUIRED ) {
            constexpr int NUM_THREADS_WARP_K = min_t<WARP_SIZE, NUM_THREADS_K>::value;
            constexpr int START = transA ? NUM_THREADS_WARP_K / 2 : WARP_SIZE / 2;
            constexpr int END = transA ? 0 : NUM_THREADS_M / 2;
#pragma unroll
            for(int j=0; j < VECTOR_WIDTH_M; ++j){
                m_vec.a[j] = warpReduce<typeCompute, START, END>(m_vec.a[j], opReduce);
            }
        }

        // 3) Threadblock reduction via shared memory
        constexpr int NUM_WARPS_K = (NUM_THREADS_K >= WARP_SIZE ) ? NUM_THREADS_K / WARP_SIZE : 1; // only required for transA
        constexpr bool SHM_REDUCTION_REQUIRED = !transA || NUM_WARPS_K > 1;
        constexpr int NUM_ELEMENTS_SHM_M = NUM_THREADS_M;
        constexpr int NUM_ELEMENTS_SHM_K = transA ? NUM_WARPS_K : NUM_WARPS;
        constexpr int NUM_ELEMENTS_SHM = NUM_THREADS_M * NUM_ELEMENTS_SHM_K;
        constexpr int NUM_WARPS_REDUCTION = max_t<NUM_ELEMENTS_SHM / WARP_SIZE, 1>::value;

        static_assert( NUM_ELEMENTS_SHM % WARP_SIZE == 0 || WARP_SIZE % NUM_ELEMENTS_SHM == 0, "");
        constexpr int NUM_THREADS_WARP_REDUCTION = min_t<NUM_ELEMENTS_SHM, WARP_SIZE>::value; // number of threads within one warp that participate in the reduction
        constexpr int NUM_ELEMENS_M_WARP = NUM_THREADS_WARP_REDUCTION / NUM_ELEMENTS_SHM_K;
        const auto m_id = warpId * NUM_ELEMENS_M_WARP + (laneId % NUM_ELEMENS_M_WARP); //< m-id for reduction
        const auto k_id = laneId / NUM_ELEMENS_M_WARP; //< k-id for reduction

        if( SHM_REDUCTION_REQUIRED ) {
            __shared__ __align__(16) vec_comp_t shm[NUM_ELEMENTS_SHM_M][NUM_ELEMENTS_SHM_K];
            static_assert(WARP_SIZE >= NUM_THREADS_M, ""); // required for reduction
            if( transA )
            {
                if (laneId == 0) {
                    shm[tid_m][warpId % NUM_WARPS_K] = m_vec;
                }
            }else{
                if (laneId < NUM_THREADS_M) {
                    shm[tid_m][warpId] = m_vec;
                }
            }
            __syncthreads();

            // One warp is responsible for WARP_SIZE / NUM_ELEMENTS_SHM_K many elements along the m-dim.
            if (warpId < NUM_WARPS_REDUCTION) {

                if( !transA || k_id < NUM_ELEMENTS_SHM_K )
                    m_vec = shm[m_id][k_id];
                constexpr int END = NUM_ELEMENS_M_WARP / 2;
                static_assert(WARP_SIZE >= NUM_ELEMENTS_SHM_K && WARP_SIZE % NUM_ELEMENTS_SHM_K == 0, ""); // All elemenents along the k-dim can be reduced by a single WARP
                constexpr int START = NUM_THREADS_WARP_REDUCTION / 2;
#pragma unroll
                for(int j=0; j < VECTOR_WIDTH_M; ++j){
                    m_vec.a[j] = warpReduce<typeCompute, START, END>(m_vec.a[j], opReduce);
                    m_vec.a[j]  = lwMul(lwGet<typeCompute>(alpha), m_vec.a[j]);
                }
            }
        }

        if (warpId < NUM_WARPS_REDUCTION) {
            // single thread writes result back to GMEM
            if (laneId < NUM_ELEMENS_M_WARP) {
                /* Callwlate correct pointer offsets */
                // the offset in the m-mode has changed (since tid_m reads, but warpId writes)
                extent_type offsetM = km + m_id * VECTOR_WIDTH_M;
                if (NUM_THREADS_M > 1 && (offsetM >= params.totalExtentM_)) continue;
                stride_type offsetC = 0;
                const typeC* myC = C;
                typeC* myD = D;
                for (auto i = 0; i < maxNumModesM; ++i) {
                    int div;
                    int mod;
                    
                    params.extentM_divmod[i](div, mod, offsetM);

                    offsetC += params.strideCm_[i] * mod;
                    offsetM = div;

                }
                myC += offsetC;
                myD += offsetC;

                const bool betaIsZero = lwIsEqual(beta, lwGet<typeScalar>(0));
                if (!betaIsZero) {
#pragma unroll
                    for(int j=0; j < VECTOR_WIDTH_M; ++j){
                        auto tmpC = loadVolatile(myC + bid_k + j * params.strideCm_[0]);
                        m_vec.a[j] = lwAdd(m_vec.a[j], lwMul(lwGet<typeCompute>(beta), lwtensorUnaryOp<typeCompute>(lwGet<typeCompute>(tmpC), opC)));
                    }
                }
#pragma unroll
                for(int j=0; j < VECTOR_WIDTH_M; ++j)
                    myD[bid_k + j * params.strideCm_[0]] = lwGet<typeC>(m_vec.a[j]);
            }
        }
    }
}

/*
 * Specialized kernel for transposeA && small k-dim
 *
 * Launches one warp per element of C such that the entire contraction can be performed
 * via warpReduce.
 */
template <int VECTOR_WIDTH, int NUM_MODES, int NUM_THREADS, int NUM_THREADS_K, bool useB, lwtensorOperator_t opAconst, lwtensorOperator_t opBconst, lwtensorOperator_t opCconst,
          lwtensorOperator_t opABconst, lwtensorOperator_t opReduceConst, typename typeA, typename typeB,
          typename typeC, typename typeCompute, typename typeScalar>
__launch_bounds__(NUM_THREADS) __global__
void reduction_kernel_transA(const typeScalar alpha, const typeA* A, const typeB* B, const typeScalar beta,
                          const typeC* C, typeC* D, lwtensorOperator_t opArun, lwtensorOperator_t opBrun,
                          lwtensorOperator_t opCrun, lwtensorOperator_t opABrun, lwtensorOperator_t opReducerun,
                          const ReductionParams params)
{
    constexpr int NUM_THREADS_M = NUM_THREADS / NUM_THREADS_K;

    using veca_t = vec_t<VECTOR_WIDTH, typeA>;
    using vecb_t = vec_t<VECTOR_WIDTH, typeB>;
    using vec_comp_t = vec_t<VECTOR_WIDTH, typeCompute>;

    // this is just a trick to enable both compile-time as well as runtime
    // operators in a single kernel
    const auto opA = (opAconst != LWTENSOR_OP_UNKNOWN) ? opAconst : opArun;
    const auto opB = (opBconst != LWTENSOR_OP_UNKNOWN) ? opBconst : opBrun;
    const auto opC = (opCconst != LWTENSOR_OP_UNKNOWN) ? opCconst : opCrun;
    const auto opAB = (opABconst != LWTENSOR_OP_UNKNOWN) ? opABconst : opABrun;
    const auto opReduce = (opReduceConst != LWTENSOR_OP_UNKNOWN) ? opReduceConst : opReducerun;
    const auto maxNumModesM = NUM_MODES != -1 ? (uint32_t) NUM_MODES : params.nmodeM_;

    const typeCompute neutralElement = getNeutralElement<typeCompute>(opReduce);
    vec_comp_t neutralElement_vec;
#pragma unroll
    for (int i = 0; i < VECTOR_WIDTH; ++i)
    {
        neutralElement_vec.a[i] = neutralElement;
    }

    /* Threadblock mapping */
    // change the mapping of the threadblocks s.t. it matches the data layout of A
    const extent_type bid_m = blockIdx.x; // blockId along m-dim
    const extent_type gridDim_m = gridDim.x;
    auto const tid = threadIdx.x % NUM_THREADS;
    /* Thread mapping */
    // maps threads to K-dim first, then to M-dim
    const extent_type tid_k = tid % NUM_THREADS_K;  // threadId along k-dim
    const extent_type tid_m = tid / NUM_THREADS_K; // threadId along m-dim

    for (extent_type km = bid_m * NUM_THREADS_M; km < params.totalExtentM_;
            km += gridDim_m * NUM_THREADS_M) // unblocked free modes
    {
        vec_comp_t reduced_vec = neutralElement_vec;

        extent_type offsetM = km + tid_m;
        const typeA* myAm = A;
        const typeC* myC = C;
        typeC* myD = D;

        // this if statement must be within the loop (km), such that all warps can still
        // participate in the subsequent reduction
        if (NUM_THREADS_M == 1 || (offsetM < params.totalExtentM_))
        {
#pragma unroll
            for (uint32_t i = 0; i < maxNumModesM; ++i)
            {
                int div;
                int mod;
                
                params.extentM_divmod[i](div, mod, offsetM);

                myAm += params.strideAm_[i] * mod;
                offsetM = div;
            }
            
            /* Map threads to blocked modes */
            for (int idxK = tid_k* VECTOR_WIDTH; idxK < params.totalExtentK_;
                    idxK += NUM_THREADS_K * VECTOR_WIDTH)
            {
                int offsetK = idxK;
                const typeA* myAkk = myAm;
                const typeB* myBkk = B;

                // This loop is entirely unrolled at compile time. The value of ReductionParams::cNumBlockedModesK could actually be
                // templatized; to be precise, this loop costs ~10% performance for cases where no modes must be fused (e.g., if the
                // stride-1 mode is large enough)
#pragma unroll
                for (auto i = 0; i < ReductionParams::cNumBlockedModes; ++i)
                {
                    int div;
                    int mod;
                    
                    params.extentK_divmod[i](div, mod, offsetK);

                    myAkk += params.strideAk_[i] * mod;
                    myBkk += params.strideBk_[i] * mod;
                    offsetK = div;
                }
                // (vectorized) load element from A
                vec_comp_t tmp
                    = colwert<VECTOR_WIDTH, veca_t, vec_comp_t, typeCompute>(*((const veca_t*) myAkk), opA);
                // (vectorized) load element from B
                if (useB)
                {
                    vec_comp_t tmpB
                        = colwert<VECTOR_WIDTH, vecb_t, vec_comp_t, typeCompute>(*((const vecb_t*) myBkk), opB);
#pragma unroll
                    for (uint32_t i = 0; i < VECTOR_WIDTH; ++i)
                    {
                        tmp.a[i] = lwtensorBinaryOp<typeCompute>(tmp.a[i], tmpB.a[i], opAB);
                    }
                }

#pragma unroll
                for (uint32_t i = 0; i < VECTOR_WIDTH; ++i)
                {
                    reduced_vec.a[i] = lwtensorBinaryOp<typeCompute>(reduced_vec.a[i], tmp.a[i], opReduce);
                }
            }
        }

        /*
         * Reduce vector locally
         */
        typeCompute m_vec = reduced_vec.a[0];

#pragma unroll
        for (uint32_t i = 1; i < VECTOR_WIDTH; ++i)
        {
            m_vec = lwtensorBinaryOp(static_cast<typeCompute>(reduced_vec.a[i]), m_vec, opReduce);
        }

        static_assert(NUM_THREADS_K <= WARP_SIZE, ""); // reduction can be performed by a single warp
        /*
         * Reduce via warpReduce
         */
        m_vec = warpReduce<typeCompute, NUM_THREADS_K / 2, 0>(m_vec, opReduce);
        // The k-dim is completely reduced at this point. 
        // Thread with k_id == 0 will hold the result

        m_vec = lwMul(lwGet<typeCompute>(alpha), m_vec);

        /*
         * Write result back to GMEM
         */
        if (tid_k == 0)
        {
            /* Callwlate correct pointer offsets */
            // the offset in the m-mode has changed (since tid_m reads, but warpId writes)
            extent_type offsetM = km + tid_m;
            if (NUM_THREADS_M > 1 && (offsetM >= params.totalExtentM_))
            {
                continue;
            }
            stride_type offsetC = 0;
            for (uint32_t i = 0; i < maxNumModesM; ++i)
            {
                int div;
                int mod;
                
                params.extentM_divmod[i](div, mod, offsetM);
                
                offsetC += params.strideCm_[i] * mod;
                offsetM = div;

            }
            myC += offsetC;
            myD += offsetC;

            const bool betaIsZero = lwIsEqual(beta, lwGet<typeScalar>(0));
            if (!betaIsZero) {
                auto tmpC = loadVolatile(myC);
                m_vec = lwAdd(m_vec, lwMul(lwGet<typeCompute>(beta), lwtensorUnaryOp<typeCompute>(lwGet<typeCompute>(tmpC), opC)));
            }
            *myD = lwGet<typeC>(m_vec);
        }
    }
}

template <uint32_t numThreads, uint32_t NUM_THREADS_M, uint32_t NUM_THREADS_K, int MAX_NUM_MODES_HIGH_PERF,
          int VECTOR_WIDTH, bool transA, bool useB, lwtensorOperator_t opAconst, lwtensorOperator_t opBconst,
          lwtensorOperator_t opCconst, lwtensorOperator_t opABconst, lwtensorOperator_t opReduceConst, typename typeA,
          typename typeB, typename typeC, typename typeCompute, typename typeScalar>
lwtensorStatus_t launchReduction_L5(const typeScalar *alpha, const typeA* A, const typeB* B,
                                    const typeScalar *beta, const typeC* C, typeC* D, lwtensorOperator_t opA,
                                    lwtensorOperator_t opB, lwtensorOperator_t opC, lwtensorOperator_t opAB,
                                    lwtensorOperator_t opReduce, const ReductionParams& params,
                                    lwdaStream_t stream, const dim3& grid)
{
    if (params.nmodeM_ <= MAX_NUM_MODES_HIGH_PERF && params.nmodeK_ <= MAX_NUM_MODES_HIGH_PERF)
    {
        // high-performance specialization (~ +20% more perf; this seems quite high but it is what it is)
        reduction_kernel<transA, VECTOR_WIDTH, MAX_NUM_MODES_HIGH_PERF, numThreads, NUM_THREADS_M, NUM_THREADS_K, useB,
                         opAconst, opBconst, opCconst, opABconst, opReduceConst, typeA, typeB, typeC,
                         typeCompute, typeScalar><<<grid, numThreads, 0, stream>>>(*alpha, A, B, *beta, C, D, opA, opB, opC, opAB,
                                                                       opReduce, params);
    }
    else
    {
        // generic fallback (for higher-dimensional tensors)
        reduction_kernel<transA, VECTOR_WIDTH, -1, numThreads, NUM_THREADS_M, NUM_THREADS_K, useB, opAconst, opBconst,
                         opCconst, opABconst, opReduceConst, typeA, typeB, typeC,
                         typeCompute, typeScalar><<<grid, numThreads, 0, stream>>>(*alpha, A, B, *beta, C, D, opA, opB, opC, opAB,
                                                                       opReduce, params);
    }
    return LWTENSOR_STATUS_SUCCESS;
}
template <int VECTOR_WIDTH, bool transA, bool useB, lwtensorOperator_t opAconst, lwtensorOperator_t opBconst,
          lwtensorOperator_t opCconst, lwtensorOperator_t opABconst, lwtensorOperator_t opReduceConst, typename typeA,
          typename typeB, typename typeC, typename typeCompute, typename typeScalar>
lwtensorStatus_t launchReduction_L4(const typeScalar* alpha, const typeA* A, const typeB* B,
                                    const typeScalar* beta, const typeC* C, typeC* D, lwtensorOperator_t opA,
                                    lwtensorOperator_t opB, lwtensorOperator_t opC, lwtensorOperator_t opAB,
                                    lwtensorOperator_t opReduce, const ReductionParams& params,
                                    void* workspace, uint64_t workspaceSize, lwdaStream_t stream)
{

    constexpr int MAX_NUM_MODES_HIGH_PERF = 6; // anything higher than 6 starts seeing performance degradation

    if( transA && params.nmodeK_ <= ReductionParams::cNumBlockedModes &&
            params.totalExtentK_ < 1024) // Steep performance fall-off after/at 1024 (data not shown)
    {
        // FYI: the constraint "params.nmodeK_ <= ReductionParams::cNumBlockedModes" could
        // be removed fairly easily but 1) supporting higher values would either require
        // more templating (i.e., larger binary) or 2) performance hits. Moreover, a value
        // of two is the most meaningful value anyway (i.e., the product of two modes can
        // easily be large).
        //
        // Specialized reduction kernel for small contracted dim:
        constexpr int NUM_THREADS = 256;
        constexpr int NUM_THREADS_K = 8; // emperically determined value (see https://jirasw.lwpu.com/browse/CUT-188)
        constexpr int NUM_THREADS_M = NUM_THREADS / NUM_THREADS_K;
        const uint32_t numBlocks = std::min((uint32_t)(params.totalExtentM_ + NUM_THREADS_M- 1) / NUM_THREADS_M, 1024U);
        if (params.nmodeM_ <= MAX_NUM_MODES_HIGH_PERF && params.nmodeK_ <= MAX_NUM_MODES_HIGH_PERF)
        {
            // high-performance specialization (~ +20% more perf; this seems quite high but it is what it is)
            reduction_kernel_transA<VECTOR_WIDTH, MAX_NUM_MODES_HIGH_PERF, NUM_THREADS, NUM_THREADS_K, useB, opAconst, opBconst, opCconst,
                opABconst, opReduceConst, typeA, typeB, typeC, typeCompute, typeScalar><<<numBlocks, NUM_THREADS, 0, stream>>>(*alpha, A, B, *beta,
                        C, D, opA, opB, opC, opAB, opReduce, params);
        }
        else
        {
            // generic fallback (for higher-dimensional tensors)
            reduction_kernel_transA<VECTOR_WIDTH, -1, NUM_THREADS, NUM_THREADS_K, useB, opAconst, opBconst, opCconst,
                opABconst, opReduceConst, typeA, typeB, typeC, typeCompute, typeScalar><<<numBlocks, NUM_THREADS, 0, stream>>>(*alpha, A, B, *beta,
                        C, D, opA, opB, opC, opAB, opReduce, params);
        }
    }
    else
    {
        constexpr uint32_t MAX_NUM_BLOCKS = 1024U;
        constexpr uint32_t numThreadsPhase1 = 256U;
        constexpr uint32_t NUM_THREADS_M = (transA) ? 1 : ReductionParams::NUM_THREADS_M;
        constexpr uint32_t NUM_THREADS_K = numThreadsPhase1 / NUM_THREADS_M;
        constexpr uint32_t numThreadsPhase2 = MAX_NUM_BLOCKS;
        const uint32_t numElementsPerBlockM = (!transA) ? VECTOR_WIDTH * NUM_THREADS_M : NUM_THREADS_M;

        if ( workspaceSize > 0 && workspace == nullptr ) {
            return handleError(LWTENSOR_STATUS_ILWALID_VALUE, "Workspace is nullptr but provided workspaceSize > 0.");
        }

        uint32_t numBlocksM = (params.totalExtentM_ + numElementsPerBlockM - 1) / numElementsPerBlockM;
        uint32_t numBlocksK = 1;
        constexpr uint32_t maxGridDimY = 65535;
        const uint32_t numBlocksMaxK = std::min(maxGridDimY, std::min(
                std::max((uint32_t) (workspaceSize / (params.totalExtentM_ * sizeof(typeCompute))), 1U), // for a parallel reduction we need to store multiple elements along k-dim for each element along the m-dim.
                (uint32_t) (params.totalExtentK_ + 767) / 768)); // ensure that the k-extent --per threadblock-- is still sufficiently large.
        static_assert(ReductionParams::targetNumBlocks <= MAX_NUM_BLOCKS, "" );
        if( numBlocksM < ReductionParams::targetNumBlocks) // insufficient blocks: Try to extract parallelism from k-dim
        {
            numBlocksK = std::min(numBlocksMaxK, std::min((uint32_t)params.unblockedExtent_, (ReductionParams::targetNumBlocks + numBlocksM -1) / numBlocksM));
        }
        if( transA )
        {
            numBlocksM = std::min(numBlocksM, maxGridDimY);
        }

        constexpr typeScalar* cNullPtr = nullptr;

        const typeScalar one = lwGet<typeScalar>(1.0f);
        const typeScalar zero = lwGet<typeScalar>(0.0f);

        const bool usesParallelReduction = numBlocksK > 1;

        const typeScalar* myAlpha = usesParallelReduction ? &one : alpha;
        const typeScalar* myBeta = usesParallelReduction ? &zero : beta;
        const void* myInput = usesParallelReduction ? nullptr : C;
        void* myOutput = usesParallelReduction ? workspace : D;
        /********************************************
         * phase 1 reduce to 'numBlocksK' many elements
         ********************************************/
        {
            dim3 grid(numBlocksM, numBlocksK);
            if (transA) {
                grid.x = numBlocksK;
                grid.y = numBlocksM;
            }

            if (usesParallelReduction) {
                ReductionParams params1(params);
                // Workspace buffer is layed out as follows: k,m1,m2,...,mn (i.e., contracted modes
                // first, followed by all free modes (as they appear in the input tensor))
                params1.strideCm_[0] = numBlocksK;
                for (uint32_t i = 0; (i + 1) < params.nmodeM_; ++i)
                    params1.strideCm_[i + 1] = params.extentM_[i] * params1.strideCm_[i];

                launchReduction_L5<numThreadsPhase1, NUM_THREADS_M, NUM_THREADS_K, MAX_NUM_MODES_HIGH_PERF,
                    VECTOR_WIDTH, transA, useB, opAconst, opBconst, opCconst, opABconst, opReduceConst,
                    typeA, typeB, typeCompute, typeCompute, typeScalar>(
                            myAlpha, A, B, myBeta, (typeCompute*) myInput, (typeCompute*) myOutput, opA, opB, opC, opAB, opReduce,
                            params1, stream, grid);
            }
            else
            {
                launchReduction_L5<numThreadsPhase1, NUM_THREADS_M, NUM_THREADS_K, MAX_NUM_MODES_HIGH_PERF,
                    VECTOR_WIDTH, transA, useB, opAconst, opBconst, opCconst, opABconst, opReduceConst,
                    typeA, typeB, typeC, typeCompute, typeScalar>(myAlpha, A, B, myBeta, (typeC*) myInput,
                            (typeC*) myOutput, opA, opB, opC, opAB, opReduce,
                            params, stream, grid);
            }
        }

        static_assert(numThreadsPhase2 == MAX_NUM_BLOCKS,
                "ensure that we have at least as many threads in the "
                "second phase as we have threadblocks in the first phase.");

        /********************************************
         * phase 2 reduce to a single element
         ********************************************/
        if (usesParallelReduction)
        {
            ReductionParams params2(params);
            params2.unblockedExtent_ = 1;
            params2.blockedExtent_ = numBlocksK;
            
            params2.extentK_[0] = numBlocksK;
            params2.extentK_divmod[0] = lwtlass::FastDivmod(params2.extentK_[0]);

            params2.totalExtentK_ = numBlocksK;
            params2.strideAk_[0] = 1;
            params2.nmodeK_ = 1;
            // TODO is this required?
            for (uint32_t i = params2.nmodeK_; i < ReductionParams::LWTENSOR_MAX_MODES; ++i)
            {
                params2.extentK_[i] = 1;
                params2.extentK_divmod[i] = lwtlass::FastDivmod(params2.extentK_[i]);
            }
            params2.strideAm_[0] = numBlocksK;
            for (uint32_t i = 0; (i + 1) < params.nmodeM_; ++i)
            {
                params2.strideAm_[i + 1] = params.extentM_[i] * params2.strideAm_[i];
            }

            constexpr int VECTOR_WIDTH_1 = 1;
            launchReduction_L4<VECTOR_WIDTH_1, true, false, LWTENSOR_OP_IDENTITY, LWTENSOR_OP_IDENTITY,
                opCconst, opABconst, opReduceConst, typeCompute, typeB, typeC, typeCompute, typeScalar> (
                    alpha, (typeCompute*)workspace, (typeB*)cNullPtr, beta, C, D, opA, opB, opC, opAB, 
                    opReduce, params2, nullptr, 0, stream);
        }
    }

    HANDLE_ERROR(lwdaGetLastError());
    return LWTENSOR_STATUS_SUCCESS;
}

/// this wrapper specializes the vectorization
template <bool transA, bool useB, lwtensorOperator_t opAconst, lwtensorOperator_t opBconst, lwtensorOperator_t opCconst,
          lwtensorOperator_t opABconst, lwtensorOperator_t opReduceConst, typename typeA, typename typeB,
          typename typeC, typename typeCompute, typename typeScalar>
lwtensorStatus_t launchReduction_L3(const typeScalar* alpha, const typeA* A, const typeB* B,
                                    const typeScalar* beta, const typeC* C, typeC* D, lwtensorOperator_t opA,
                                    lwtensorOperator_t opB, lwtensorOperator_t opC, lwtensorOperator_t opAB,
                                    lwtensorOperator_t opReduce, const ReductionParams& params,
                                    void* workspace, uint64_t workspaceSize, lwdaStream_t stream)
{
    if( params.strideAk_[0] != 1 && params.strideAm_[0] != 1 )
    {
        return handleError(
            LWTENSOR_STATUS_NOT_SUPPORTED, "A reduction for which no stride is one is not yet supported.");
    }
    constexpr int cPreferredVectorWidthBytes = ReductionParams::cPreferredVectorWidthBytes;
    constexpr int cPreferredVectorWidthElementsA
        = sizeof(typeA) >= cPreferredVectorWidthBytes ? 1 : cPreferredVectorWidthBytes / sizeof(typeA);
    constexpr int cPreferredVectorWidthElementsB
        = sizeof(typeB) >= cPreferredVectorWidthBytes ? 1 : cPreferredVectorWidthBytes / sizeof(typeB);

    constexpr auto cPreferredVectorWidthElements = cPreferredVectorWidthElementsA;

    auto checkVectorizable  = [&] (const int kVectorWidth, const void* ptr, const int typeSize,
                                   const stride_type *strideM, const int numModesM, const stride_type *strideK)
    {
        bool result = ( (uint64_t)ptr % (typeSize * kVectorWidth) ) == 0;

        const auto strideOneExtent = transA ? params.extentK_[0] : params.extentM_[0];

        // check extent of vectorized mode
        result = result && ((strideOneExtent % cPreferredVectorWidthElements) == 0);

        // ensure that first stride is unit stride
        if (transA)
        {
            result = result && (strideK[0] == 1);
        }
        else if(numModesM > 0)
        {
            result = result && (strideM[0] == 1);
        }

        // ensure that all other strides are a multiple of the vectorwidth
        for(int i= transA ? 0 : 1; i < numModesM; ++i)
        {
            result = result && (strideM[i] % kVectorWidth == 0);
        }
        for(int i= transA ? 1 : 0; i < params.nmodeK_; ++i)
        {
            result = result && (strideK[i] % kVectorWidth == 0);
        }
        assert(params.nmodeL_ <= 1); // not yet supported

        return result;
    };

    const bool isVectorizableA = checkVectorizable(cPreferredVectorWidthElements, A, sizeof(typeA), params.strideAm_, params.nmodeM_, params.strideAk_);
    const bool isVectorizableB = checkVectorizable(cPreferredVectorWidthElements, B, sizeof(typeB), nullptr, 0, params.strideBk_);

    static_assert(cPreferredVectorWidthElements >= 1, "VECTOR_WIDTH is invalid.");

    // TODO do not vectorize loads of B (B is less important than A and loads of B should not get into
    // the way of vectorizing A)
    // TODO we could run two kernels back-to-back (a vectorized one and a non-vectorized
    // kernel that deals with the peel-off elements)
    // in the case of !transA (i.e., vectorization along m-dimension) loads of B are not
    // vectorized and can thus not interfere
    if (isVectorizableA &&
       (!useB || !transA || (cPreferredVectorWidthElementsA == cPreferredVectorWidthElementsB && isVectorizableB && params.strideBk_[0] == 1)))
    {
        // vectorized kernel
        return launchReduction_L4<cPreferredVectorWidthElements, transA, useB, opAconst, opBconst, opCconst,
                                        opABconst, opReduceConst, typeA, typeB, typeC, typeCompute, typeScalar>(
            alpha, A, B, beta, C, D, opA, opB, opC, opAB, opReduce, params, workspace, workspaceSize, stream);
    }
    else
    {
        // non-vectorized kernel
        return launchReduction_L4<1, transA, useB, opAconst, opBconst, opCconst, opABconst, opReduceConst, typeA,
                                        typeB, typeC, typeCompute, typeScalar>(alpha, A, B, beta, C, D, opA, opB, opC, opAB,
                                                                   opReduce, params, workspace, workspaceSize, stream);
    }
}

/// this wrapper specializes w.r.t. the data layout of A
template <bool useB, lwtensorOperator_t opAconst, lwtensorOperator_t opBconst, lwtensorOperator_t opCconst,
          lwtensorOperator_t opABconst, lwtensorOperator_t opReduceConst, typename typeA, typename typeB,
          typename typeC, typename typeCompute, typename typeScalar>
lwtensorStatus_t launchReduction_L2(const typeScalar* alpha, const typeA* A, const typeB* B,
                                    const typeScalar* beta, const typeC* C, typeC* D, lwtensorOperator_t opA,
                                    lwtensorOperator_t opB, lwtensorOperator_t opC, lwtensorOperator_t opAB,
                                    lwtensorOperator_t opReduce, const ReductionParams& params,
                                    void* workspace, uint64_t workspaceSize, lwdaStream_t stream)
{
    const bool transA = (params.strideAk_[0] == 1);
    if (transA) {
        return launchReduction_L3<true, useB, opAconst, opBconst, opCconst, opABconst, opReduceConst, typeA,
                                        typeB, typeC, typeCompute, typeScalar>(alpha, A, B, beta, C, D, opA, opB, opC, opAB,
                                                                   opReduce, params, workspace, workspaceSize, stream);
    }
    else
    {
        return launchReduction_L3<false, useB, opAconst, opBconst, opCconst, opABconst, opReduceConst, typeA,
                                        typeB, typeC, typeCompute, typeScalar>(alpha, A, B, beta, C, D, opA, opB, opC, opAB,
                                                                   opReduce, params, workspace, workspaceSize, stream);
    }
}

/// this wrapper specializes the operators
template <bool useB, typename typeA, typename typeB, typename typeC, typename typeCompute, typename typeScalar>
lwtensorStatus_t launchReduction_L1(const typeScalar* alpha, const typeA* A, const typeB* B,
                                    const typeScalar* beta, const typeC* C, typeC* D, lwtensorOperator_t opA,
                                    lwtensorOperator_t opB, lwtensorOperator_t opC, lwtensorOperator_t opAB,
                                    lwtensorOperator_t opReduce, const ReductionParams& params,
                                    void* workspace, uint64_t workspaceSize, lwdaStream_t stream)
{
    if ((opA == LWTENSOR_OP_IDENTITY) && (opB == LWTENSOR_OP_IDENTITY) && (opC == LWTENSOR_OP_IDENTITY)
        && (opAB == LWTENSOR_OP_MUL) && (opReduce == LWTENSOR_OP_ADD))
    {
        return launchReduction_L2<useB, LWTENSOR_OP_IDENTITY, LWTENSOR_OP_IDENTITY, LWTENSOR_OP_IDENTITY,
                                        LWTENSOR_OP_MUL, LWTENSOR_OP_ADD, typeA, typeB, typeC, typeCompute, typeScalar>(
            alpha, A, B, beta, C, D, opA, opB, opC, opAB, opReduce, params, workspace, workspaceSize, stream);
    }
    else
    {
        return launchReduction_L2<useB, LWTENSOR_OP_UNKNOWN, LWTENSOR_OP_UNKNOWN, LWTENSOR_OP_UNKNOWN,
                                        LWTENSOR_OP_UNKNOWN, LWTENSOR_OP_UNKNOWN, typeA, typeB, typeC, typeCompute, typeScalar>(
            alpha, A, B, beta, C, D, opA, opB, opC, opAB, opReduce, params, workspace, workspaceSize, stream);
    }
}

/**
 * \param[in] B May be null pointer, in that case the input is ignored.
 */
template <typename typeA, typename typeB, typename typeC, typename typeCompute, typename typeScalar>
lwtensorStatus_t launchReduction_L0(const typeScalar* alpha, const typeA* A, const typeB* B, const typeScalar* beta,
                                 const typeC* C, typeC* D, lwtensorOperator_t opA, lwtensorOperator_t opB,
                                 lwtensorOperator_t opC, lwtensorOperator_t opAB, lwtensorOperator_t opReduce,
                                 const ReductionParams& params, void* workspace, uint64_t workspaceSize,
                                 lwdaStream_t stream)
{
    static_assert( std::is_same<typeA, typeB>::value, "since we may swap");
    if (B != nullptr) {
        return launchReduction_L1<true, typeA, typeB, typeC, typeCompute, typeScalar>(
            alpha, A, B, beta, C, D, opA, opB, opC, opAB, opReduce, params, workspace, workspaceSize, stream);
    }
    else
    {
        return launchReduction_L1<false, typeA, typeB, typeC, typeCompute, typeScalar>(
            alpha, A, B, beta, C, D, opA, opB, opC, opAB, opReduce, params, workspace, workspaceSize, stream);
    }
}
}

