#pragma once
#include <lwtensor/internal/types.h>
#include <lwtensor/internal/defines.h>

#include <functional>
#include <sstream>
#include <vector>
#include <algorithm>

namespace LWTENSOR_NAMESPACE
{
    class ContractionDescriptorInternal : public Initializable<73>
    {

        public:
            ContractionDescriptorInternal() {}

        lwtensorStatus_t initContractionDescriptorInternal(
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
                bool &swapAB);

        extent_type getTotalExtentM() const;
        extent_type getTotalExtentN() const;
        extent_type getTotalExtentK() const;
        extent_type getTotalExtentL() const;

        static constexpr int LWTENSOR_MAX_MODES_M = ElementwiseParameters::LWTENSOR_MAX_MODES;
        static constexpr int LWTENSOR_MAX_MODES_N = ElementwiseParameters::LWTENSOR_MAX_MODES;
        static constexpr int LWTENSOR_MAX_MODES_K = ElementwiseParameters::LWTENSOR_MAX_MODES;
        static constexpr int LWTENSOR_MAX_MODES_L = ElementwiseParameters::LWTENSOR_MAX_MODES;

        lwdaDataType_t typeA_;
        lwdaDataType_t typeB_;
        lwdaDataType_t typeC_;
        lwtensorComputeType_t typeCompute_;

        int nmodeM;
        int nmodeN;
        int nmodeK;
        int nmodeL;

        extent_type extentM[LWTENSOR_MAX_MODES_M];
        extent_type extentN[LWTENSOR_MAX_MODES_N];
        extent_type extentK[LWTENSOR_MAX_MODES_K];
        extent_type extentL[LWTENSOR_MAX_MODES_L];

        stride_type strideAm[LWTENSOR_MAX_MODES_M];
        stride_type strideAk[LWTENSOR_MAX_MODES_K];
        stride_type strideBn[LWTENSOR_MAX_MODES_N];
        stride_type strideBk[LWTENSOR_MAX_MODES_K];
        stride_type strideCm[LWTENSOR_MAX_MODES_M];
        stride_type strideCn[LWTENSOR_MAX_MODES_N];

        stride_type strideAl[LWTENSOR_MAX_MODES_L];
        stride_type strideBl[LWTENSOR_MAX_MODES_L];
        stride_type strideCl[LWTENSOR_MAX_MODES_L];

        uint32_t alignmentRequirementA_; //< minimal alignment requirement for A (in bytes)
        bool stridedLoadsReqA_;
        bool contiguousModeIsBatchedA_ = false;
        lwtensorOperator_t opA_;
        bool transA_;

        uint32_t alignmentRequirementB_; //< minimal alignment requirement for B (in bytes)
        bool stridedLoadsReqB_;
        bool contiguousModeIsBatchedB_ = false;
        lwtensorOperator_t opB_;
        bool transB_;

        uint32_t alignmentRequirementC_; //< minimal alignment requirement for C (in bytes)
        lwtensorOperator_t opC_;

        bool forceUseNaiveEpilogue_; ///< Forces the use of the naive epilogue (for correctness reasons)

        int32_t partitions_ = -1; ///< -1 denotes that the heuristic should be chosen

        template <class T>
        static inline void combineHash(size_t& seed, const T& v)
        {
            std::hash<T> hasher;
            seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        }

        static size_t getHash(
                const ModeList &modeA, const StrideMap &strideA, lwdaDataType_t typeA, lwtensorOperator_t opA, uint32_t alignmentReqA,
                const ModeList &modeB, const StrideMap &strideB, lwdaDataType_t typeB, lwtensorOperator_t opB, uint32_t alignmentReqB,
                const ModeList &modeC, const StrideMap &strideC, lwdaDataType_t typeC, lwtensorOperator_t opC, uint32_t alignmentReqC,
                const ExtentMap &extent, const lwtensorComputeType_t typeCompute, const uint64_t workspaceSize, const uint32_t autotuneMode,
                const uint32_t tag)
        {
            size_t hash = 0;

            std::vector<mode_type> vecModeA(modeA.size());
            std::vector<mode_type> vecModeB(modeB.size());
            std::vector<mode_type> vecModeC(modeC.size());
            /* Colwert data structures */
            int i = 0;
            for (auto mode : modeA)
            {
                vecModeA[i++] = mode;
                combineHash(hash, strideA.at(mode));
            }
            i = 0;
            for (auto mode : modeB)
            {
                vecModeB[i++] = mode;
                combineHash(hash, strideB.at(mode));
            }
            i = 0;
            for (auto mode : modeC)
            {
                vecModeC[i++] = mode;
                combineHash(hash, strideC.at(mode));
            }

            std::vector<mode_type> canonicalizedA(modeA.size());
            std::vector<mode_type> canonicalizedB(modeB.size());
            std::vector<mode_type> canonicalizedC(modeC.size());

            canonicalizeModes(
                    vecModeA,
                    vecModeB,
                    vecModeC,
                    canonicalizedA,
                    canonicalizedB,
                    canonicalizedC);

            /* A */
            combineHash(hash, static_cast<int32_t>(typeA));
            combineHash(hash, static_cast<int32_t>(opA));
            combineHash(hash, alignmentReqA);
            for (auto mode : canonicalizedA)
            {
                combineHash(hash, mode);
            }

            /* B */
            combineHash(hash, static_cast<int32_t>(typeB));
            combineHash(hash, static_cast<int32_t>(opB));
            combineHash(hash, alignmentReqB);
            for (auto mode : canonicalizedB)
            {
                combineHash(hash, mode);
            }

            /* C */
            combineHash(hash, static_cast<int32_t>(typeC));
            combineHash(hash, static_cast<int32_t>(opC));
            combineHash(hash, alignmentReqC);
            for (auto mode : canonicalizedC)
            {
                combineHash(hash, mode);
            }

            for (auto item : extent)
            {
                combineHash(hash, item.second);
            }
            combineHash(hash, static_cast<int32_t>(typeCompute));
            combineHash(hash, workspaceSize);
            combineHash(hash, autotuneMode);
            combineHash(hash, tag);
            return hash;
        }

        // helper function for ContractionDescriptor::getIdentifier()
        static void getIdentifier(std::stringstream& s,
                const ModeList &modeA, const StrideMap &strideA, lwdaDataType_t typeA, lwtensorOperator_t opA, uint32_t alignmentReqA,
                const ModeList &modeB, const StrideMap &strideB, lwdaDataType_t typeB, lwtensorOperator_t opB, uint32_t alignmentReqB,
                const ModeList &modeC, const StrideMap &strideC, lwdaDataType_t typeC, lwtensorOperator_t opC, uint32_t alignmentReqC,
                const ExtentMap &extent)
        {
            std::vector<mode_type> vecModeA(modeA.size());
            std::vector<mode_type> vecModeB(modeB.size());
            std::vector<mode_type> vecModeC(modeC.size());
            std::vector<stride_type> vecStrideA(strideA.size());
            std::vector<stride_type> vecStrideB(strideB.size());
            std::vector<stride_type> vecStrideC(strideC.size());
            /* Colwert data structures */
            int i = 0;
            for (auto mode : modeA)
            {
                vecModeA[i] = mode;
                vecStrideA[i++] = strideA.at(mode);
            }
            i = 0;
            for (auto mode : modeB)
            {
                vecModeB[i] = mode;
                vecStrideB[i++] = strideB.at(mode);
            }
            i = 0;
            for (auto mode : modeC)
            {
                vecModeC[i] = mode;
                vecStrideC[i++] = strideC.at(mode);
            }

            getIdentifier(s,
                    vecModeA, vecStrideA, typeA, opA, alignmentReqA,
                    vecModeB, vecStrideB, typeB, opB, alignmentReqB,
                    vecModeC, vecStrideC, typeC, opC, alignmentReqC, extent);
        }

        int info(char* dst, int sz) const
        {
            if( sz <= 0 )
            {
                return 0;
            }
            auto s = this->getIdentifier(0);

            strncpy(dst, s.c_str(), sz - 1);
            dst[sz - 1] = '\0';
            return strlen(dst);
        }

    private:

        uint32_t getAlignmentRequirement(const lwdaDataType_t dataType,
                                         const StrideMap& stride,
                                         const ExtentMap& extent,
                                         const mode_type vecMode) const;

        /**
         * \pre Expects that modes are already sorted and fused; moreover, A and B must
         * already be swaped (i.e., the first mode in C should be found in A--not B)
         */
        static void getIdentifier(std::stringstream& s,
                const std::vector<mode_type> &modeA, const std::vector<stride_type> &strideA, lwdaDataType_t typeA, lwtensorOperator_t opA, uint32_t alignmentReqA,
                const std::vector<mode_type> &modeB, const std::vector<stride_type> &strideB, lwdaDataType_t typeB, lwtensorOperator_t opB, uint32_t alignmentReqB,
                const std::vector<mode_type> &modeC, const std::vector<stride_type> &strideC, lwdaDataType_t typeC, lwtensorOperator_t opC, uint32_t alignmentReqC,
                const ExtentMap &extent)
        {
            std::vector<mode_type> canonicalizedA(modeA.size());
            std::vector<mode_type> canonicalizedB(modeB.size());
            std::vector<mode_type> canonicalizedC(modeC.size());

            canonicalizeModes(
                    modeA,
                    modeB,
                    modeC,
                    canonicalizedA,
                    canonicalizedB,
                    canonicalizedC);

            s << "A(" << std::to_string(static_cast<int32_t>(typeA)) << "," << std::to_string(static_cast<int32_t>(opA)) << "," << std::to_string(alignmentReqA) << ")";
            for (auto mode : canonicalizedA) {
                s << mode << ",";
            }

            int i = 0;
            for (auto mode : modeA) {
                s << "(" << extent.at(mode) << ":" << strideA.at(i++) << ")";
            }
            s << "B(" << std::to_string(static_cast<int32_t>(typeB)) << "," << std::to_string(static_cast<int32_t>(opB)) << "," << std::to_string(alignmentReqB) << ")";
            for (auto mode : canonicalizedB) {
                s << mode << ",";
            }
            i = 0;
            for (auto mode : modeB) {
                s << "(" << extent.at(mode) << ":" << strideB.at(i++) << ")";
            }
            s << "C(" << std::to_string(static_cast<int32_t>(typeC)) << "," << std::to_string(static_cast<int32_t>(opC)) << "," << std::to_string(alignmentReqC) << ")";
            for (auto mode : canonicalizedC) {
                s << mode << ",";
            }
            i = 0;
            for (auto mode : modeC) {
                s << "(" << extent.at(mode) << ":" << strideC.at(i++) << ")";
            }
        }

        /**
         * Returns a string that uniquely identifies the given tensor contraction;
         * this is mostly used to identify the contraction within the plan cache.
         */
        std::string getIdentifier(const uint64_t workspaceSize) const
        {
            ExtentMap extents;
            int mode = 0;
            std::vector<mode_type> modeA;
            std::vector<mode_type> modeB;
            std::vector<mode_type> modeC;
            StrideMap strideA;
            StrideMap strideB;
            StrideMap strideC;

            /*
             * Recover modesA, modeB, and modeC from M-, N-, K-, and L-indices
             */
            for (int i = 0; i < nmodeM; i++) {
              modeA.push_back(mode);
              modeC.push_back(mode);
              strideA[mode] = strideAm[i];
              strideC[mode] = strideCm[i];
              extents[mode++] = extentM[i];
            }
            for (int i = 0; i < nmodeN; i++) {
              modeB.push_back(mode);
              modeC.push_back(mode);
              strideB[mode] = strideBn[i];
              strideC[mode] = strideCn[i];
              extents[mode++] = extentN[i];
            }
            for (int i = 0; i < nmodeK; i++) {
              modeA.push_back(mode);
              modeB.push_back(mode);
              strideA[mode] = strideAk[i];
              strideB[mode] = strideBk[i];
              extents[mode++] = extentK[i];
            }
            for (int i = 0; i < nmodeL; i++) {
              modeA.push_back(mode);
              modeB.push_back(mode);
              modeC.push_back(mode);
              strideA[mode] = strideAl[i];
              strideB[mode] = strideBl[i];
              strideC[mode] = strideCl[i];
              extents[mode++] = extentL[i];
            }

            // NOTE (ajedrych): This part is commented to retain information about ordering of modes in the lwtensorTest output.
            //                  It is necessary for callwlating features to DNN.

            // sort modes w.r.t. increasing strides
            /* std::sort(modeA.begin(), modeA.end(), [&](int a, int b) { */
            /*   return (strideA[a] < strideA[b]) || ((strideA[a] == strideA[b]) && extents[a] > extents[b]) ; // use extent as tie-breaker */
            /* }); */
            /* std::sort(modeB.begin(), modeB.end(), [&](int a, int b) { */
            /*   return (strideB[a] < strideB[b]) || ((strideB[a] == strideB[b]) && extents[a] > extents[b]) ; // use extent as tie-breaker */
            /* }); */
            /* std::sort(modeC.begin(), modeC.end(), [&](int a, int b) { */
            /*   return (strideC[a] < strideC[b]) || ((strideC[a] == strideC[b]) && extents[a] > extents[b]) ; // use extent as tie-breaker */
            /* }); */

            std::vector<stride_type> sortedStrideA;
            for (auto a : modeA)
            {
                sortedStrideA.push_back(strideA[a]);
            }

            std::vector<stride_type> sortedStrideB;
            for (auto a : modeB)
            {
                sortedStrideB.push_back(strideB[a]);
            }

            std::vector<stride_type> sortedStrideC;
            for (auto a : modeC)
            {
                sortedStrideC.push_back(strideC[a]);
            }

            bool swapAB = false; // ensures that the stride-1 mode of C is part of A -- not B
            if( modeC.size() > 0 )
            {
                for (const auto mode : modeB) {
                    if (mode == modeC.front()) {
                        swapAB = true;
                        break;
                    }
                }
            }

            std::stringstream s;
            if( swapAB )
            {
                getIdentifier(s,
                        modeB, sortedStrideB, typeB_, opB_, alignmentRequirementB_,
                        modeA, sortedStrideA, typeA_, opA_, alignmentRequirementA_,
                        modeC, sortedStrideC, typeC_, opC_, alignmentRequirementC_,
                        extents);
            }
            else
            {
                getIdentifier(s,
                        modeA, sortedStrideA, typeA_, opA_, alignmentRequirementA_,
                        modeB, sortedStrideB, typeB_, opB_, alignmentRequirementB_,
                        modeC, sortedStrideC, typeC_, opC_, alignmentRequirementC_,
                        extents);
            }
            s << ",c=" << std::to_string(static_cast<int32_t>(typeCompute_));
            s << ",ws=" + std::to_string(workspaceSize);
            s << ",p=" + std::to_string(partitions_) + ",";
            return s.str();
        }
        lwtensorStatus_t initContractionDescriptorInternal_(
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
                bool contiguousModeIsBatchedB);

    public:
        /**
         * Canonicalizes mode names: start naming modes in A a,b,c,... followed by
         * modes in B and eventually C
         * E.g., C[m,n] = A[m,k] * B[k,n] -> C[a,c] = A[a,b] * B[b,c]
         */
        static void canonicalizeModes(
                const std::vector<mode_type> &inA,
                const std::vector<mode_type> &inB,
                const std::vector<mode_type> &inC,
                std::vector<mode_type> &outA,
                std::vector<mode_type> &outB,
                std::vector<mode_type> &outC)
        {
            const int numModesA = inA.size();
            const int numModesB = inB.size();
            const int numModesC = inC.size();
            assert(outA.size() == numModesA);
            assert(outB.size() == numModesB);
            assert(outC.size() == numModesC);
            char lwrrentMode = 0;
            // canonicalize the modes that appear in A and (B or C)
            for(int i=0; i < numModesA; ++i)
            {
                for(int j=0; j < numModesB; ++j)
                {
                    if( inA[i] == inB[j] )
                    {
                        outB[j] = lwrrentMode;
                    }
                }
                for(int j=0; j < numModesC; ++j)
                {
                    if( inA[i] == inC[j] )
                    {
                        outC[j] = lwrrentMode;
                    }
                }
                outA[i] = lwrrentMode;
                lwrrentMode++;
            }
            // canonicalize the modes that appear in B and C and not in A
            for(int i=0; i < numModesB; ++i)
            {
                bool found =  false;
                for(int j=0; j < numModesA; ++j)
                {
                    if( inA[j] == inB[i] )
                    {
                        found = true;
                        break;
                    }
                }
                if( found )
                {
                    continue; // skip this mode (we already canonicalized it above)
                }
                for(int j=0; j < numModesC; ++j)
                {
                    if( inC[j] == inB[i] )
                    {
                        outC[j] = lwrrentMode;
                    }
                }
                outB[i] = lwrrentMode;
                lwrrentMode++;
            }
        }
    };
}
