/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2017-2020 by LWPU Corporation. All rights reserved. All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation. Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#include <atomic>
#include <iterator>
#include <fstream>
#include <ostream>
#include <random>
#include <tuple>
#include <vector>

#include <sys/time.h>

#include <boost/math/common_factor_rt.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multi_array.hpp>
#include <boost/range.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/scope_exit.hpp>

#include "protoutil_common.h"
#include "protoutil_commands.h"
#include "protoutil_routing.h"
#include "protoutil_routing_lr.h"

#include "topology.pb.h"

using namespace std;
using boost::multiprecision::cpp_int;

// This is a three dimensional array that stores information how sublinks (i.e.
// one direction of an LWLink link) are oclwpied in a response to each pair of
// source and sink devices. For every connection it stores a 2D array of
// numbers. Each number with indices [i, j] in this array indicate how this
// connection is oclwpied when device j reads data from device i.
typedef vector<boost::multi_array<OccNumType, 2>> RespOclwpancyTensor;

// This is a four dimensional array that stores information how sublinks (i.e.
// one direction of an LWLink link) are oclwpied in a response to each pair of
// source and sink devices. For every connection it stores a 3D array of
// numbers. The additional dimension corresponds to the address regions of the
// receiving device. These addresses can influence the data path and thus change
// the oclwpancy. Each number with indices [i, j, k] in this array indicate how
// this connection is oclwpied when device i writes to the k address region of
// the device j.
typedef vector<boost::multi_array<OccNumType, 3>> ReqOclwpancyTensor;

// When we search for a solution we try various combinations of a fixed number
// of candidates. This table for each candidate stores a bit mask of other
// candidates this candidate is compatible with. When we merge candidates we
// just bitwise OR the correspondent elements of this table.
typedef vector<cpp_int> CompatibilityTable;

// Oclwpancy matrix O is the matrix such as if we find a vector c that Oc = 1,
// then c is the solution for the maximum traffic.
typedef boost::multi_array<OccNumType, 2> OclwpancyMatrix;

// A candidate vector is a vector that has a few elements set to 0 and a few to
// 1, other elements are unknown.
struct CandidateVec
{
    ConnectionId   id;     // We can tag each candidate with a connection id. It
                           // means that this candidate sets this connection
                           // oclwpancy to 1.
    vector<size_t> zeroes; // A set of indices that this candidate has zeroes
                           // at.
    vector<size_t> ones;   // A set of indices that this candidate has ones at.
};

typedef vector<CandidateVec> Candidates;

// The following structures are for searching all subsums of elements of a
// multiset that add up to a certain fixed number.

// A multiset element with some additional information attached.
struct Element
{
    size_t     occId; // Position of this element inside a candidate vector.
    size_t     id;    // A unique id of this element in the multiset.
    OccNumType num;   // The numeric value.
};

// It's the same as `Element`, only the numeric value is normalized to an int.
struct IntElement
{
    size_t               occId;
    size_t               id;
    OccNumType::int_type num;
};

template <typename BitsStore>
struct SubsetSum
{
    size_t                 numEl;          // this subset is taken from first
                                           // `numEl + 1` elements of the
                                           // multiset
    OccNumType::int_type   sumVal;         // subset sum value
    const SubsetSum       *next = nullptr; // to save memory we store a pointer
                                           // to subsets we reuse
    vector<BitsStore>      ids;            // multiset element ids stored as
                                           // bits
};

// This is a sparse array that holds matrix Q. Each element Q[i, s] holds all
// subsets of first i elements of our multiset that add up to s.
template <typename BitsStore>
using SubsetSums = multi_index_container<
    SubsetSum<BitsStore>
  , indexed_by<
        ordered_unique<
            composite_key<
                SubsetSum<BitsStore>
              , member<
                    SubsetSum<BitsStore>
                  , size_t
                  , &SubsetSum<BitsStore>::numEl
                  >
              , member<
                    SubsetSum<BitsStore>
                  , OccNumType::int_type
                  , &SubsetSum<BitsStore>::sumVal
                  >
              >
          >
      >
  >;

// Search for all subsets that add up to `sumVal`. The idea is that in order to 
// callwlate Q[i, s] (defined above), we have to
//    (i)   get the subsets from Q[i - 1, s] - i.e. all subsets that sum up to s
//          from previous i - 1 elements.
//    (ii)  add element x[i] if it is equal s as a new subset, where x[i] is the
//          i-th element of our multiset
//    (iii) take all subsets from Q[i - 1, s - x[i]] and add x[i] to each of
//          them
// The function below does these steps relwrsively until it reaches Q[0, s].
// Element Q[0, s] is trivial: we either leave the subset empty or add a subset
// with element x[0], if x[0] == s.
// Values x[i] are assumed to be sorted.
// We need array `maxSums` - maximum possible sums of certain amount of the
// first conselwtive values of the multiset to tell right away if Q[i - 1, s]
// exists. If s exceeds `maxSums[i - 1]`, Q[i - 1, s] doesn't exist.
template <
    typename BitsStore
  , typename Num
  , typename MaxSumsCont
  , typename ElementsCont
  >
typename SubsetSums<BitsStore>::iterator
SubsetSumRecStep(
    SubsetSums<BitsStore> &subsetSums,
    const Num sumVal,
    const Num minSum,
    const MaxSumsCont &maxSums,
    const ElementsCont &es,
    size_t lwrIdx
)
{
    const BitsStore one(1);

    SubsetSum<BitsStore> newSubsetSum;
    newSubsetSum.numEl = lwrIdx;
    newSubsetSum.sumVal = sumVal;

    if (0 == lwrIdx)
    {
        if (sumVal == es[0].num)
        {
            newSubsetSum.ids.push_back(one << es[0].id);
        }
        return get<0>(subsetSums.insert(newSubsetSum));
    }

    auto prevMaxSum = maxSums[lwrIdx - 1];
    // add all subsums equal `s` for previous `i - 1` elements (if there are any)
    if (prevMaxSum >= sumVal)
    {
        auto it = subsetSums.find(make_tuple(lwrIdx - 1, sumVal));
        if (subsetSums.end() == it)
        {
            it = SubsetSumRecStep(subsetSums, sumVal, minSum, maxSums, es, lwrIdx - 1);
        }
        newSubsetSum.next = &*it;
    }

    // if the current element is equal s, add it
    if (sumVal == es[lwrIdx].num) newSubsetSum.ids.push_back(one << es[lwrIdx].id);

    // check subsums for previous `i - 1` elements that are equal `s - es[i].num` and add the
    // current element to them
    auto ss = sumVal - es[lwrIdx].num;
    if (ss >= minSum && ss <= prevMaxSum)
    {
        auto it = subsetSums.find(make_tuple(lwrIdx - 1, ss));
        if (subsetSums.end() == it)
        {
            it = SubsetSumRecStep(subsetSums, ss, minSum, maxSums, es, lwrIdx - 1);
        }

        for (const auto *lwr = &*it; nullptr != lwr; lwr = lwr->next)
        {
            for (auto &ids : lwr->ids)
            {
                newSubsetSum.ids.push_back(ids | (one << es[lwrIdx].id));
            }
        }
    }

    return get<0>(subsetSums.insert(newSubsetSum));
}

bool SolSearchRecStep(
    const Candidates &candidates,                 // pool of candidates we merge one by one to
                                                  // construct a solution
    const OclwpancyMatrix &occMtx,                // matrix that colwerts a list of sources and
                                                  // sinks into load of each connection
    const CompatibilityTable &compatibilityTable, // this table tells us what candidates can be
                                                  // merged together
    CandidateVec &solution,                       // current result of candidates merge
    ConnectionId lwrrentConn,                     // current connection we are trying
    cpp_int lwrrentMask                           // current bit mask of compatible candidates
)
{
    auto lwrrCandMask = cpp_int(1);
    // go through all candidates until we fail or a solution is found
    for (size_t i = 0; i < candidates.size(); ++i, lwrrCandMask <<= 1)
    {
        const auto &lwrCand = candidates[i];
        // try only candidates that are compatible with those already merged and
        // those that belong to `lwrrentConn`
        if (lwrrentConn == lwrCand.id && 0 != (lwrrentMask & lwrrCandMask))
        {                             //   ^ current candidate is compatible
#if defined(PRINT_MAXTRAFFIC_SEARCH_STEPS)
            static vector<size_t> mergedCandidates;
            mergedCandidates.push_back(i);

            BOOST_SCOPE_EXIT_ALL(&) { mergedCandidates.pop_back(); };
            static size_t stepNum(0);
            printf("Step %zd\n", ++stepNum);
#endif
            const auto &ones = lwrCand.ones; // all positions that are set to 1 in the new candidate
            const auto &zeroes = lwrCand.zeroes; // all positions that are set to 0 in the new candidate

            // merge the current solution with the new candidate by consolidating all 1 positions
            CandidateVec mergedSol = solution;
            boost::copy(ones, back_inserter(mergedSol.ones));
            boost::sort(mergedSol.ones);
            mergedSol.ones.erase(
                unique(mergedSol.ones.begin(), mergedSol.ones.end()), mergedSol.ones.end());

            // do the same with 0
            boost::copy(zeroes, back_inserter(mergedSol.zeroes));
            boost::sort(mergedSol.zeroes);
            mergedSol.zeroes.erase(
                unique(mergedSol.zeroes.begin(), mergedSol.zeroes.end()), mergedSol.zeroes.end());

#if defined(PRINT_MAXTRAFFIC_SEARCH_STEPS)
            printf("Current solution: ");
            for (size_t i = 0; occMtx.shape()[1] > i; ++i)
            {
                if (0 != i)
                {
                    printf(" ");
                }
                if (mergedSol.zeroes.cend() != find(mergedSol.zeroes.cbegin(), mergedSol.zeroes.cend(), i))
                {
                    printf("0");
                }
                else if (mergedSol.ones.cend() != find(mergedSol.ones.cbegin(), mergedSol.ones.cend(), i))
                {
                    printf("1");
                }
                else
                {
                    printf("*");
                }
            }
            printf("\n");
            printf("Merged candidates: ");
            for (size_t i = 0; mergedCandidates.size() > i; ++i)
            {
                if (0 != i)
                {
                    printf(", ");
                }
                printf("c%zu(%zu)", mergedCandidates[i], candidates[mergedCandidates[i]].id);
            }
            printf("\n");
#endif
            // Check if we found a solution. Start from checking we have all
            // positions set.
            if (occMtx.shape()[1] == mergedSol.zeroes.size() + mergedSol.ones.size())
            {
                // If we do have all positions set, multiply the solution to the
                // oclwpancy matrix. Since in our solution vector all elements
                // are either 0 or 1, all we have to do is to sum up the
                // positions of the oclwpancy matrix where our candidate vector
                // has 1.
                bool allAreOne = true;
                for (size_t i = 0; occMtx.shape()[0] > i; ++i)
                {
                    OccNumType sum;
                    for (auto it = mergedSol.ones.begin(); mergedSol.ones.end() != it; ++it)
                    {
                        sum += occMtx[i][*it];
                    }
                    if (sum != 1)
                    {
                        // We found a not 1 element, i.e. connection i is either
                        // underoclwpied or overoclwpied. We can continue to the
                        // next candidate.
                        allAreOne = false;
                        break;
                    }
                }
                if (allAreOne)
                {
                    // We found a solution. Assign solutions with this
                    // candidate to the output parameter, set a global flag and
                    // leave the function.
                    solution = mergedSol;
                    return true;
                }
                else
                {
                    continue;
                }
            }

#if defined(PRINT_MAXTRAFFIC_SEARCH_STEPS)
            {
                printf("Oc = [");
                for (size_t i = 0; occMtx.shape()[0] > i; ++i)
                {
                    if (0 != i)
                    {
                        printf(" ");
                    }
                    std::vector<size_t> notZero;
                    for (size_t j = 0; occMtx.shape()[1] > j; ++j)
                    {
                        if (0 != occMtx[i][j]) notZero.push_back(j);
                    }
                    std::vector<size_t> known;
                    copy(mergedSol.zeroes.begin(), mergedSol.zeroes.end(), back_inserter(known));
                    copy(mergedSol.ones.begin(), mergedSol.ones.end(), back_inserter(known));
                    sort(known.begin(), known.end());
                    auto notZeroIt = notZero.begin();
                    auto knownIt = known.begin();
                    bool allAreKnown = true;
                    while (notZeroIt != notZero.end())
                    {
                        if (knownIt == known.end())
                        {
                            allAreKnown = false;
                            break;
                        }

                        if (*notZeroIt < *knownIt)
                        {
                            allAreKnown = false;
                            break;
                        }
                        else
                        {
                            if (*knownIt == *notZeroIt)
                            {
                                ++notZeroIt;
                            }
                            ++knownIt;
                        }
                    }
                    if (!allAreKnown)
                    {
                        OccNumType sum;
                        for (auto it = mergedSol.ones.begin(); mergedSol.ones.end() != it; ++it)
                        {
                            sum += occMtx[i][*it];
                        }
                        if (1 < sum)
                        {
                            printf("H");
                        }
                        else
                        {
                            printf("*");
                        }
                    }
                    else
                    {
                        OccNumType sum;
                        for (auto it = mergedSol.ones.begin(); mergedSol.ones.end() != it; ++it)
                        {
                            sum += occMtx[i][*it];
                        }
                        if (1 == sum.denominator())
                        {
                            printf("%d", static_cast<int>(sum.numerator()));
                        }
                        else
                        {
                            printf("%d/%d", static_cast<int>(sum.numerator()), static_cast<int>(sum.denominator()));
                        }
                    }
                }
                printf("]\n");
            }
#endif
            // The next two checks verify if we have reached a dead end. I.e.
            // there is no way any subsequent merges will produce a solution.
            
            // First check if candidates produce something bigger than 1 in the
            // right part.
            OccNumType sum;
            for (size_t i = 0; occMtx.shape()[0] > i && 1 >= sum; ++i)
            {
                sum = 0;
                for (auto it = mergedSol.ones.begin(); mergedSol.ones.end() != it; ++it)
                {
                    sum += occMtx[i][*it];
                }
            }
            if (1 < sum) continue;

            // Second check if candidates zero out more than they should.
            sum = 1;
            for (size_t i = 0; occMtx.shape()[0] > i && 1 <= sum; ++i)
            {
                sum = 0;
                for (size_t j = 0; occMtx.shape()[1] > j; ++j)
                {
                    // We sum up everything that is not zeroed, i.e. anything
                    // that has a chance to incorporate to a solution in
                    // subsequent merges. It should be 1 or more.
                    if (mergedSol.zeroes.end() == find(mergedSol.zeroes.begin(), mergedSol.zeroes.end(), j))
                    {
                        sum += occMtx[i][j];
                    }
                }
            }
            if (1 > sum) continue;
#if defined(PRINT_MAXTRAFFIC_SEARCH_STEPS)
            {
                cpp_int singleBitMask = 1;
                bool firstIter = true;
                for (size_t i = 0; candidates.size() > i; ++i, singleBitMask <<= 1)
                {
                    if (0 != (lwrrentMask & singleBitMask))
                    {
                        if (!firstIter) printf(" ");
                        printf("%u", static_cast<unsigned int>(i));
                        firstIter = false;
                    }
                }
                printf("\n");
            }
#endif
            // Finally, if we are here, then the current candidate fits in,
            // there is a chance to find a solution. Go to the next connection
            // relwrsively.
            if (SolSearchRecStep
            (
                candidates,
                occMtx,
                compatibilityTable,
                mergedSol,
                lwrrentConn + 1,
                lwrrentMask & compatibilityTable[i]
            ))
            {
                solution = mergedSol;
                return true;
            }
        }
    }

    return false;
}

bool SolveOclwpancyEq(const OclwpancyMatrix &occMtx, CandidateVec &solution)
{
    Candidates candidates;
    CompatibilityTable compatibilityTable;

    const int numRows = static_cast<int>(occMtx.shape()[0]);

    #pragma omp parallel for
    for (int i = 0; i < numRows; ++i)
    {
        typedef OclwpancyMatrix::index_range range;
        auto lwrRow = occMtx[boost::indices[i][range::all()]];

        if (all_of(lwrRow.begin(), lwrRow.end(), [](const OccNumType &o) { return o == 0 || o == 1; }))
        {
            // simple case, just generate a candidate for every 1
            vector<size_t> onePositions;
            for (auto it = lwrRow.begin(); lwrRow.end() != it; ++it)
            {
                if (1 == *it) onePositions.emplace_back(distance(lwrRow.begin(), it));
            }
            for (auto lwrOnePos : onePositions)
            {
                CandidateVec newCand;
                newCand.id = i;
                for (auto pos : onePositions)
                {
                    if (pos == lwrOnePos) newCand.ones.push_back(pos);
                    else newCand.zeroes.push_back(pos);
                }
                sort(newCand.ones.begin(), newCand.ones.end());
                sort(newCand.zeroes.begin(), newCand.zeroes.end());
                #pragma omp critical
                {
                    candidates.push_back(newCand);
                }
            }
        }
        else
        {
            // fast search of all subsets that sum up to 1

            vector<Element> nonZeroElements;
            // find non-zero elements
            for (auto it = lwrRow.begin(); lwrRow.end() != it; ++it)
            {
                if (0 != *it)
                {
                    size_t occId = distance(lwrRow.begin(), it);
                    nonZeroElements.push_back({ occId, nonZeroElements.size(), *it });
                }
            }
            using namespace boost::adaptors;
            auto denoms = nonZeroElements
                | transformed([](const auto &e) { return e.num.denominator(); });
            OccNumType::int_type lcm;
            tie(lcm, ignore) = boost::math::lcm_range(denoms.begin(), denoms.end());

            // colwert rational numbers to integers
            vector<IntElement> multiSet;
            boost::copy(
                nonZeroElements |
                    transformed([lcm](const auto &e)
                    {
                        IntElement ie { e.occId, e.id, (e.num * lcm).numerator() };
                        return ie;
                    }),
                back_inserter(multiSet)
            );

            boost::sort(multiSet, [](const auto &e1, const auto &e2) { return e1.num < e2.num; });

            // callwlate the vector of maximum sums
            vector<OccNumType::int_type> maxSums;
            OccNumType::int_type maxSum(0);
            const OccNumType::int_type minSum = multiSet[0].num;
            boost::copy(
                multiSet |
                    transformed([&maxSum](const auto &e)
                    {
                        maxSum += e.num;
                        return maxSum;
                    }),
                back_inserter(maxSums)
            );

            SubsetSums<cpp_int> subsetSums;
            auto it = SubsetSumRecStep(
                subsetSums, lcm, minSum, maxSums, multiSet, multiSet.size() - 1
            );
            for (const auto *lwr = &*it; nullptr != lwr; lwr = lwr->next)
            {
                for (const auto subset : lwr->ids)
                {
                    CandidateVec newCand;
                    newCand.id = i;
                    cpp_int mask = 1;
                    for (size_t bit = 0; multiSet.size() > bit; ++bit, mask <<= 1)
                    {
                        if (0 != (subset & mask))
                        {
                            newCand.ones.push_back(multiSet[bit].occId);
                        }
                        else
                        {
                            newCand.zeroes.push_back(multiSet[bit].occId);
                        }
                    }
                    #pragma omp critical
                    {
                        candidates.push_back(newCand);
                    }
                }
            }
        }
    }

    // build compatibility table
    compatibilityTable.resize(candidates.size());
    cpp_int one(1);
    const int numCandidates = static_cast<int>(candidates.size());
    #pragma omp parallel for
    for (int i = 0; i < numCandidates; ++i)
    {
        auto it1 = candidates.cbegin() + i;
        for (auto it2 = candidates.cbegin(); candidates.cend() != it2; ++it2)
        {
            if (it2 == it1) continue;
            // it's compatible only if all 0 in it1 correspond either to 0
            // or N/D in it2, and all 1 in it1 correspond either to 1 or
            // N/D in it2
            auto it3 = it1->zeroes.cbegin();
            auto it4 = it2->ones.cbegin();
            bool compatible = true;
            while (it3 != it1->zeroes.cend() && it4 != it2->ones.cend())
            {
                if (*it3 < *it4) ++it3;
                else
                {
                    if (*it4 == *it3)
                    {
                        compatible = false;
                        break;
                    }
                    ++it4;
                }
            }
            if (compatible)
            {
                it3 = it1->ones.cbegin();
                it4 = it2->zeroes.cbegin();
                while (it3 != it1->ones.cend() && it4 != it2->zeroes.cend())
                {
                    if (*it3 < *it4) ++it3;
                    else
                    {
                        if (*it4 == *it3)
                        {
                            compatible = false;
                            break;
                        }
                        ++it4;
                    }
                }
            }
            if (compatible)
            {
                compatibilityTable[i] |= one << distance(candidates.cbegin(), it2);
            }
        }
    }

    // searching for a solution
    cpp_int lwrrentMask(1);
    lwrrentMask <<= candidates.size();
    lwrrentMask -= 1;

    bool res = SolSearchRecStep(candidates, occMtx, compatibilityTable, solution, 0, lwrrentMask);

    return res;
}

template <typename OutputIterator>
void SolveResp(const TopologyRouting &topoRouting, OutputIterator outIt)
{
    const auto &connections = topoRouting.GetConnections();
    const size_t numGPUs{ topoRouting.GetNumGpus() };

    RespOclwpancyTensor occTensor;

    occTensor.resize(connections.size());

    for (auto &m : occTensor) m.resize(boost::extents[numGPUs][numGPUs]);
    for (size_t i = 0; numGPUs > i; ++i)
    {
        for (size_t j = 0; numGPUs > j; ++j)
        {
            auto gpu1Id = make_tuple(DevType::GPU, i);
            auto gpu2Id = make_tuple(DevType::GPU, j);
            auto hammock = topoRouting.GetRespHammock(gpu1Id, gpu2Id);
            hammock.GetTraffic();
            for (auto it = hammock.begin(); hammock.end() != it; ++it)
            {
                occTensor[it->GetId()][i][j] = it->Get<OccNumType>();
            }
        }
    }
    sort(occTensor.begin(), occTensor.end());
    occTensor.erase(std::unique(occTensor.begin(), occTensor.end()), occTensor.end());

    boost::multi_array<bool, 2> nonZeroElements(boost::extents[numGPUs][numGPUs]);
    for (const auto &m: occTensor)
    {
        for (size_t i = 0; numGPUs > i; ++i)
        {
            for (size_t j = 0; numGPUs > j; ++j)
            {
                if (0 != m[i][j])
                {
                    nonZeroElements[i][j] = true;
                }
            }
        }
    }

    vector<array<size_t, 2>> sinksAndSources;
    for (size_t i = 0; numGPUs > i; ++i)
    {
        for (size_t j = 0; numGPUs > j; ++j)
        {
            if (nonZeroElements[i][j]) sinksAndSources.push_back({i, j});
        }
    }

    OclwpancyMatrix occMtx{ boost::extents[occTensor.size()][sinksAndSources.size()] };

    for (auto it = occTensor.cbegin(); occTensor.cend() != it; ++it)
    {
        for (size_t i = 0; numGPUs > i; ++i)
        {
            for (size_t j = 0; numGPUs > j; ++j)
            {
                array<size_t, 2> idx{i, j};
                auto idxIt = boost::find(sinksAndSources, idx);
                if (sinksAndSources.end() != idxIt)
                {
                    occMtx[distance(occTensor.cbegin(), it)]
                          [distance(sinksAndSources.begin(), idxIt)] = (*it)(idx);
                }
            }
        }
    }

    CandidateVec solution;
    if (SolveOclwpancyEq(occMtx, solution))
    {
        for (auto i : solution.ones)
        {
            *outIt++ = sinksAndSources[i];
        }
    }
}

template <typename OutputIterator>
void SolveReq(const TopologyRouting &topoRouting, OutputIterator outIt)
{
    const auto &connections = topoRouting.GetConnections();
    const size_t numGPUs{ topoRouting.GetNumGpus() };
    size_t maxNumRegions{ 0 };

    for (size_t i = 0; numGPUs > i; ++i)
    {
        maxNumRegions = max(maxNumRegions, topoRouting.GetNumRegions(make_tuple(DevType::GPU, i)));
    }
    // the algorithm needs to be improved to support several address regions, limit it to 1 for now
    maxNumRegions = 1;

    ReqOclwpancyTensor occTensor;

    occTensor.resize(connections.size());

    for (auto &m : occTensor) m.resize(boost::extents[numGPUs][numGPUs][maxNumRegions]);
    for (size_t i = 0; numGPUs > i; ++i)
    {
        for (size_t j = 0; numGPUs > j; ++j)
        {
            size_t numRegions = topoRouting.GetNumRegions(make_tuple(DevType::GPU, j));
            // the algorithm needs to be improved to support several address regions, limit it to 1
            // for now
            numRegions = 1;
            for (size_t k = 0; numRegions > k; ++k)
            {
                auto gpu1Id = make_tuple(DevType::GPU, i);
                auto gpu2Id = make_tuple(DevType::GPU, j);
                auto hammock = topoRouting.GetRequestHammock(gpu1Id, gpu2Id, k);
                hammock.GetTraffic();
                for (auto it = hammock.begin(); hammock.end() != it; ++it)
                {
                    occTensor[it->GetId()][i][j][k] = it->Get<OccNumType>();
                }
            }
        }
    }
    sort(occTensor.begin(), occTensor.end());
    occTensor.erase(std::unique(occTensor.begin(), occTensor.end()), occTensor.end());

    boost::multi_array<bool, 3> nonZeroElements(boost::extents[numGPUs][numGPUs][maxNumRegions]);
    for (const auto &m: occTensor)
    {
        for (size_t i = 0; numGPUs > i; ++i)
        {
            for (size_t j = 0; numGPUs > j; ++j)
            {
                for (size_t k = 0; maxNumRegions > k; ++k)
                {
                    if (0 != m[i][j][k])
                    {
                        nonZeroElements[i][j][k] = true;
                    }
                }
            }
        }
    }

    vector<array<size_t, 3>> sinksAndSources;
    for (size_t i = 0; numGPUs > i; ++i)
    {
        for (size_t j = 0; numGPUs > j; ++j)
        {
            for (size_t k = 0; maxNumRegions > k; ++k)
            {
                if (nonZeroElements[i][j][k]) sinksAndSources.push_back({ i, j, k });
            }
        }
    }

    OclwpancyMatrix occMtx{ boost::extents[occTensor.size()][sinksAndSources.size()] };

    for (auto it = occTensor.cbegin(); occTensor.cend() != it; ++it)
    {
        for (size_t i = 0; numGPUs > i; ++i)
        {
            for (size_t j = 0; numGPUs > j; ++j)
            {
                for (size_t k = 0; maxNumRegions > k; ++k)
                {
                    array<size_t, 3> idx{ i, j, k };
                    auto idxIt = boost::find(sinksAndSources, idx);
                    if (sinksAndSources.end() != idxIt)
                    {
                        occMtx[distance(occTensor.cbegin(), it)]
                              [distance(sinksAndSources.begin(), idxIt)] = (*it)(idx);
                    }
                }
            }
        }
    }

    // searching for a solution
    CandidateVec solution;
    if (SolveOclwpancyEq(occMtx, solution))
    {
        for (auto i : solution.ones)
        {
            *outIt++ = sinksAndSources[i];
        }
    }
}

namespace
{
    TopoArch GetArch(const ::fabric* pTopology)
    {
        const ::node& node = pTopology->fabricnode(0);
        for (int swIdx = 0; swIdx < node.lwswitch_size(); swIdx++)
        {
            const ::lwSwitch& sw = node.lwswitch(swIdx);
            for (int apIdx = 0; apIdx < sw.access_size(); apIdx++)
            {
                const ::accessPort& ap = sw.access(apIdx);
                if (ap.reqrte_size() > 0 || ap.rsprte_size() > 0)
                    return TOPO_WILLOW;
                if (ap.rmappolicytable_size() > 0 ||
                    ap.ridroutetable_size() > 0 ||
                    ap.rlanroutetable_size() > 0)
                    return TOPO_LIMEROCK;
            }
            for (int tpIdx = 0; tpIdx < sw.trunk_size(); tpIdx++)
            {
                const ::trunkPort& tp = sw.trunk(tpIdx);
                if (tp.reqrte_size() > 0 || tp.rsprte_size() > 0)
                    return TOPO_WILLOW;
                if (tp.ridroutetable_size() > 0 ||
                    tp.rlanroutetable_size() > 0)
                    return TOPO_LIMEROCK;
            }
        }
        return TOPO_UNKNOWN;
    }
}

bool MaxTraffic(const string &outFileName, const ::fabric* pTopology)
{
    auto_ptr<TopologyRouting> topoRouting;
    
    switch (GetArch(pTopology))
    {
        case TOPO_WILLOW:
            topoRouting.reset(new TopologyRouting(pTopology->fabricnode(0)));
            break;
        case TOPO_LIMEROCK:
            topoRouting.reset(new TopologyRoutingLR(pTopology->fabricnode(0)));
            break;
        default:
            cout << "Unknown LwSwitch chip!\n";
            return false;
    }

    vector<array<size_t, 2>> respSolution;
    vector<array<size_t, 3>> reqSolution;

    SolveResp(*topoRouting, back_inserter(respSolution));
    SolveReq(*topoRouting, back_inserter(reqSolution));

    if (respSolution.empty() && reqSolution.empty())
    {
        cout << "Solution is not found.\n";
        return false;
    }

    unique_ptr<ostream> os;
    if (!outFileName.empty())
    {
        os = make_unique<ofstream>(outFileName);
        if (!*os)
        {
            cerr << "Cannot open \"" << outFileName << "\": " << strerror(errno) << '\n';
            return false;
        }
    }
    else
    {
        os = make_unique<ostream>(cout.rdbuf());
    }

    *os << "{\n";
    if (!respSolution.empty())
    {
        *os <<
            "    \"response\" :\n"
            "    ["
          ;
        for (auto it = respSolution.cbegin(); respSolution.cend() != it; ++it)
        {
            if (respSolution.cbegin() != it)
            {
                *os << ',';
            }
            *os <<
                "\n"
                "        { \"source\" : \"gpu" << (*it)[0] << "\", "
                          "\"sink\" : \"gpu" << (*it)[1] << "\" }"
              ;
        }
        *os <<
            "\n"
            "    ]"
          ;
    }

    if (!reqSolution.empty())
    {
        *os <<
            ",\n"
            "    \"request\" :\n"
            "    ["
          ;
        for (auto it = reqSolution.cbegin(); reqSolution.cend() != it; ++it)
        {
            if (reqSolution.cbegin() != it)
            {
                *os << ',';
            }
            *os <<
                "\n"
                "        { \"source\" : \"gpu" << (*it)[0] << "\", "
                          "\"sink\" : { \"node\" : \"gpu" << (*it)[1] << "\", "
                                       "\"addr_region\" : " << (*it)[2] << " }}"
              ;
        }
        *os <<
            "\n"
            "    ]"
          ;
    }
    *os << "\n}\n";

    // return true only if both solutions are present
    return !respSolution.empty() && !reqSolution.empty();
}
