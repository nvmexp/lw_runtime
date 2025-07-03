// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#pragma once

#include <iosfwd>
#include <string>
#include <utility>
#include <vector>

// Given a vector of integers, compute a permutation thereof that minimizes a cost function
// Uses a simulated annealing approach where fewer swaps are performed as the temperature cools

// Keeps a population of good candidates, chooses one at random, permutes it, and gives it to the caller to test.
// Gets back from the caller the TestVector and its run time to add to the population.

template <typename Elem_t>
class OptimizePermutations
{
  public:
    typedef std::vector<Elem_t> TestVector;
    typedef std::pair<float, TestVector> TestResult;  // first is the run time of this ordering

    OptimizePermutations( size_t length, size_t max_population, int temperature, int patience, std::vector<std::string>& vpcNames );

    void getTestVector( TestVector& );           // Get a new TestVector to try, and remember it for later
    void insertTestResult( float );              // Add the result time of m_lwrrentTestVector
    void insertTestResult( const TestResult& );  // Add a TestResult that was just tested

    void getBestTestVector( TestVector& vec ) const;

  private:
    void permuteTestVector( TestVector& vec, int num_swaps ) const;
    void fillTestVectorSequential( TestVector& vec ) const;
    void fillTestVectorHardCoded( TestVector& vec ) const;

    void printResults() const;

    typedef std::vector<TestResult> Population_t;

    Population_t m_population;  // Up to max_population of the best TestResults we have.

    TestVector m_lwrrentTestVector;  // A copy of the most recently handed out TestVector

    std::vector<std::string>& m_vpcNames;  // Names of each item in the TestVector

    size_t m_length;          // How many elements in all the TestVectors
    size_t m_max_population;  // How many TestResults to keep around and choose permutations of
    int    m_temperature;     // How many swaps to perform at each step. This gradually decreases.
    int    m_patience;       // How many attempted steps without improvement before giving up and decreasing temperature
    int    m_bad_times;      // How many tests have been done since the last improvement
    int    m_target_frames;  // How many frames to run per test
    int    m_lwr_frames;     // How many frames have been done in the current test
    float  m_lwr_time;       // Time so far in the current test
};

template <typename Elem_t>
std::ostream& operator<<( std::ostream&, const typename OptimizePermutations<Elem_t>::TestResult& );
