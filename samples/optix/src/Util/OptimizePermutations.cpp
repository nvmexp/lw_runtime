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


#include <Util/OptimizePermutations.h>

#include <Util/ContainerAlgorithm.h>

#include <prodlib/exceptions/Assert.h>
#include <prodlib/system/Logger.h>

#include <algorithm>
#include <limits.h>


static const unsigned int WARP_FINISHED = UINT_MAX;

template <typename Elem_t>
OptimizePermutations<Elem_t>::OptimizePermutations( size_t                    length,
                                                    size_t                    max_population,
                                                    int                       temperature,
                                                    int                       patience,
                                                    std::vector<std::string>& vpcNames )
    : m_vpcNames( vpcNames )
    , m_length( length )
    , m_max_population( max_population )
    , m_temperature( temperature )
    , m_patience( patience )
    , m_bad_times( 0 )
    , m_target_frames( 1 )
    , m_lwr_frames( 0 )
    , m_lwr_time( 0.0f )
{
}
template OptimizePermutations<int>::OptimizePermutations( size_t                    length,
                                                          size_t                    max_population,
                                                          int                       temperature,
                                                          int                       patience,
                                                          std::vector<std::string>& vpcNames );

template <typename Elem_t>
void OptimizePermutations<Elem_t>::getTestVector( TestVector& vec )
{
    if( !m_lwrrentTestVector.empty() )
    {
        vec = m_lwrrentTestVector;
        return;
    }

    if( m_population.empty() )
    {
        fillTestVectorSequential( vec );
        fillTestVectorHardCoded( vec );  // If this one is the right size it will overwrite sequential
    }
    else
    {
        // Create a TestVector to return
        // Randomly choose a TestResult from the population and do a random perturbation to it
        int i = rand() % m_population.size();

        vec = m_population[i].second;  // Copy the TestVector into the caller's copy

        if( rand() % 64 == 0 )  // Occasionally retest the 1:1 mapping or a permutation of it
            fillTestVectorSequential( vec );
        if( rand() % 64 == 0 )  // Occasionally retest the hardcoded mapping or a permutation of it
            fillTestVectorHardCoded( vec );
        if( rand() % 64 )  // Occasionally retest a good one that we already have
            permuteTestVector( vec, m_temperature );
    }
    m_lwrrentTestVector = vec;
}
template void OptimizePermutations<int>::getTestVector( TestVector& vec );

template <typename Elem_t>
void OptimizePermutations<Elem_t>::insertTestResult( float time )
{
    RT_ASSERT( m_lwrrentTestVector.size() == m_length );

    // Average m_target_frames frames together before inserting the test result into the population
    m_lwr_time += time;

    if( ++m_lwr_frames >= m_target_frames )
    {
        if( m_lwr_time < 100.0f )
        {
            // Throw out this TestResult and adjust loop iterations for a constant test length
            m_target_frames++;
            lprint << "m_target_frames = " << m_target_frames << std::endl;
        }
        else
        {
            // Insert this TestResult into the population
            float avg_time = m_lwr_time / static_cast<float>( m_target_frames );

            TestResult trec = make_pair( avg_time, m_lwrrentTestVector );
            insertTestResult( trec );
            m_lwrrentTestVector.clear();
        }

        m_lwr_time   = 0;
        m_lwr_frames = 0;
    }
}
template void OptimizePermutations<int>::insertTestResult( float time );

template <typename Elem_t>
std::ostream& operator<<( std::ostream& ost, const std::vector<Elem_t>& vec )
{
    for( size_t i = 0; i < vec.size(); i++ )
        ost << vec[i] << ", ";

    return ost;
}

template <typename Elem_t>
void OptimizePermutations<Elem_t>::insertTestResult( const TestResult& tres )
{
    // If this TestVector already exists, just update the time
    for( typename Population_t::iterator i = m_population.begin(); i != m_population.end(); ++i )
    {
        if( i->second == tres.second )
        {
            i->first = i->first * 0.75f + tres.first * 0.25f;  // Don't count how many samples because that takes work. Instead assume about four.
            // i->first = std::min(i->first, tres.first); // Min is more stable than average.
            lprint << "\nAveraging: " << i->first << ": " << i->second << std::endl;
            optix::algorithm::sort( m_population );
            return;
        }
    }

    // If the TestResult is not an improvement, advance the annealing state.
    if( m_population.size() >= m_max_population && tres.first > m_population.back().first )
    {
        if( m_bad_times % 10 == 0 )
        {
            lprint << m_temperature << ',' << m_bad_times << ' ';
            lprint.flush();
        }

        if( ++m_bad_times >= m_patience )
        {
            // We haven't improved in a long time so decrease the temperature.
            m_bad_times = 0;

            if( --m_temperature <= 0 )
            {
                lprint << "\nOptimizePermutations finished.";
                printResults();

                m_temperature = static_cast<int>( m_length );  // Reset and start over because there's nothing better to do.
                m_target_frames++;
            }
            lprint << "m_temperature = " << m_temperature << std::endl;
        }
    }
    else
    {
        m_bad_times /= 2;  // Extend the time, but not all the way because variance in framerate makes that rarely finish.

        // Insert tres into the already sorted m_max_population.
        TestResult tmp = tres;
        for( typename Population_t::iterator i = m_population.begin(); i != m_population.end(); ++i )
        {
            if( tmp.first < i->first )
                swap( tmp, *i );
        }
        if( m_population.size() < m_max_population )
            m_population.push_back( tmp );

        printResults();
    }

    RT_ASSERT( m_population.size() <= m_max_population );
}

template <typename Elem_t>
void OptimizePermutations<Elem_t>::getBestTestVector( TestVector& vec ) const
{
    RT_ASSERT( !m_population.empty() );

    vec = m_population[0].second;
}
template void OptimizePermutations<int>::getBestTestVector( TestVector& vec ) const;

template <typename Elem_t>
void OptimizePermutations<Elem_t>::permuteTestVector( TestVector& vec, const int num_swaps ) const
{
    for( int swaps = 0; swaps < num_swaps; swaps++ )
    {
        // Swap some element with its neighbor
        int i;
        do
        {
            i = rand() % ( m_length - 1 );
        } while( i == 1 );
        int j = i ? i + 1 : 2;  // Don't swap element 1

        Elem_t tmp = vec[i];
        vec[i]     = vec[j];
        vec[j]     = tmp;
    }
}

template <typename Elem_t>
void OptimizePermutations<Elem_t>::fillTestVectorSequential( TestVector& vec ) const
{
    vec.resize( m_length );
    for( size_t i = 0; i < m_length; i++ )
        vec[i]    = static_cast<Elem_t>( i );
    vec[1]        = WARP_FINISHED;
}

template <typename Elem_t>
void OptimizePermutations<Elem_t>::fillTestVectorHardCoded( TestVector& vec ) const
{
    Elem_t HC[] = {13, -1, 12, 9, 99, 0,  17, 8, 2,  18, 14, 3,
                   10, 19, 11, 7, 21, 15, 20, 4, 16, 5,  6};  // path_tracer -m 3 GTX 690

    vec.resize( m_length );

    if( sizeof( HC ) / sizeof( Elem_t ) == m_length )
    {
        for( size_t i = 0; i < m_length; i++ )
            vec[i]    = HC[i];
        vec[1]        = WARP_FINISHED;
    }
}

template <typename Elem_t>
void OptimizePermutations<Elem_t>::printResults() const
{
#if 0
  lprint << std::endl;
  if(m_population.size() > 0) {
    const TestVector& TV = m_population[0].second;
    for(int i=0; i<m_length; i++) {
      int ind = TV[i];
      if(ind >= 0)
        lprint << m_vpcNames[ind] << std::endl;
      else
        lprint << "FINISHED" << std::endl;
    }
  }

  lprint << std::endl;
  for(typename Population_t::const_iterator i = m_population.begin(); i != m_population.end(); ++i) {
    lprint << (i - m_population.begin()) << ": " << i->first << ": " << i->second << std::endl;
  }
  lprint << std::endl;

  std::cerr << "DONE" << std::endl;
#endif
}
