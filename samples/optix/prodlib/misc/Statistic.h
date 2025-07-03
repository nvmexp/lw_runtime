// Copyright LWPU Corporation 2017
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

#include <float.h>
#include <math.h>

namespace prodlib {

// If I need a more accurate mean and variance() algorithm it can be found here:
// http://www.johndcook.com/standard_deviation.html
class Statistic
{
  public:
    Statistic() { reset(); }

    void reset()
    {
        m_sum        = 0;
        m_sumSquared = 0;
        m_min        = DBL_MAX;
        m_max        = -DBL_MAX;
        m_count      = 0;
    }

    void add( double val )
    {
        m_sum += val;
        m_sumSquared += val * val;
        m_count++;
        if( val < m_min )
            m_min = val;
        if( val > m_max )
            m_max = val;
    }

    size_t count() const { return m_count; }

    double sum() const { return m_sum; }

    double mean() const
    {
        if( m_count > 0 )
            return m_sum / m_count;
        else
            return 0;
    }

    double minimum() const
    {
        if( m_count > 0 )
            return m_min;
        else
            return 0;
    }

    double maximum() const
    {
        if( m_count > 0 )
            return m_max;
        else
            return 0;
    }

    double variance() const
    {
        if( m_count > 0 )
        {
            double m = mean();
            return ( m_sumSquared / m_count ) - m * m;
        }
        else
            return 0;
    }

    double stddev() const
    {
        if( m_count > 0 )
            return sqrt( variance() );
        else
            return 0;
    }

  private:
    double m_sum;
    double m_sumSquared;
    double m_min;
    double m_max;
    size_t m_count;
};

}  // namespace prodlib