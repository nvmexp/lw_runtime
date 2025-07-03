// Copyright LWPU Corporation 2008
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

template <typename T>
OptixResult KnobRegistry::setKnobTyped( const std::string& name, const T& value, T& oldValue, KnobBase::Source source, KnobBase::Source& oldSource )
{
    std::lock_guard<std::mutex> lock( m_mutex );

    auto range = m_knobs.equal_range( name );
    if( range.first == range.second )
        return OPTIX_ERROR_ILWALID_VALUE;

    for( auto it = range.first; it != range.second; ++it )
    {
        Knob<T>* knob = dynamic_cast<Knob<T>*>( it->second );
        if( !knob )
            return OPTIX_ERROR_ILWALID_VALUE;

        oldValue = knob->get();
        knob->set( value );
        oldSource = knob->getSource();
        knob->setSource( source );
    }

    return OPTIX_SUCCESS;
}
