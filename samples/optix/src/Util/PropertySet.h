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

#include <corelib/misc/String.h>
#include <prodlib/exceptions/Assert.h>

#include <iosfwd>
#include <map>
#include <string>

namespace optix {

class Serializer;
class Deserializer;

class PropertySet
{
  public:
    typedef std::map<std::string, std::string> PropertyMap;

    // Set a property.
    template <typename T>
    void set( const std::string& name, const T& value )
    {
        RT_ASSERT( !name.empty() );
        m_properties[name] = corelib::to_string( value );
    }

    // Set a property only if it is found.  Returns true if found, false otherwise.
    template <typename T>
    bool setIfDefined( const std::string& name, const T& value )
    {
        RT_ASSERT( !name.empty() );
        PropertyMap::iterator it = m_properties.find( name );
        if( it == m_properties.end() )
            return false;
        it->second = corelib::to_string( value );
        return true;
    }

    // Get a property with the given type, return the default value if
    // the property was not found in the set.
    template <typename T>
    T get( const std::string& name, const T& defaultvalue ) const
    {
        PropertyMap::const_iterator it = m_properties.find( name );
        if( it == m_properties.end() )
            return defaultvalue;
        return corelib::from_string<T>( it->second );
    }

    // Get a property with the given type, assert if
    // the property was not found in the set.
    template <typename T>
    T get( const std::string& name ) const
    {
        PropertyMap::const_iterator it = m_properties.find( name );
        if( it == m_properties.end() )
            throw prodlib::AssertionFailure( RT_EXCEPTION_INFO, "Property not found: " + name );
        return corelib::from_string<T>( it->second );
    }

    // Returns true if the property is defined in the set
    bool defined( const std::string& name ) const
    {
        PropertyMap::const_iterator it = m_properties.find( name );
        return it != m_properties.end();
    }

    // Print to stream
    void print( std::ostream& out, const std::string& indent ) const;

    // Serialize the property set.
    void serialize( Serializer& serializer ) const;

    // Deserialize the property set.
    void deserialize( Deserializer& deserializer );

    // Access the property map.
    const PropertyMap& getPropertyMap() const { return m_properties; }

    // Returns true if the map is empty
    bool empty() const { return m_properties.empty(); }

  private:
    PropertyMap m_properties;
};

}  // namespace optix
