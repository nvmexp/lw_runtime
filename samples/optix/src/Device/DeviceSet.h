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

#include <string>
#include <vector>


namespace optix {

class Device;

class DeviceSet
{
  public:
    // A single device
    typedef int position;

    // Simple iteration over set bits.  Note: we may end up implementing something
    // faster.
    struct const_iterator
    {
      public:
        position operator*() const;
        const_iterator& operator++();
        position operator++( int );
        bool operator!=( const const_iterator& b ) const;
        bool operator==( const const_iterator& b ) const;

      private:
        friend class DeviceSet;
        const_iterator( const DeviceSet* parent, position pos );
        const DeviceSet* parent;
        position         pos;
    };

    // Constructors / destructor
    DeviceSet();
    DeviceSet( const DeviceSet& );
    DeviceSet( const const_iterator& );
    DeviceSet( const Device* );
    DeviceSet( const std::vector<Device*>& );
    DeviceSet( const std::vector<unsigned int>& allDeviceListIndices );
    explicit DeviceSet( int allDeviceListIndex );
    ~DeviceSet();

    // Manipulate individual devices
    void remove( const Device* );
    void insert( const Device* );

    // Union
    DeviceSet operator|( const DeviceSet& b ) const;
    DeviceSet& operator|=( const DeviceSet& b );

    // Intersection
    DeviceSet operator&( const DeviceSet& b ) const;
    DeviceSet& operator&=( const DeviceSet& b );

    // Difference
    DeviceSet operator-( const DeviceSet& b ) const;
    DeviceSet& operator-=( const DeviceSet& b );

    // Comparisons - exact
    bool operator==( const DeviceSet& b ) const;
    bool operator!=( const DeviceSet& b ) const;

    // Complement
    DeviceSet operator~() const;

    // Return the all-device-index of the N-th set device
    int operator[]( int n ) const;

    // Returns the position in the range of [0,count()) based on the all-device-index
    int getArrayPosition( int allDeviceListIndex ) const;

    // Non-empty intersection
    bool overlaps( const DeviceSet& b ) const;

    bool         empty() const;
    unsigned int count() const;
    bool isSet( int allDeviceListIndex ) const;
    bool isSet( const Device* ) const;

    void clear();

    // Human readable string in the form {empty}, {all} or {0,1,6}
    std::string toString() const;

    const_iterator begin() const;
    const_iterator end() const;


  private:
    unsigned int m_devices;
};
}
