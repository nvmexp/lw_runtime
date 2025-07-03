/*
* Copyright (c) 2017, LWPU CORPORATION.
*
* LWPU Corporation and its licensors retain all intellectual property and proprietary
* rights in and to this software, related documentation and any modifications thereto.
* Any use, reproduction, disclosure or distribution of this software and related
* documentation without an express license agreement from LWPU Corporation is strictly
* prohibited.
*
* TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
* INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
* PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
* SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
* LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
* BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
* INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
* SUCH DAMAGES
*/

#pragma once

#include <Memory/DeviceSpecificTable.h>

#include <rtcore/interface/types.h>

namespace optix {

// NOTE: This structure should match the layout of RtcInstance; it simply
// divides fields up into nested structs so we can use DeviceSpecificTable<>
struct InstanceDescriptorHost
{
    struct DeviceIndependent
    {
        float  transform[12];       // 4 row x 3 col, row-major affine transformation matrix
        Rtlw32 instanceId : 24;     /* Application supplied ID */
        Rtlw32 mask : 8;            /* Visibility mask */
        Rtlw32 instanceOffset : 24; /* SBT record offset */
        Rtlw32 flags : 8;           /* combinations of RtcInstanceFlags */
        DeviceIndependent()
            : transform()
            , instanceId( 0 )
            , mask( 0xff )
            , instanceOffset( 0 )
            , flags( 0 )
        {
        }
    } di;

    struct DeviceDependent
    {
        Rtlw64 accelOrTraversableHandle; /* Set with a RtcGpuVA to a bottom level acceleration
                                           or a RtcTraversableHandle.  If you set with a
                                           RtcGpuVA, you must also set the
                                           RtcBuildInputInstanceArray::accelType. */
    } dd;
};

static_assert( sizeof( InstanceDescriptorHost ) == sizeof( RtcInstance ),
               "Host and rtcore structure views must be the same size" );

using InstanceDescriptorTable = DeviceSpecificTable<InstanceDescriptorHost>;

}  // namespace optix
