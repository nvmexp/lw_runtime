/*
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
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

namespace optix_exp {

class OpaqueApiObject
{
  public:
    enum class ApiType : unsigned int
    {
        DeviceContext = 0x11111111,
        Module        = 0x22222222,
        ProgramGroup  = 0x33333333,
        Pipeline      = 0x44444444,
        Denoiser      = 0x55555555,
        Task          = 0x66666666,
        Unknown       = 0xFFFFFFFF
    };

    ApiType m_apiType = ApiType::Unknown;

    explicit OpaqueApiObject( ApiType t )
        : m_apiType( t )
    {
    }
    OpaqueApiObject(OpaqueApiObject&&) = default;
    OpaqueApiObject& operator=(OpaqueApiObject&&) = default;

    ~OpaqueApiObject() { m_apiType = ApiType::Unknown; }
};
}  // end namespace optix_exp
