/* Copyright (c) 2022, LWPU CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of LWPU CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef LW_UTIL_NPP_SIGNAL_ALLOCATORS_NPP_H
#define LW_UTIL_NPP_SIGNAL_ALLOCATORS_NPP_H


#include "Exceptions.h"

#include <npps.h>
#include <lwda_runtime.h>

namespace npp
{

    template <typename D>
    class SignalAllocator
    {
    };

    template<>
    class SignalAllocator<Npp8u>
    {
        public:
            static
            Npp8u *
            Malloc1D(size_t nSize)
            {
                Npp8u *pResult = nppsMalloc_8u(static_cast<int>(nSize));
                NPP_ASSERT(pResult != 0);

                return pResult;
            };

            static
            void
            Free1D(Npp8u *pValues)
            {
                nppsFree(pValues);
            };

            static
            void
            Copy1D(Npp8u *pDst, const Npp8u *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp8u),lwdaMemcpyDeviceToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy1D(Npp8u *pDst, const Npp8u *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp8u), lwdaMemcpyHostToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy1D(Npp8u *pDst, const Npp8u *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp8u), lwdaMemcpyDeviceToHost);
                NPP_ASSERT(lwdaSuccess == eResult);
            };
    };

    template<>
    class SignalAllocator<Npp16s>
    {
        public:
            static
            Npp16s *
            Malloc1D(size_t nSize)
            {
                Npp16s *pResult = nppsMalloc_16s(static_cast<int>(nSize));
                NPP_ASSERT(pResult != 0);

                return pResult;
            };

            static
            void
            Free1D(Npp16s *pValues)
            {
                nppsFree(pValues);
            };

            static
            void
            Copy1D(Npp16s *pDst, const Npp16s *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp16s),lwdaMemcpyDeviceToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy1D(Npp16s *pDst, const Npp16s *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp16s), lwdaMemcpyHostToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy1D(Npp16s *pDst, const Npp16s *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp16s), lwdaMemcpyDeviceToHost);
                NPP_ASSERT(lwdaSuccess == eResult);
            };
    };

    template<>
    class SignalAllocator<Npp16u>
    {
        public:
            static
            Npp16u *
            Malloc1D(size_t nSize)
            {
                Npp16u *pResult = nppsMalloc_16u(static_cast<int>(nSize));
                NPP_ASSERT(pResult != 0);

                return pResult;
            };

            static
            void
            Free1D(Npp16u *pValues)
            {
                nppsFree(pValues);
            };

            static
            void
            Copy1D(Npp16u *pDst, const Npp16u *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp16u),lwdaMemcpyDeviceToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy1D(Npp16u *pDst, const Npp16u *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp16u), lwdaMemcpyHostToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy1D(Npp16u *pDst, const Npp16u *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp16u), lwdaMemcpyDeviceToHost);
                NPP_ASSERT(lwdaSuccess == eResult);
            };
    };

    template<>
    class SignalAllocator<Npp16sc>
    {
        public:
            static
            Npp16sc *
            Malloc1D(size_t nSize)
            {
                Npp16sc *pResult = nppsMalloc_16sc(static_cast<int>(nSize));
                NPP_ASSERT(pResult != 0);

                return pResult;
            };

            static
            void
            Free1D(Npp16sc *pValues)
            {
                nppsFree(pValues);
            };

            static
            void
            Copy1D(Npp16sc *pDst, const Npp16sc *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp16sc),lwdaMemcpyDeviceToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy1D(Npp16sc *pDst, const Npp16sc *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp16sc), lwdaMemcpyHostToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy1D(Npp16sc *pDst, const Npp16sc *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp16sc), lwdaMemcpyDeviceToHost);
                NPP_ASSERT(lwdaSuccess == eResult);
            };
    };

    template<>
    class SignalAllocator<Npp32u>
    {
        public:
            static
            Npp32u *
            Malloc1D(size_t nSize)
            {
                Npp32u *pResult = nppsMalloc_32u(static_cast<int>(nSize));
                NPP_ASSERT(pResult != 0);

                return pResult;
            };

            static
            void
            Free1D(Npp32u *pValues)
            {
                nppsFree(pValues);
            };

            static
            void
            Copy1D(Npp32u *pDst, const Npp32u *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp32u),lwdaMemcpyDeviceToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy1D(Npp32u *pDst, const Npp32u *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp32u), lwdaMemcpyHostToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy1D(Npp32u *pDst, const Npp32u *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp32u), lwdaMemcpyDeviceToHost);
                NPP_ASSERT(lwdaSuccess == eResult);
            };
    };

    template<>
    class SignalAllocator<Npp32s>
    {
        public:
            static
            Npp32s *
            Malloc1D(size_t nSize)
            {
                Npp32s *pResult = nppsMalloc_32s(static_cast<int>(nSize));
                NPP_ASSERT(pResult != 0);

                return pResult;
            };

            static
            void
            Free1D(Npp32s *pValues)
            {
                nppsFree(pValues);
            };

            static
            void
            Copy1D(Npp32s *pDst, const Npp32s *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp32s),lwdaMemcpyDeviceToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy1D(Npp32s *pDst, const Npp32s *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp32s), lwdaMemcpyHostToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy1D(Npp32s *pDst, const Npp32s *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp32s), lwdaMemcpyDeviceToHost);
                NPP_ASSERT(lwdaSuccess == eResult);
            };
    };

    template<>
    class SignalAllocator<Npp32sc>
    {
        public:
            static
            Npp32sc *
            Malloc1D(size_t nSize)
            {
                Npp32sc *pResult = nppsMalloc_32sc(static_cast<int>(nSize));
                NPP_ASSERT(pResult != 0);

                return pResult;
            };

            static
            void
            Free1D(Npp32sc *pValues)
            {
                nppsFree(pValues);
            };

            static
            void
            Copy1D(Npp32sc *pDst, const Npp32sc *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp32sc),lwdaMemcpyDeviceToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy1D(Npp32sc *pDst, const Npp32sc *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp32sc), lwdaMemcpyHostToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy1D(Npp32sc *pDst, const Npp32sc *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp32sc), lwdaMemcpyDeviceToHost);
                NPP_ASSERT(lwdaSuccess == eResult);
            };
    };

    template<>
    class SignalAllocator<Npp32f>
    {
        public:
            static
            Npp32f *
            Malloc1D(size_t nSize)
            {
                Npp32f *pResult = nppsMalloc_32f(static_cast<int>(nSize));
                NPP_ASSERT(pResult != 0);

                return pResult;
            };

            static
            void
            Free1D(Npp32f *pValues)
            {
                nppsFree(pValues);
            };

            static
            void
            Copy1D(Npp32f *pDst, const Npp32f *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp32f),lwdaMemcpyDeviceToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy1D(Npp32f *pDst, const Npp32f *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp32f), lwdaMemcpyHostToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy1D(Npp32f *pDst, const Npp32f *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp32f), lwdaMemcpyDeviceToHost);
                NPP_ASSERT(lwdaSuccess == eResult);
            };
    };

    template<>
    class SignalAllocator<Npp32fc>
    {
        public:
            static
            Npp32fc *
            Malloc1D(size_t nSize)
            {
                Npp32fc *pResult = nppsMalloc_32fc(static_cast<int>(nSize));
                NPP_ASSERT(pResult != 0);

                return pResult;
            };

            static
            void
            Free1D(Npp32fc *pValues)
            {
                nppsFree(pValues);
            };

            static
            void
            Copy1D(Npp32fc *pDst, const Npp32fc *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp32fc),lwdaMemcpyDeviceToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy1D(Npp32fc *pDst, const Npp32fc *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp32fc), lwdaMemcpyHostToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy1D(Npp32fc *pDst, const Npp32fc *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp32fc), lwdaMemcpyDeviceToHost);
                NPP_ASSERT(lwdaSuccess == eResult);
            };
    };

    template<>
    class SignalAllocator<Npp64s>
    {
        public:
            static
            Npp64s *
            Malloc1D(size_t nSize)
            {
                Npp64s *pResult = nppsMalloc_64s(static_cast<int>(nSize));
                NPP_ASSERT(pResult != 0);

                return pResult;
            };

            static
            void
            Free1D(Npp64s *pValues)
            {
                nppsFree(pValues);
            };

            static
            void
            Copy1D(Npp64s *pDst, const Npp64s *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp64s),lwdaMemcpyDeviceToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy1D(Npp64s *pDst, const Npp64s *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp64s), lwdaMemcpyHostToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy1D(Npp64s *pDst, const Npp64s *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp64s), lwdaMemcpyDeviceToHost);
                NPP_ASSERT(lwdaSuccess == eResult);
            };
    };

    template<>
    class SignalAllocator<Npp64sc>
    {
        public:
            static
            Npp64sc *
            Malloc1D(size_t nSize)
            {
                Npp64sc *pResult = nppsMalloc_64sc(static_cast<int>(nSize));
                NPP_ASSERT(pResult != 0);

                return pResult;
            };

            static
            void
            Free1D(Npp64sc *pValues)
            {
                nppsFree(pValues);
            };

            static
            void
            Copy1D(Npp64sc *pDst, const Npp64sc *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp64sc),lwdaMemcpyDeviceToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy1D(Npp64sc *pDst, const Npp64sc *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp64sc), lwdaMemcpyHostToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy1D(Npp64sc *pDst, const Npp64sc *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp64sc), lwdaMemcpyDeviceToHost);
                NPP_ASSERT(lwdaSuccess == eResult);
            };
    };

    template<>
    class SignalAllocator<Npp64f>
    {
        public:
            static
            Npp64f *
            Malloc1D(size_t nSize)
            {
                Npp64f *pResult = nppsMalloc_64f(static_cast<int>(nSize));
                NPP_ASSERT(pResult != 0);

                return pResult;
            };

            static
            void
            Free1D(Npp64f *pValues)
            {
                nppsFree(pValues);
            };

            static
            void
            Copy1D(Npp64f *pDst, const Npp64f *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp64f),lwdaMemcpyDeviceToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy1D(Npp64f *pDst, const Npp64f *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp64f), lwdaMemcpyHostToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy1D(Npp64f *pDst, const Npp64f *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp64f), lwdaMemcpyDeviceToHost);
                NPP_ASSERT(lwdaSuccess == eResult);
            };
    };

    template<>
    class SignalAllocator<Npp64fc>
    {
        public:
            static
            Npp64fc *
            Malloc1D(size_t nSize)
            {
                Npp64fc *pResult = nppsMalloc_64fc(static_cast<int>(nSize));
                NPP_ASSERT(pResult != 0);

                return pResult;
            };

            static
            void
            Free1D(Npp64fc *pValues)
            {
                nppsFree(pValues);
            };

            static
            void
            Copy1D(Npp64fc *pDst, const Npp64fc *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp64fc),lwdaMemcpyDeviceToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            HostToDeviceCopy1D(Npp64fc *pDst, const Npp64fc *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp64fc), lwdaMemcpyHostToDevice);
                NPP_ASSERT(lwdaSuccess == eResult);
            };

            static
            void
            DeviceToHostCopy1D(Npp64fc *pDst, const Npp64fc *pSrc, size_t nSize)
            {
                lwdaError_t eResult;
                eResult = lwdaMemcpy(pDst, pSrc, nSize * sizeof(Npp64fc), lwdaMemcpyDeviceToHost);
                NPP_ASSERT(lwdaSuccess == eResult);
            };
    };
} // npp namespace

#endif // LW_UTIL_NPP_SIGNAL_ALLOCATORS_NPP_H
