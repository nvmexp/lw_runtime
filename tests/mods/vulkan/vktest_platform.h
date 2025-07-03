/*
* LWIDIA_COPYRIGHT_BEGIN
*
* Copyright 2017-2021 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* LWIDIA_COPYRIGHT_END
*/

#pragma once

#if defined(__GNUC__) || defined(__RESHARPER__)
// GCC can type-check printf like 'Arguments' against the 'Format' string.
#define GNU_FORMAT_PRINTF(f, a) [[gnu::format(printf, f, a)]]
#else
#define GNU_FORMAT_PRINTF(f, a)
#endif

GNU_FORMAT_PRINTF(1, 2)
int PlatformPrintf
(
    const char * Format,
    ... //       Arguments
);

int PlatformVAPrintf
(
    const char * Format,
    va_list      RestOfArgs
);

void PlatformOnEntry();
void SetEarlyExit(uint32_t* earlyExit);

namespace SharedSysmem
{
    RC Initialize();
    void Shutdown();
}

namespace Xp
{
    enum OperatingSystem
    {
        OS_WINDOWS = 2,
        OS_LINUXRM = 4
    };
    OperatingSystem GetOperatingSystem();
    void* AllocOsEvent(UINT32 hClient, UINT32 hDevice);
    void BreakPoint();
    void FreeOsEvent(void* pFd, UINT32 hClient, UINT32 hDevice);
    RC WaitOsEvents
    (
       void**  pOsEvents,
       UINT32  numOsEvents,
       UINT32* pCompletedIndices,
       UINT32  maxCompleted,
       UINT32* pNumCompleted,
       FLOAT64 timeoutMs
    );
    void SetOsEvent(void* pFd);
    bool HasClientSideResman();
    UINT64 QueryPerformanceCounter();
    UINT64 QueryPerformanceFrequency();
    UINT64 GetWallTimeMS();
    UINT64 GetWallTimeNS();
    UINT64 GetWallTimeUS();
}

namespace Utility
{
    template<size_t StrSize>
    class HiddenStringColwerter
    {
    public:
        constexpr HiddenStringColwerter(const char *s)
        {
            HiddenColwert(s, m_Storage, StrSize);
        }

        constexpr const char* GetStorage() const
        {
            return m_Storage;
        }

        static constexpr void HiddenColwert(const char *src, char *dst, size_t size)
        {
            char op = 0x5a;
            for (size_t i = 0; i < size; i++)
            {
                dst[i] = src[i] ^ op;
                op += 0x21;
            }
        }

    private:
        char m_Storage[StrSize]{};
    };

    template<size_t StrSize>
    class HiddenStringStorage
    {
    public:
        HiddenStringStorage(const HiddenStringColwerter<StrSize> &colwerter)
        {
            for (size_t i = 0; i < StrSize; i++)
            {
                m_Storage[i] = colwerter.GetStorage()[i];
            }
        }

        operator char*()
        {
            if (!m_Loaded)
            {
                HiddenStringColwerter<StrSize>::HiddenColwert(m_Storage, m_Storage, StrSize-1);
                m_Storage[StrSize - 1] = 0;
                m_Loaded = true;
            }
            return m_Storage;
        }
        string operator+(string s)
        {
            if (!m_Loaded)
            {
                HiddenStringColwerter<StrSize>::HiddenColwert(m_Storage, m_Storage, StrSize-1);
                m_Storage[StrSize - 1] = 0;
                m_Loaded = true;
            }
            string ret = m_Storage + s;
            return ret;
        }
    private:
        char m_Storage[StrSize];
        bool m_Loaded = false;
    };

    template<size_t StrSize>
    constexpr auto MakeHiddenStringColwerter(const char(&str)[StrSize])
    {
        return HiddenStringColwerter<StrSize>(str);
    }
}
