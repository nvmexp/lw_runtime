#pragma once

/*
 * Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#define PB_NUM_CHARS 16

using namespace std;

class Timer
{
public:
    Timer()
    {
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        m_oneOverFreq = 1.0f / freq.QuadPart;
        start();
    }

    void start()
    {
        LARGE_INTEGER startTime;
        QueryPerformanceCounter(&startTime);
        m_startTime.QuadPart = startTime.QuadPart; // separate the function from the write to ensure atomicity
    }

    float getTime() const
    {
        LARGE_INTEGER endTime;
        QueryPerformanceCounter(&endTime);
        return (float(endTime.QuadPart - m_startTime.QuadPart) * m_oneOverFreq);
    }

    float getTimeAndReset()
    {
        LARGE_INTEGER endTime;
        QueryPerformanceCounter(&endTime);
        float time = (float(endTime.QuadPart - m_startTime.QuadPart) * m_oneOverFreq);

        m_startTime.QuadPart = endTime.QuadPart;
        return time;

    }

private:
    LARGE_INTEGER m_startTime;
    float m_oneOverFreq;
};

class Alarm
{
public:
    Alarm()
    {
        hTimer = CreateWaitableTimer(NULL, false, NULL);
    }

    ~Alarm()
    {
        CloseHandle(hTimer);
    }

    void Set(float _ms)
    {
        t.start();
        ms = _ms;
        LARGE_INTEGER li;

        // schedule for 1 ms less than the expected alarm time
        li.QuadPart = (LONGLONG)std::round(-10000 * (ms - 1));
        SetWaitableTimer(hTimer, &li, false, NULL, NULL, false);
    }

    float Wait() // returns how late we are
    {
        WaitForSingleObject(hTimer, INFINITE);

        // spin loop the remaining time
        float time;
        while ((time = 1000.0f * t.getTime()) < ms)
        {
#ifndef _WIN32  // Function doesn't seem to be present on WIN32
            YieldProcessor();
#endif
        }
        return time - ms;
    }

private:
    HANDLE hTimer;
    Timer t;
    float ms;
};

void CopyToClipboard(const string& str);
void PasteFromClipboard(string& str);

uint32_t CreateImage(const char* path, uint32_t *pWidth, uint32_t *pHeight, void **ppdata);
void DestroyImage(uint32_t ilTex);
void SaveImage(const char* path, unsigned int width, unsigned int height, void *data, bool hasAlpha = false);
