#pragma once

#include <Windows.h>
#include <Psapi.h>
#include <vector>
#include <stdint.h>
#include <assert.h>

class OverlayDetector
{
public:
    bool isOtherOverlayActive();

protected:
    std::vector<HMODULE> m_modulesEnumerated;
};