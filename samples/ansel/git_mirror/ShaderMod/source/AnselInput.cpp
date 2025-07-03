#include "Config.h"
#include "UI.h"
#include "AnselServer.h"
#include "AnselInput.h"
#include "CommonStructs.h"
#include "Log.h"

namespace input
{
    unsigned int KeyboardState::getNumVKeys() const
    {
        return c_stateStorageSize * c_bitSizeOfUInt;
    }

    unsigned int KeyboardState::getNextKeyDown(unsigned int key) const
    {
        USHORT vkey = key == StartIterating ? 0 : (USHORT)key + 1;
        int word = vkey / c_bitSizeOfUInt, bit = vkey % c_bitSizeOfUInt;

        for (; word < c_stateStorageSize; ++word, bit = 0)
        {
            for (; bit < c_bitSizeOfUInt; ++bit)
            {
                if (m_keyDown[word] & (1ul << bit))
                {
                    return word * c_bitSizeOfUInt + bit;
                }
            }
        }

        return DoneIterating;
    }

    unsigned int MomentaryKeyboardState::getNextKeyStateChangedToUp(unsigned int key) const
    {
        USHORT vkey = key == StartIterating ? 0 : (USHORT)key + 1;
        int word = vkey / c_bitSizeOfUInt, bit = vkey % c_bitSizeOfUInt;

        for (; word < c_stateStorageSize; ++word, bit = 0)
        {
            for (; bit < c_bitSizeOfUInt; ++bit)
            {
                bool down = (m_keyDown[word] & (1ul << bit)) != 0;
                bool wasdown = (m_wasKeyDown[word] & (1ul << bit)) != 0;

                if (!down && wasdown)
                {
                    return word * c_bitSizeOfUInt + bit;
                }
            }
        }

        return DoneIterating;
    }

    unsigned int MomentaryKeyboardState::getNextKeyStateChangedToDown(unsigned int key) const
    {
        USHORT vkey = key == StartIterating ? 0 : (USHORT)key + 1;
        int word = vkey / c_bitSizeOfUInt, bit = vkey % c_bitSizeOfUInt;

        for (; word < c_stateStorageSize; ++word, bit = 0)
        {
            for (; bit < c_bitSizeOfUInt; ++bit)
            {
                bool down = (m_keyDown[word] & (1ul << bit)) != 0;
                bool wasdown = (m_wasKeyDown[word] & (1ul << bit)) != 0;

                if (down && !wasdown)
                {
                    return word * c_bitSizeOfUInt + bit;
                }
            }
        }

        return DoneIterating;
    }

    void KeyboardState::resetState()
    {
        for (int i = 0; i < c_stateStorageSize; ++i)
            m_keyDown[i] = 0;
    }

    void MomentaryKeyboardState::resetThis()
    {
        for (int i = 0; i < c_stateStorageSize; ++i)
            m_wasKeyDown[i] = 0;
    }

    void MomentaryKeyboardState::resetState()
    {
        KeyboardState::resetState();
        resetThis();
    }
    
    void MomentaryKeyboardState::consumeEvent(const KeyDownEvent& ev)
    {
        m_keyDown[ev.vkey / c_bitSizeOfUInt] |= (1ul << (ev.vkey % c_bitSizeOfUInt));

        if (ev.vkey == VK_LSHIFT || ev.vkey == VK_RSHIFT ||
            ev.vkey == VK_LCONTROL || ev.vkey == VK_RCONTROL ||
            ev.vkey == VK_LMENU || ev.vkey == VK_RMENU)
        {
            USHORT newKey;

            if (ev.vkey == VK_LSHIFT || ev.vkey == VK_RSHIFT)
                newKey = VK_SHIFT;
            else if (ev.vkey == VK_LCONTROL || ev.vkey == VK_RCONTROL)
                newKey = VK_CONTROL;
            else if (ev.vkey == VK_LMENU || ev.vkey == VK_RMENU)
                newKey = VK_MENU;

            m_keyDown[newKey / c_bitSizeOfUInt] |= (1ul << (newKey % c_bitSizeOfUInt));
        }
        else if (ev.vkey == VK_SHIFT || ev.vkey == VK_CONTROL || ev.vkey == VK_MENU)
        {
            USHORT newKeyL, newKeyR;

            if (ev.vkey == VK_SHIFT)
                newKeyL = VK_LSHIFT, newKeyR = VK_RSHIFT;
            else if (ev.vkey == VK_CONTROL)
                newKeyL = VK_LCONTROL, newKeyR = VK_RCONTROL;
            else if (ev.vkey == VK_MENU)
                newKeyL = VK_LMENU, newKeyR = VK_RMENU;

            m_keyDown[newKeyL / c_bitSizeOfUInt] |= (1ul << (newKeyL % c_bitSizeOfUInt));
            m_keyDown[newKeyR / c_bitSizeOfUInt] |= (1ul << (newKeyR % c_bitSizeOfUInt));
        }
    }

    void MomentaryKeyboardState::consumeEvent(const KeyUpEvent& ev)
    {
        m_keyDown[ev.vkey / c_bitSizeOfUInt] &= ~(1ul << (ev.vkey % c_bitSizeOfUInt));

        if (ev.vkey == VK_LSHIFT || ev.vkey == VK_RSHIFT ||
            ev.vkey == VK_LCONTROL || ev.vkey == VK_RCONTROL ||
            ev.vkey == VK_LMENU || ev.vkey == VK_RMENU)
        {
            USHORT newKey;

            if (ev.vkey == VK_LSHIFT || ev.vkey == VK_RSHIFT)
                newKey = VK_SHIFT;
            else if (ev.vkey == VK_LCONTROL || ev.vkey == VK_RCONTROL)
                newKey = VK_CONTROL;
            else if (ev.vkey == VK_LMENU || ev.vkey == VK_RMENU)
                newKey = VK_MENU;

            m_keyDown[newKey / c_bitSizeOfUInt] &= ~(1ul << (newKey % c_bitSizeOfUInt));
        }
        else if (ev.vkey == VK_SHIFT || ev.vkey == VK_CONTROL || ev.vkey == VK_MENU)
        {
            USHORT newKeyL, newKeyR;

            if (ev.vkey == VK_SHIFT)
                newKeyL = VK_LSHIFT, newKeyR = VK_RSHIFT;
            else if (ev.vkey == VK_CONTROL)
                newKeyL = VK_LCONTROL, newKeyR = VK_RCONTROL;
            else if (ev.vkey == VK_MENU)
                newKeyL = VK_LMENU, newKeyR = VK_RMENU;


            m_keyDown[newKeyL / c_bitSizeOfUInt] &= ~(1ul << (newKeyL % c_bitSizeOfUInt));
            m_keyDown[newKeyL / c_bitSizeOfUInt] &= ~(1ul << (newKeyL % c_bitSizeOfUInt));
        }
    }

    void MomentaryKeyboardState::updatePreviousState()
    {
        for (int i = 0; i < c_stateStorageSize; ++i)
        {
            m_wasKeyDown[i] = m_keyDown[i];
        }
    }

    void MomentaryKeyboardState::onKillFolws()
    {
        for (int i = 0; i < c_stateStorageSize; ++i)
        {
            m_wasKeyDown[i] = m_keyDown[i];
            m_keyDown[i] = 0;
        }
    }
    
    void MouseState::resetState()
    {
        for (int i = 0; i < EMouseButton::numButtons; ++i)
            m_buttonDown[i] = false;

        m_aclwmCoordsX = m_aclwmCoordsY = m_aclwmCoordsWheel = 0;
    }
    
    void MomentaryMouseState::resetThis()
    {
        for (int i = 0; i < EMouseButton::numButtons; ++i)
            m_wasButtonDown[i] = false;

        m_lastCoordsX = m_lastCoordsY = m_lastCoordsWheel = 0;
    }

    void MomentaryMouseState::resetState()
    {
        MouseState::resetState();
        resetThis();
    }

    void MomentaryMouseState::consumeEvent(const MouseButtonDownEvent& ev)
    {
        m_buttonDown[ev.btn.value] = true;

        m_lastCoordsX = ev.lastCoordsX, m_lastCoordsY = ev.lastCoordsY, m_lastCoordsWheel = ev.lastCoordsWheel;
        m_aclwmCoordsX += ev.lastCoordsX, m_aclwmCoordsY += ev.lastCoordsY, m_aclwmCoordsWheel += ev.lastCoordsWheel;
    }

    void MomentaryMouseState::consumeEvent(const MouseButtonUpEvent& ev)
    {
        m_buttonDown[ev.btn.value] = false;

        m_lastCoordsX = ev.lastCoordsX, m_lastCoordsY = ev.lastCoordsY, m_lastCoordsWheel = ev.lastCoordsWheel;
        m_aclwmCoordsX += ev.lastCoordsX, m_aclwmCoordsY += ev.lastCoordsY, m_aclwmCoordsWheel += ev.lastCoordsWheel;
    }

    void MomentaryMouseState::consumeEvent(const MouseMoveEvent& ev)
    {
        m_lastCoordsX = ev.lastCoordsX, m_lastCoordsY = ev.lastCoordsY, m_lastCoordsWheel = ev.lastCoordsWheel;
        m_aclwmCoordsX += ev.lastCoordsX, m_aclwmCoordsY += ev.lastCoordsY, m_aclwmCoordsWheel += ev.lastCoordsWheel;
    }

    void MomentaryMouseState::updatePreviousState()
    {
        for (int i = 0; i < EMouseButton::numButtons; ++i)
        {
            m_wasButtonDown[i] = m_buttonDown[i];
        }

        m_lastCoordsX = m_lastCoordsY = m_lastCoordsWheel = 0;
    }

    void MomentaryMouseState::onKillFolws()
    {
        for (int i = 0; i < EMouseButton::numButtons; ++i)
        {
            m_wasButtonDown[i] = m_buttonDown[i];
            m_buttonDown[i] = false;
        }
    }
    
    float GamepadState::removeBacklash(float value, float backlashGap)
    {
        float absValue = std::abs(value);
        if (absValue < backlashGap)
        {
            value = 0.0f;
        }
        else
        {
            value = (value < 0.0f ? -1.0f : 1.0f) * (absValue - backlashGap) / (1.0f - backlashGap);
        }
        return value;
    }
    
    void GamepadState::resetState()
    {
        for (int i = 0; i < EGamepadButton::numButtons; ++i)
            m_buttonDown[i] = false;

        m_dpadState.value = EDPadDirection::kCenter;

        m_axisLX = 0;
        m_axisLY = 0;
        m_axisZ = 0;
        m_axisRX = 0;
        m_axisRY = 0;
    }

    void MomentaryGamepadState::resetThis()
    {
        for (int i = 0; i < EGamepadButton::numButtons; ++i)
            m_wasButtonDown[i] = false;

        m_dpadStateWas.value = EDPadDirection::kCenter;
    }

    void MomentaryGamepadState::resetState()
    {
        GamepadState::resetState();
        resetThis();
    }
    void MomentaryGamepadState::consumeEvent(const GamepadStateUpdateEvent& ev)
    {
        for (int i = 0; i < EGamepadButton::numButtons; ++i)
        {
            m_buttonDown[i] = ev.buttonDown[i];
        }

        m_dpadState.value = ev.dpadState.value;

        m_axisLX = ev.axisLX;
        m_axisLY = ev.axisLY;
        m_axisZ = ev.axisZ;
        m_axisRX = ev.axisRX;
        m_axisRY = ev.axisRY;
    }

    void MomentaryGamepadState::updatePreviousState()
    {
        for (int i = 0; i < EGamepadButton::numButtons; ++i)
        {
            m_wasButtonDown[i] = m_buttonDown[i];
        }

        m_dpadStateWas.value = m_dpadState.value;
    }

    void MomentaryGamepadState::onKillFolws()
    {
        for (int i = 0; i < EGamepadButton::numButtons; ++i)
        {
            m_wasButtonDown[i] = m_buttonDown[i];
            m_buttonDown[i] = false;
        }

        m_dpadStateWas.value = m_dpadState.value;
        m_dpadState.value = EDPadDirection::kCenter;

        m_axisLX = 0;
        m_axisLY = 0;
        m_axisZ = 0;
        m_axisRX = 0;
        m_axisRY = 0;
    }

    void InputEventQueue::initialize()
    {
        if (!InitializeCriticalSectionAndSpinCount(&m_lock, 0x00000400))
        {
            LOG_ERROR(LogChannel::kInput, "InitializeCriticalSectionAndSpinCount failed !");
            return;
        }
    }

    void InputEventQueue::destroy()
    {
        if (m_bIsValid)
        {
            DeleteCriticalSection(&m_lock);
            m_bIsValid = false;
        }
    }

    //call from produce thread
    unsigned char* InputEventQueue::beginInsertRawInput(unsigned int size)
    {
        assert(m_bIsValid);

        EnterCriticalSection(&m_lock);
        size_t oldSize = m_bproduce->size();
        m_bproduce->resize(oldSize + size);

        return m_bproduce->data() + oldSize;
    }

    void InputEventQueue::endInsertRawInput(bool undo)
    {
        if (undo)
        {
            size_t oldSize = m_sproduce->size() ? m_sproduce->back() : 0;
            m_bproduce->resize(oldSize);
        }
        else
        {
            m_sproduce->push_back((unsigned int)m_bproduce->size());
        }

        LeaveCriticalSection(&m_lock);
    }
    
    InputEventQueue::RawData InputEventQueue::swapAndConsume()
    {
        assert(m_bIsValid);

        m_bconsume->resize(0);
        m_sconsume->resize(0);

        EnterCriticalSection(&m_lock);

        std::vector<unsigned char>* btemp = m_bconsume;
        std::vector<unsigned int>* stemp = m_sconsume;

        m_bconsume = m_bproduce;
        m_sconsume = m_sproduce;

        m_bproduce = btemp;
        m_sproduce = stemp;

        LeaveCriticalSection(&m_lock);

        RawData ret;

        ret.numEvents = (unsigned int)m_sconsume->size();
        ret.eventsData = m_bconsume->data();
        ret.eventsOffsets = m_sconsume->data();

        return ret;
    }
    
    EGamepadButton::Enum GamepadDevice::translateButton(unsigned long idx) const
    {
        const unsigned int maxUsages = 16;

        EGamepadButton::Enum mapping_xbox[maxUsages] = { EGamepadButton::numButtons, EGamepadButton::kA, EGamepadButton::kB,
            EGamepadButton::kX, EGamepadButton::kY, EGamepadButton::kLeftShoulder, EGamepadButton::kRightShoulder,
            EGamepadButton::numButtons, EGamepadButton::numButtons, EGamepadButton::kLeftStickPress, EGamepadButton::kRightStickPress,
            EGamepadButton::numButtons, EGamepadButton::numButtons, EGamepadButton::numButtons, EGamepadButton::numButtons,
            EGamepadButton::numButtons };

        EGamepadButton::Enum mapping_dualshock4[maxUsages] = { EGamepadButton::numButtons, EGamepadButton::kX, EGamepadButton::kA,
            EGamepadButton::kB, EGamepadButton::kY, EGamepadButton::kLeftShoulder, EGamepadButton::kRightShoulder,
            EGamepadButton::numButtons, EGamepadButton::numButtons, EGamepadButton::numButtons, EGamepadButton::numButtons,
            EGamepadButton::kLeftStickPress, EGamepadButton::kRightStickPress, EGamepadButton::numButtons, EGamepadButton::numButtons,
            EGamepadButton::numButtons };

        EGamepadButton::Enum mapping_shield[maxUsages] = { EGamepadButton::numButtons, EGamepadButton::kA, EGamepadButton::kB,
            EGamepadButton::numButtons, EGamepadButton::kX, EGamepadButton::kY, EGamepadButton::numButtons,
            EGamepadButton::kLeftShoulder, EGamepadButton::kRightShoulder, EGamepadButton::numButtons, EGamepadButton::numButtons,
            EGamepadButton::numButtons, EGamepadButton::numButtons, EGamepadButton::numButtons, EGamepadButton::kLeftStickPress,
            EGamepadButton::kRightStickPress };

        EGamepadButton ret;
        ret.value = EGamepadButton::numButtons;

        if (idx < maxUsages)
        {
            if (m_deviceType == EGamepadDevice::kXbox360 || m_deviceType == EGamepadDevice::kXboxOne)
            {
                ret.value = mapping_xbox[idx];
            }
            else if (m_deviceType == EGamepadDevice::kDualShock4)
            {
                ret.value = mapping_dualshock4[idx];
            }
            else if (m_deviceType == EGamepadDevice::kShield)
            {
                ret.value = mapping_shield[idx];
            }
            else
            {
                ret.value = mapping_xbox[idx]; //for other devices, assume xbox-style
            }
        }
    
        return ret.value;
    }

    void GamepadDevice::translateAxes(GamepadStateUpdateEvent& evt, const ULONG* values) const
    {
        for (unsigned int naxis = 0; naxis < m_numAxes; ++naxis)
        {
            ULONG value = values[naxis];
            unsigned int valueCapIdx = m_axisToValueCap[naxis];
        
            LONG logicalMin = m_hidValueCaps[valueCapIdx].LogicalMin;
            LONG logicalMax = m_hidValueCaps[valueCapIdx].LogicalMax;
            
            if (m_axisType[naxis] != EGamepadAxis::kDpad)
            {
                int bitWidth = m_hidValueCaps[valueCapIdx].BitSize;
            
                unsigned long range;

                if (logicalMin < logicalMax)
                {
                    range = 1ull + logicalMax - logicalMin;
                }
                else
                {
                    range = 1ull << bitWidth;
                }

                assert(range > 0 && range <= 65536);
                long midPoint = range / 2;
                long scaleToShort = 65536l / range;
                short scaledValue = (short)(scaleToShort * ((long)value - logicalMin - midPoint));

                switch (m_axisType[naxis])
                {
                case EGamepadAxis::kLeftStickX:
                    evt.axisLX = scaledValue;
                    break;
                case EGamepadAxis::kLeftStickY:
                    evt.axisLY = scaledValue;
                    break;
                case EGamepadAxis::kRightStickX:
                    evt.axisRX = scaledValue;
                    break;
                case EGamepadAxis::kRightStickY:
                    evt.axisRY = scaledValue;
                    break;
                case EGamepadAxis::kBothTriggers:
                    evt.axisZ = scaledValue;
                    break;
                case EGamepadAxis::kLeftTrigger:
                    evt.axisZ += scaledValue / 2;
                    break;
                case EGamepadAxis::kRightTrigger:
                    evt.axisZ -= scaledValue / 2;
                    break;
                }
            }
            else
            {
                if ((long)value < logicalMin || (long)value > logicalMax)
                {
                    evt.dpadState.value = EDPadDirection::kCenter;
                }
                else
                {
                    evt.dpadState.value = (EDPadDirection::Enum) (value - logicalMin);
                }
            }
        }
    }

    GamepadDevice::EGamepadDevice GamepadDevice::deviceFromDeviceInfo(DWORD dwVendorId, DWORD dwProductId, DWORD dwVersionNumber)
    {
        struct DeviceMapping
        {
            EGamepadDevice dev;
            DWORD dwVendorId;
            DWORD dwProductId;
            DWORD dwVersionNumber;
        };

        static const int numEntries = 8;
        DeviceMapping devs[numEntries] =
        {
            { EGamepadDevice::kXboxOne, 0x0000045e, 0x000002d1, 0 },
            { EGamepadDevice::kXboxOne, 0x0000045e, 0x000002ff, 0 },
            { EGamepadDevice::kShield, 0x00000955, 0x00007210, 0x100 },
            { EGamepadDevice::kXbox360, 0x0000045e, 0x000002a1, 0 },
            { EGamepadDevice::kXbox360, 0x0000045e, 0x0000028e, 0 },
            { EGamepadDevice::kXboxOne, 0x0000046d, 0x0000c21f, 0 },//logitech  F710 Wireless Gamepad [XInput Mode] 
            { EGamepadDevice::kDualShock4, 0x0000054c, 0x000005c4, 0x100 },//DualShock 4 
            { EGamepadDevice::kDualShock4, 0x0000054c, 0x000009CC, 0x100 },//DualShock 4 (LWH-ZCT2)
        };
        // EA Sports Controller 0x0E6F > 0x0131, 0x0401, 0x011F, 0x0133
        for (int i = 0; i < numEntries; ++i)
        {
            //Let's ignore the version number for now..

            if (dwVendorId == devs[i].dwVendorId &&
                dwProductId == devs[i].dwProductId)

                return devs[i].dev;
        }

        // The gamepad is not supported
        LOG_WARN(LogChannel::kInput_RawInput_Gamepad, "Unsupported controller detected (0x%08x/0x%08x/0x%08x)", dwVendorId, dwProductId, dwVersionNumber);

        return EGamepadDevice::kUnknown;
    }

    void GamepadDevice::initialize(HANDLE hGamepad, DWORD dwVendorId, DWORD dwProductId, DWORD dwVersionNumber)
    {
        m_statsForTelemetry.dwProductId = m_dwProductId = dwProductId;
        m_statsForTelemetry.dwVendorId = m_dwVendorId = dwVendorId;
        m_statsForTelemetry.dwVersionNumber = m_dwVersionNumber = dwVersionNumber;
        GamepadDevice::EGamepadDevice devType = deviceFromDeviceInfo(dwVendorId, dwProductId, dwVersionNumber);
        m_statsForTelemetry.type = devType;

        m_handle = 0;
        
        UINT bufferSize;
        GetRawInputDeviceInfo(hGamepad, RIDI_PREPARSEDDATA, NULL, &bufferSize);
        m_hidPreparsedData.resize(bufferSize);
        PHIDP_PREPARSED_DATA preparsedData = (PHIDP_PREPARSED_DATA)&(m_hidPreparsedData[0]);

        if (GetRawInputDeviceInfo(hGamepad, RIDI_PREPARSEDDATA, preparsedData, &bufferSize) != m_hidPreparsedData.size())
        {
            LOG_ERROR(LogChannel::kInput_RawInput_Gamepad, "GetRawInputDeviceInfo does not return correct size !");

            return;
        }

        // Button caps
        if (HidP_GetCaps(preparsedData, &m_hidCaps) != HIDP_STATUS_SUCCESS)
        {
            LOG_ERROR(LogChannel::kInput_RawInput_Gamepad, "Failed to get hid caps!");

            return;
        }

        m_hidButtonCaps.resize(m_hidCaps.NumberInputButtonCaps);
        USHORT capsLength = m_hidCaps.NumberInputButtonCaps;

        if (HidP_GetButtonCaps(HidP_Input, &(m_hidButtonCaps[0]), &capsLength, preparsedData) != HIDP_STATUS_SUCCESS)
        {
            LOG_ERROR(LogChannel::kInput_RawInput_Gamepad, "Failed HidP_GetButtonCaps!");

            return;
        }

        if (capsLength != m_hidCaps.NumberInputButtonCaps)
        {
            LOG_ERROR(LogChannel::kInput_RawInput_Gamepad, "HidP_GetButtonCaps returned too few button caps!");

            return;
        }

        // Value caps
        m_hidValueCaps.resize(m_hidCaps.NumberInputValueCaps);
        capsLength = m_hidCaps.NumberInputValueCaps;

        if (HidP_GetValueCaps(HidP_Input, &(m_hidValueCaps[0]), &capsLength, preparsedData) != HIDP_STATUS_SUCCESS)
        {
            LOG_ERROR(LogChannel::kInput_RawInput_Gamepad, "Failed HidP_GetValueCaps!");

            return;
        }

        if (capsLength != m_hidCaps.NumberInputValueCaps)
        {
            LOG_ERROR(LogChannel::kInput_RawInput_Gamepad, "HidP_GetValueCaps returned too few button caps!");
        
            return;
        }
            
        m_deviceType = devType;

        unsigned short xb_axisUsages[(int)EGamepadAxis::numAxes] = { 0x30, 0x31, 0x33, 0x34, 0x32, 0x35, 0xFFff, 0x39 };
        unsigned short dualshock4_axisUsages[(int)EGamepadAxis::numAxes] = { 0x30, 0x31, 0x32, 0x35, 0x33, 0x34, 0xFFff, 0x39 };
        unsigned short shield_axisUsages[(int)EGamepadAxis::numAxes] = { 0x30, 0x31, 0x32, 0x35, 0xc5, 0xc4, 0xFFff, 0x39 };

        unsigned short* lwrrentControllerAxisUsages = 0;
                
        if (m_deviceType == EGamepadDevice::kShield)
        {
            lwrrentControllerAxisUsages = shield_axisUsages;
        }
        else  if (m_deviceType == EGamepadDevice::kDualShock4)
        {
            lwrrentControllerAxisUsages = dualshock4_axisUsages;
        }
        else  if (m_deviceType == EGamepadDevice::kXboxOne || m_deviceType == EGamepadDevice::kXbox360)
        {
            lwrrentControllerAxisUsages = xb_axisUsages;
        }
        else
        {
            lwrrentControllerAxisUsages = xb_axisUsages; // try xbox standard for unknown devices
        }

        m_numButtons = 0;

        m_buttonUsagePages.resize(0);
        m_buttonUsagePageButtonCount.resize(0);

        for (size_t i = 0u; i < m_hidCaps.NumberInputButtonCaps; ++i)
        {
            int usagePageIndex = -1;

            for (size_t j = 0u; j < m_buttonUsagePages.size(); ++j)
            {
                if (m_buttonUsagePages[j] == m_hidButtonCaps[i].UsagePage)
                {
                    usagePageIndex = int(j);

                    break;
                }
            }

            if (usagePageIndex < 0)
            {
                usagePageIndex = (int)m_buttonUsagePages.size();
                m_buttonUsagePages.push_back(m_hidButtonCaps[i].UsagePage);
                m_buttonUsagePageButtonCount.push_back(0);
            }

            unsigned int n = m_hidButtonCaps[i].IsRange ? m_hidButtonCaps[i].Range.UsageMax - m_hidButtonCaps[i].Range.UsageMin + 1 : 1;

            m_numButtons += n;
            m_buttonUsagePageButtonCount[usagePageIndex] += n;
        }

        m_numAxes = 0;
        bool rightZfound = false;
        int leftZIndex = -1;

        for (int i = 0; i < (int)m_hidCaps.NumberInputValueCaps && m_numAxes < (int) EGamepadAxis::numAxes; ++i)
        {
            for (USAGE u = m_hidValueCaps[i].IsRange ? m_hidValueCaps[i].Range.UsageMin : m_hidValueCaps[i].NotRange.Usage,
                 maxu = m_hidValueCaps[i].IsRange ? m_hidValueCaps[i].Range.UsageMax : m_hidValueCaps[i].NotRange.Usage;
                 (unsigned long) u <= (unsigned long) maxu; u++)
            {
                for (int j = 0; j < (int)EGamepadAxis::numAxes; ++j)
                {
                    if (u == lwrrentControllerAxisUsages[j])
                    {
                        int axisIdx = m_numAxes++;
                        m_axisType[axisIdx] = (EGamepadAxis)j;
                        m_axisToValueCap[axisIdx] = i;
                        m_axisUsages[axisIdx] = u;

                        if ((EGamepadAxis)j == EGamepadAxis::kRightTrigger)
                            rightZfound = true;

                        if ((EGamepadAxis)j == EGamepadAxis::kLeftTrigger)
                            leftZIndex = axisIdx;

                        break;
                    }
                }
            }
        }

        if (!rightZfound && leftZIndex >= 0)
        {
            m_axisType[leftZIndex] = EGamepadAxis::kBothTriggers;
        }
        
        m_handle = hGamepad;

        return;
    }

    void GamepadDevice::getStats(GamepadStats& stats) const
    {
        stats = m_statsForTelemetry;
    }

    void RawInputManager::firstTimeInit(HWND pumpWnd, bool checkRIDs)
    {
        m_bSavedGameKeyboard = false;
        m_bSavedGameMouse = false;
        m_bSavedGameGamepad = false;
        m_bSavedGameJoystick = false;

        m_bInstalledAnselKeyboard = false;
        m_bInstalledAnselMouse = false;
        m_bInstalledAnselGamepad = false;
        m_bInstalledAnselJoystick = false;

        m_pumpWindow = pumpWnd;

        resetEventReceivedFlags();

        if (checkRIDs)
        {
            checkRawInputDevices();
        }
        else
        {
            m_bCheckRIDsNextFrame = true;
        }

        m_framesToCheckDeviceChangeRemaining = framesToCheckDeviceChange;
        m_framesToCheckRidsRemaining = framesToCheckRawInput;
    }

    void RawInputManager::deinit()
    {
        m_pumpWindow = 0;

        resetEventReceivedFlags();
        removeRestoreRawInputDevices();

        m_hkeyboard = 0;
        m_hmouse = 0;
        m_gamepad.ilwalidate();
    }

    void RawInputManager::tick(bool haveFolws, bool allowCheckingRawInputDevices)
    {
        if (haveFolws)
        {
            --m_framesToCheckDeviceChangeRemaining;

            if (allowCheckingRawInputDevices)
                --m_framesToCheckRidsRemaining;

            if (m_bCheckRIDsNextFrame || (allowCheckingRawInputDevices && m_framesToCheckRidsRemaining <= 0))
            {
                checkRawInputDevices();
                m_bCheckRIDsNextFrame = false;
                m_framesToCheckRidsRemaining = framesToCheckRawInput;
            }

            if (m_framesToCheckDeviceChangeRemaining <= 0)
            {
                checkDeviceHandles();
                resetEventReceivedFlags();
                m_framesToCheckDeviceChangeRemaining = framesToCheckDeviceChange;
            }
        }
    }
        
    bool RawInputManager::setGamepad(HANDLE gamepad)
    {
        RID_DEVICE_INFO rdi;
        rdi.cbSize = sizeof(RID_DEVICE_INFO);

        UINT cbSize = rdi.cbSize;

        if (GetRawInputDeviceInfo(gamepad, RIDI_DEVICEINFO, &rdi, &cbSize) == UINT(-1))
        {
            LOG_ERROR(LogChannel::kInput_RawInput_Gamepad, "Failed GetRawInputDeviceInfo!");

            return false;
        }

        if (rdi.dwType != RIM_TYPEHID)
            return false;

        //!!! the device may be kUnkown - in this case, this code will try to treat it as XBox
        m_gamepad.initialize(gamepad, rdi.hid.dwVendorId, rdi.hid.dwProductId, rdi.hid.dwVersionNumber);
        
        return true;
    }
        
    void RawInputManager::resetEventReceivedFlags()
    {
        m_selectedMouseEventRegistered = false;
        m_unselectedMouseEventRegistered = false;
        m_selectedKeyboardEventRegistered = false;
        m_unselectedKeyboardEventRegistered = false;
        m_selectedGamepadEventRegistered = false;
        m_unselectedGamepadEventRegistered = false;
    }

    void RawInputManager::checkDeviceHandles()
    {
        if (m_hmouse && !m_selectedMouseEventRegistered &&  m_unselectedMouseEventRegistered)
            m_hmouse = 0;

        if (m_hkeyboard && !m_selectedKeyboardEventRegistered && m_unselectedKeyboardEventRegistered)
            m_hkeyboard = 0;

        if (m_gamepad.isInitialzied() && !m_selectedGamepadEventRegistered &&   m_unselectedGamepadEventRegistered)
            m_gamepad.ilwalidate();
    }

    bool RawInputManager::checkRawInputDevices()
    {
        HWND pumpWindow = m_pumpWindow;
        UINT numDevices = 0;

        GetRegisteredRawInputDevices(nullptr, &numDevices, sizeof(RAWINPUTDEVICE));
        std::vector<RAWINPUTDEVICE> rids;
        rids.resize(numDevices);

        if (rids.size())
        {
            if (GetRegisteredRawInputDevices(&(rids[0]), &numDevices, sizeof(RAWINPUTDEVICE)) != rids.size())
            {
                LOG_ERROR(LogChannel::kInput_RawInput, "Failed GetRegisteredRawInputDevices!");

                return false;
            }
        }

        int mouseAt = -1, keyboardAt = -1, gamepadAt = -1, joystickAt = -1;
        bool mouseReady = false, keyboardReady = false, gamepadReady = false, joystickReady = false;

        int debug_numMice = 0, debug_numKbd = 0, debug_numGamepads = 0, debug_numJoysticks = 0;

        for (size_t i = 0u; i < rids.size(); ++i)
        {
            if (rids[i].usUsagePage == 0x01 && rids[i].usUsage == 0x02)
            {
                mouseAt = int(i);
                debug_numMice++;

                mouseReady = true;
            }
            else if (rids[i].usUsagePage == 0x01 && rids[i].usUsage == 0x06)
            {
                keyboardAt = int(i);
                debug_numKbd++;

                keyboardReady = true;
            }
            else if (rids[i].usUsagePage == 0x01 && rids[i].usUsage == 0x05)
            {
                gamepadAt = int(i);
                debug_numGamepads++;

                gamepadReady = true;
            }
            else if (rids[i].usUsagePage == 0x01 && rids[i].usUsage == 0x04)
            {
                joystickAt = int(i);
                debug_numJoysticks++;

                joystickReady = true;
            }
        }

        if (debug_numMice > 1 || debug_numKbd > 1 || debug_numGamepads > 1 || debug_numJoysticks > 1)
        {
            LOG_VERBOSE(LogChannel::kInput_RawInput, "debug_numMice > 1 || debug_numKbd > 1 || debug_numGamepads > 1");
        }

        if (mouseReady && keyboardReady && gamepadReady && joystickReady)
            return true;

        int numExtraRids = (mouseAt < 0) * 1 + (keyboardAt < 0) * 1 + (gamepadAt < 0) * 1 + (joystickAt < 0) * 1;
        unsigned int newRidsPos = (unsigned int)rids.size();
        rids.resize(newRidsPos + numExtraRids);
                
        if (!mouseReady) //mouse not set by the game, or set (reset) to Ansel-unfriendly values
        {
            if (mouseAt >= 0) //means it wasn't bound to 0 hwnd and it wasn't an input sink
            {
                //we save it even if it is already saved, because it's clearly the game's new values overwritten by itself
                memcpy(&m_gameRidMouse, &(rids[mouseAt]), sizeof(RAWINPUTDEVICE));
                m_bSavedGameMouse = true;
                m_bInstalledAnselMouse = false;

                RAWINPUTDEVICE& r = rids[mouseAt];

                r.hwndTarget = 0; //let it follow the focus; hooks will block messages where we don't need them
            }
            else
            {
                //the game might not have installed, or may have removed the mouse
                m_bSavedGameMouse = false;
                m_bInstalledAnselMouse = true;

                mouseAt = newRidsPos++;
                RAWINPUTDEVICE& r = rids[mouseAt];
                r.hwndTarget = pumpWindow;
                r.dwFlags = RIDEV_INPUTSINK;
                r.usUsagePage = 0x01;
                r.usUsage = 0x02;
            }
        }

        if (!keyboardReady)
        {
            if (keyboardAt >= 0)
            {
                m_bSavedGameKeyboard= true;
                m_bInstalledAnselKeyboard = false;

                memcpy(&m_gameRidKeyboard, &(rids[keyboardAt]), sizeof(RAWINPUTDEVICE));
                m_bSavedGameKeyboard = true;

                RAWINPUTDEVICE& r = rids[keyboardAt];

                r.hwndTarget = 0;
            }
            else
            {
                m_bSavedGameKeyboard = false;
                m_bInstalledAnselKeyboard = true;

                keyboardAt = newRidsPos++;
                RAWINPUTDEVICE& r = rids[keyboardAt];
                r.hwndTarget = pumpWindow;
                r.dwFlags = RIDEV_INPUTSINK;
                r.usUsagePage = 0x01;
                r.usUsage = 0x06;
            }
        }

        if (!gamepadReady)
        {
            if (gamepadAt >= 0)
            {
                m_bSavedGameGamepad = true;
                m_bInstalledAnselGamepad = false;

                memcpy(&m_gameRidGamepad, &(rids[gamepadAt]), sizeof(RAWINPUTDEVICE));
                m_bSavedGameGamepad = true;

                RAWINPUTDEVICE& r = rids[gamepadAt];

                r.hwndTarget = 0;
            }
            else
            {
                m_bSavedGameGamepad = false;
                m_bInstalledAnselGamepad = true;

                gamepadAt = newRidsPos++;
                RAWINPUTDEVICE& r = rids[gamepadAt];
                r.hwndTarget = pumpWindow;
                r.dwFlags = RIDEV_INPUTSINK;
                r.usUsagePage = 0x01;
                r.usUsage = 0x05;
            }
        }

        if (!joystickReady)
        {
            if (joystickAt >= 0)
            {
                m_bSavedGameJoystick = true;
                m_bInstalledAnselJoystick = false;

                memcpy(&m_gameRidJoystick, &(rids[joystickAt]), sizeof(RAWINPUTDEVICE));
                m_bSavedGameJoystick = true;

                RAWINPUTDEVICE& r = rids[joystickAt];

                r.hwndTarget = 0;
            }
            else
            {
                m_bSavedGameJoystick = false;
                m_bInstalledAnselJoystick = true;

                joystickAt = newRidsPos++;
                RAWINPUTDEVICE& r = rids[joystickAt];
                r.hwndTarget = pumpWindow;
                r.dwFlags = RIDEV_INPUTSINK;
                r.usUsagePage = 0x01;
                r.usUsage = 0x04;
            }
        }

        //if the mouse was at least once not ready, it's either saved or installed. If neither happened, the game mouse was picked up, so save it 
        if (mouseAt >= 0 && !m_bSavedGameMouse && !m_bInstalledAnselMouse) //game installed mouse first time
        {
            memcpy(&m_gameRidMouse, &(rids[mouseAt]), sizeof(RAWINPUTDEVICE));
            m_bSavedGameMouse = true;
        }

        if (keyboardAt >= 0 && !m_bSavedGameKeyboard && !m_bInstalledAnselKeyboard) //game installed kbd first time
        {
            memcpy(&m_gameRidKeyboard, &(rids[keyboardAt]), sizeof(RAWINPUTDEVICE));
            m_bSavedGameKeyboard = true;
        }

        if (gamepadAt >= 0 && !m_bSavedGameGamepad && !m_bInstalledAnselGamepad) //game installed kbd first time
        {
            memcpy(&m_gameRidGamepad, &(rids[gamepadAt]), sizeof(RAWINPUTDEVICE));
            m_bSavedGameGamepad = true;
        }

        if (joystickAt >= 0 && !m_bSavedGameJoystick && !m_bInstalledAnselJoystick) //game installed kbd first time
        {
            memcpy(&m_gameRidJoystick, &(rids[joystickAt]), sizeof(RAWINPUTDEVICE));
            m_bSavedGameJoystick = true;
        }

        if (RegisterRawInputDevices(&(rids[0]), (UINT)rids.size(), sizeof(RAWINPUTDEVICE)) == FALSE)
        {
            LOG_ERROR(LogChannel::kInput_RawInput, "RegisterRawInputDevices failed!");

            return false;
        }

        return false;
    }

    void RawInputManager::removeRestoreRawInputDevices()
    {
        UINT numDevices = 0;

        GetRegisteredRawInputDevices(nullptr, &numDevices, sizeof(RAWINPUTDEVICE));
        std::vector<RAWINPUTDEVICE> rids;
        rids.resize(numDevices);

        if (!numDevices)
            return;

        HWND pumpWindow = m_pumpWindow;

        if (GetRegisteredRawInputDevices(&(rids[0]), &numDevices, sizeof(RAWINPUTDEVICE)) != rids.size())
        {
            LOG_ERROR(LogChannel::kInput_RawInput, "Failed GetRegisteredRawInputDevices!");

            return;
        }

        int mouseAt = -1, keyboardAt = -1, gamepadAt = -1, joystickAt = -1;
        bool mouseReady = false, keyboardReady = false, gamepadReady = false, joystickReady = false;

        int debug_numMice = 0, debug_numKbd = 0, debug_numGamepads = 0, debug_numJoysticks = 0;

        for (size_t i = 0u; i < rids.size(); ++i)
        {
            if (rids[i].usUsagePage == 0x01 && rids[i].usUsage == 0x02)
            {
                mouseAt = int(i);
                debug_numMice++;

                mouseReady = true;
            }
            else if (rids[i].usUsagePage == 0x01 && rids[i].usUsage == 0x06)
            {
                keyboardAt = int(i);
                debug_numKbd++;

                keyboardReady = true;
            }
            else if (rids[i].usUsagePage == 0x01 && rids[i].usUsage == 0x05)
            {
                gamepadAt = int(i);
                debug_numGamepads++;

                gamepadReady = true;
            }
            else if (rids[i].usUsagePage == 0x01 && rids[i].usUsage == 0x04)
            {
                joystickAt = int(i);
                debug_numJoysticks++;

                joystickReady = true;
            }
        }

        if (debug_numMice > 1 || debug_numKbd > 1 || debug_numGamepads > 1 || debug_numJoysticks > 1)
        {
            LOG_VERBOSE(LogChannel::kInput_RawInput, "debug_numMice > 1 || debug_numKbd > 1 || debug_numGamepads > 1");
        }

        if (mouseAt >= 0)
        {
            if (m_bSavedGameMouse)
            {
                memcpy(&(rids[mouseAt]), &m_gameRidMouse, sizeof(RAWINPUTDEVICE));
            }
            else if (mouseReady && m_bInstalledAnselMouse)//if the game set the mouse after we saved, and we can't distinguish it from ours, rely on flag 
            {
                RAWINPUTDEVICE& r = rids[mouseAt];
                r.dwFlags = RIDEV_REMOVE;
                r.hwndTarget = 0;
            }
        }

        if (keyboardAt >= 0)
        {
            if (m_bSavedGameKeyboard)
            {
                memcpy(&(rids[keyboardAt]), &m_gameRidKeyboard, sizeof(RAWINPUTDEVICE));
            }
            else if (keyboardReady && m_bInstalledAnselKeyboard) //this may fail if the game changes the keyboard to ansel-ready after we initialzied
            {
                RAWINPUTDEVICE& r = rids[keyboardAt];
                r.dwFlags = RIDEV_REMOVE;
                r.hwndTarget = 0;
            }
        }

        if (gamepadAt >= 0)
        {
            if (m_bSavedGameGamepad)
            {
                memcpy(&(rids[gamepadAt]), &m_gameRidGamepad, sizeof(RAWINPUTDEVICE));
            }
            else if (gamepadReady && m_bInstalledAnselGamepad)
            {
                RAWINPUTDEVICE& r = rids[gamepadAt];
                r.dwFlags = RIDEV_REMOVE;
                r.hwndTarget = 0;
            }
        }

        if (joystickAt >= 0)
        {
            if (m_bSavedGameJoystick)
            {
                memcpy(&(rids[joystickAt]), &m_gameRidJoystick, sizeof(RAWINPUTDEVICE));
            }
            else if (joystickReady && m_bInstalledAnselJoystick)
            {
                RAWINPUTDEVICE& r = rids[joystickAt];
                r.dwFlags = RIDEV_REMOVE;
                r.hwndTarget = 0;
            }
        }

        if (RegisterRawInputDevices(&(rids[0]), (UINT)rids.size(), sizeof(RAWINPUTDEVICE)) == FALSE)
        {
            LOG_WARN(LogChannel::kInput_RawInput, "UnRegisterRawInputDevices failed!");

            return;
        }

        m_bSavedGameJoystick = false, m_bSavedGameGamepad = false, m_bSavedGameKeyboard = false, m_bSavedGameMouse = false;
        m_bInstalledAnselJoystick = false, m_bInstalledAnselGamepad = false, m_bInstalledAnselKeyboard = false, m_bInstalledAnselMouse = false;

        return;
    }
    
    
    RawInputEventParser::EventRange RawInputEventParser::parseEvent(RAWINPUT* raw)
    {
        m_parsedEvents.clear();

        if (raw->header.dwType == RIM_TYPEKEYBOARD)
        {
            if (!m_RIMan.getKeyboard())
                m_RIMan.setKeyboard(raw->header.hDevice);
            else if (raw->header.hDevice == m_RIMan.getKeyboard())
                m_RIMan.notifySelectedKeyboardEvent();
            else
                m_RIMan.notifyUnselectedKeyboardEvent();
        }
        else if (raw->header.dwType == RIM_TYPEMOUSE)
        {
            if (!m_RIMan.getMouse())
                m_RIMan.setMouse(raw->header.hDevice);
            else if (raw->header.hDevice == m_RIMan.getMouse())
                m_RIMan.notifySelectedMouseEvent();
            else
                m_RIMan.notifyUnselectedMouseEvent();
        }
        else if (raw->header.dwType == RIM_TYPEHID)
        {
            if (!m_RIMan.getGamepad().isInitialzied())
            {
                m_RIMan.setGamepad(raw->header.hDevice);
            }
            else if (raw->header.hDevice == m_RIMan.getGamepad().getHandle())
            {
                m_RIMan.notifySelectedGamepadEvent();
            }
            else
            {
                m_RIMan.notifyUnselectedGamepadEvent();
            }
        }

        const GamepadDevice& gamepad = m_RIMan.getGamepad();

        if (raw->header.dwType == RIM_TYPEKEYBOARD && raw->header.hDevice == m_RIMan.getKeyboard())
        {
            USHORT virtualKey = raw->data.keyboard.VKey;

            if (virtualKey == VK_SHIFT)
            {
                // correct left-hand / right-hand SHIFT
                virtualKey = MapVirtualKey(raw->data.keyboard.MakeCode, MAPVK_VSC_TO_VK_EX);
            }

            const bool isE0 = ((raw->data.keyboard.Flags & RI_KEY_E0) != 0);
            const bool isE1 = ((raw->data.keyboard.Flags & RI_KEY_E1) != 0);

            switch (virtualKey)
            {
                // right-hand CONTROL and ALT have their e0 bit set
            case VK_CONTROL:
                if (isE0)
                    virtualKey = VK_RCONTROL;
                else
                    virtualKey = VK_LCONTROL;
                break;

            case VK_MENU:
                if (isE0)
                    virtualKey = VK_RMENU;
                else
                    virtualKey = VK_LMENU;
                break;

                // NUMPAD ENTER has its e0 bit set - no virtual code for that
            case VK_RETURN:
                if (isE0)
                    virtualKey = VK_SEPARATOR;
                break;

                // the standard INSERT, DELETE, HOME, END, PRIOR and NEXT keys will always have their e0 bit set, but the
                // corresponding keys on the NUMPAD will not.
            case VK_INSERT:
                if (!isE0)
                    virtualKey = VK_NUMPAD0;
                break;

            case VK_DELETE:
                if (!isE0)
                    virtualKey = VK_DECIMAL;
                break;

            case VK_HOME:
                if (!isE0)
                    virtualKey = VK_NUMPAD7;
                break;

            case VK_END:
                if (!isE0)
                    virtualKey = VK_NUMPAD1;
                break;

            case VK_PRIOR:
                if (!isE0)
                    virtualKey = VK_NUMPAD9;
                break;

            case VK_NEXT:
                if (!isE0)
                    virtualKey = VK_NUMPAD3;
                break;

                // the standard arrow keys will always have their e0 bit set, but the
                // corresponding keys on the NUMPAD will not.
            case VK_LEFT:
                if (!isE0)
                    virtualKey = VK_NUMPAD4;
                break;

            case VK_RIGHT:
                if (!isE0)
                    virtualKey = VK_NUMPAD6;
                break;

            case VK_UP:
                if (!isE0)
                    virtualKey = VK_NUMPAD8;
                break;

            case VK_DOWN:
                if (!isE0)
                    virtualKey = VK_NUMPAD2;
                break;

                // NUMPAD 5 doesn't have its e0 bit set
            case VK_CLEAR:
                if (!isE0)
                    virtualKey = VK_NUMPAD5;
                break;
            }

            // Flags contains PRESS(0)/BREAK(1) flags, as well as E0/E1 (additional mapping flags) key versions for some keys
            // e.g. arrows press will have FLAGS==PRESS|E0 and break FLAGS==BREAK|E0
            if ((raw->data.keyboard.Flags & 1) == RI_KEY_MAKE)
            {
                m_parsedEvents.push_back(InputEvent());
                InputEvent& ev = m_parsedEvents.back();
                ev.type = InputEvent::Type::kKeyDown;
                ev.event.keyDown.vkey = virtualKey;
            }
            else if ((raw->data.keyboard.Flags & 1) == RI_KEY_BREAK)
            {
                m_parsedEvents.push_back(InputEvent());
                InputEvent& ev = m_parsedEvents.back();
                ev.type = InputEvent::Type::kKeyUp;
                ev.event.keyUp.vkey = virtualKey;
            }
            else
            {
                LOG_VERBOSE(LogChannel::kInput_RawInput_Kbd, "Keyboard event which is neither key up nor key down!");
            }
        }
        else if (raw->header.dwType == RIM_TYPEMOUSE && raw->header.hDevice == m_RIMan.getMouse())
        {
            //Debug:
            {
                if (raw->data.mouse.usFlags & MOUSE_MOVE_ABSOLUTE)
                    LOG_VERBOSE(LogChannel::kInput_RawInput_Mouse, "Absolute coords in a mouse event!");
            }
                        
            int lastWheel = 0;

            if (raw->data.mouse.usButtonFlags & RI_MOUSE_WHEEL)
            {
                lastWheel = reinterpret_cast<const short &>(raw->data.mouse.usButtonData); //this value is in fact signed
            }

            if ((raw->data.mouse.usButtonFlags & RI_MOUSE_LEFT_BUTTON_DOWN) ||
                (raw->data.mouse.usButtonFlags & RI_MOUSE_MIDDLE_BUTTON_DOWN) ||
                (raw->data.mouse.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_DOWN) ||
                (raw->data.mouse.usButtonFlags & RI_MOUSE_LEFT_BUTTON_UP) ||
                (raw->data.mouse.usButtonFlags & RI_MOUSE_MIDDLE_BUTTON_UP) ||
                (raw->data.mouse.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_UP))
            {
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_LEFT_BUTTON_DOWN)
                {
                    m_parsedEvents.push_back(InputEvent());
                    InputEvent& ev = m_parsedEvents.back();
                    ev.type = InputEvent::Type::kMouseButtonDown;
                    ev.event.mouseButtonDown.btn.value = EMouseButton::kLButton;
                    ev.event.mouseButtonDown.lastCoordsX = raw->data.mouse.lLastX;
                    ev.event.mouseButtonDown.lastCoordsY = raw->data.mouse.lLastY;
                    ev.event.mouseButtonDown.lastCoordsWheel = lastWheel;
                }
                else if (raw->data.mouse.usButtonFlags & RI_MOUSE_LEFT_BUTTON_UP)
                {
                    m_parsedEvents.push_back(InputEvent());
                    InputEvent& ev = m_parsedEvents.back();
                    ev.type = InputEvent::Type::kMouseButtonUp;
                    ev.event.mouseButtonUp.btn.value = EMouseButton::kLButton;
                    ev.event.mouseButtonUp.lastCoordsX = raw->data.mouse.lLastX;
                    ev.event.mouseButtonUp.lastCoordsY = raw->data.mouse.lLastY;
                    ev.event.mouseButtonUp.lastCoordsWheel = lastWheel;
                }

                if (raw->data.mouse.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_DOWN)
                {
                    m_parsedEvents.push_back(InputEvent());
                    InputEvent& ev = m_parsedEvents.back();
                    ev.type = InputEvent::Type::kMouseButtonDown;
                    ev.event.mouseButtonDown.btn.value = EMouseButton::kRButton;
                    ev.event.mouseButtonDown.lastCoordsX = raw->data.mouse.lLastX;
                    ev.event.mouseButtonDown.lastCoordsY = raw->data.mouse.lLastY;
                    ev.event.mouseButtonDown.lastCoordsWheel = lastWheel;
                }
                else if (raw->data.mouse.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_UP)
                {
                    m_parsedEvents.push_back(InputEvent());
                    InputEvent& ev = m_parsedEvents.back();
                    ev.type = InputEvent::Type::kMouseButtonUp;
                    ev.event.mouseButtonUp.btn.value = EMouseButton::kRButton;
                    ev.event.mouseButtonUp.lastCoordsX = raw->data.mouse.lLastX;
                    ev.event.mouseButtonUp.lastCoordsY = raw->data.mouse.lLastY;
                    ev.event.mouseButtonUp.lastCoordsWheel = lastWheel;
                }

                if (raw->data.mouse.usButtonFlags & RI_MOUSE_MIDDLE_BUTTON_DOWN)
                {
                    m_parsedEvents.push_back(InputEvent());
                    InputEvent& ev = m_parsedEvents.back();
                    ev.type = InputEvent::Type::kMouseButtonDown;
                    ev.event.mouseButtonDown.btn.value = EMouseButton::kMButton;
                    ev.event.mouseButtonDown.lastCoordsX = raw->data.mouse.lLastX;
                    ev.event.mouseButtonDown.lastCoordsY = raw->data.mouse.lLastY;
                    ev.event.mouseButtonDown.lastCoordsWheel = lastWheel;
                }
                else if (raw->data.mouse.usButtonFlags & RI_MOUSE_MIDDLE_BUTTON_UP)
                {
                    m_parsedEvents.push_back(InputEvent());
                    InputEvent& ev = m_parsedEvents.back();
                    ev.type = InputEvent::Type::kMouseButtonUp;
                    ev.event.mouseButtonUp.btn.value = EMouseButton::kMButton;
                    ev.event.mouseButtonUp.lastCoordsX = raw->data.mouse.lLastX;
                    ev.event.mouseButtonUp.lastCoordsY = raw->data.mouse.lLastY;
                    ev.event.mouseButtonUp.lastCoordsWheel = lastWheel;
                }
            }
            else
            {
                m_parsedEvents.push_back(InputEvent());
                InputEvent& ev = m_parsedEvents.back();
                ev.type = InputEvent::Type::kMouseMove;
                ev.event.mouseMove.lastCoordsX = raw->data.mouse.lLastX;
                ev.event.mouseMove.lastCoordsY = raw->data.mouse.lLastY;
                ev.event.mouseMove.lastCoordsWheel = lastWheel;
            }
        }
        else if (raw->header.dwType == RIM_TYPEHID && gamepad.isInitialzied() && raw->header.hDevice == gamepad.getHandle())
        {
            for (unsigned int nReport = 0; nReport < raw->data.hid.dwCount; ++nReport)
            {
                m_parsedEvents.push_back(InputEvent());
                InputEvent& ev = m_parsedEvents.back();
                ev.type = InputEvent::Type::kGamepadStateUpdate;

                GamepadStateUpdateEvent& evt = ev.event.gamepadStateUpdate;
                evt.axisLX = evt.axisLY = evt.axisRX = evt.axisRY = evt.axisZ = 0;

                for (int i = 0; i < EGamepadButton::numButtons; ++i)
                    evt.buttonDown[i] = false;

                evt.dpadState.value = EDPadDirection::kCenter;

                m_usageList.resize(gamepad.getTotalNumButtons());

                ULONG aclwmButtons = 0;

                for (int ui = 0, endui = gamepad.getNumButtonUsagePages(); ui != endui; ++ui)
                {
                    ULONG usageLength = (ULONG)gamepad.getButtonUsagePageNumButtons(ui);

                    if (HidP_GetUsages(HidP_Input, gamepad.getButtonUsagePage(ui), 0, &(m_usageList[aclwmButtons]), &usageLength, gamepad.getHidPreparsedData(),
                        ((PCHAR)raw->data.hid.bRawData) + nReport * raw->data.hid.dwSizeHid, raw->data.hid.dwSizeHid) != HIDP_STATUS_SUCCESS)
                    {
                        LOG_WARN(LogChannel::kInput_RawInput_Gamepad, "Failed to get usages for the hid!");

                        continue;
                    }

                    aclwmButtons += usageLength;
                }

                for (unsigned int i = 0; i < (unsigned int)aclwmButtons; i++)
                {
                    EGamepadButton::Enum btn = gamepad.translateButton(m_usageList[i]);

                    if (btn < EGamepadButton::numButtons)
                        evt.buttonDown[btn] = true;
                }

#if 0 //for debugging
                m_values.resize(gamepad.m_hidValueCaps.size());

                for (unsigned int naxis = 0; naxis < m_values.size(); ++naxis)
                {
                    ULONG value;

                    if (HidP_GetUsageValue(HidP_Input, gamepad.m_hidValueCaps[naxis].UsagePage, 0, gamepad.m_hidValueCaps[naxis].NotRange.Usage, &value,
                        gamepad.getHidPreparsedData(), ((PCHAR)raw->data.hid.bRawData) + nReport * raw->data.hid.dwSizeHid, raw->data.hid.dwSizeHid)
                        != HIDP_STATUS_SUCCESS)
                    {
                        OutputDebugString(TEXT("Failed to get usages value for hid!\n"));

                        continue;
                    }

                    m_values[naxis] = value;
                }
#endif
                m_values.resize(gamepad.getNumAxes());

                for (unsigned int naxis = 0; naxis < gamepad.getNumAxes(); ++naxis)
                {
                    ULONG value;

                    if (HidP_GetUsageValue(HidP_Input, gamepad.getAxisUsagePage(naxis), 0, gamepad.getAxisUsage(naxis), &value,
                        gamepad.getHidPreparsedData(), ((PCHAR)raw->data.hid.bRawData) + nReport * raw->data.hid.dwSizeHid, raw->data.hid.dwSizeHid)
                        != HIDP_STATUS_SUCCESS)
                    {
                        LOG_WARN(LogChannel::kInput_RawInput_Gamepad, "Failed to get usages value for hid!");
            
                        continue;
                    }

                    m_values[naxis] = value;
                }

                gamepad.translateAxes(evt, &(m_values[0]));
            }
        }

        return  m_parsedEvents.size() ? std::make_pair(m_parsedEvents.data(), m_parsedEvents.data() + m_parsedEvents.size()) : ilwalidRange();
    }
    
    RawInputFilter::RawInputFilter()
    {
        for (int i = 0; i < 8; ++i)
        {
            m_keyDown[i] = 0;
        }

        for (int i = 0; i < numMouseButtons; ++i)
        {
            m_buttonDown[i] = false;
        }

        m_firstRun = true;
    }

    void RawInputFilter::initializeWithKeyState()
    {
        BYTE keys[256];
        GetKeyboardState(keys);

        for (int i = 0; i < 256; ++i)
        {
            if (keys[i] & 0x80)
                m_keyDown[i / bitSizeOfUInt] |= (1ul << (i % bitSizeOfUInt));
        }
        
        //WAR for sticky Alt
        m_keyDown[VK_MENU / bitSizeOfUInt] |= (1ul << (VK_MENU % bitSizeOfUInt));

        if (m_keyDown[VK_LBUTTON / bitSizeOfUInt] & (1ul << (VK_LBUTTON % bitSizeOfUInt)))
            m_buttonDown[kLeft] = true;

        if (m_keyDown[VK_RBUTTON / bitSizeOfUInt] & (1ul << (VK_RBUTTON % bitSizeOfUInt)))
            m_buttonDown[kRight] = true;

        if (m_keyDown[VK_MBUTTON / bitSizeOfUInt] & (1ul << (VK_MBUTTON % bitSizeOfUInt)))
            m_buttonDown[kMiddle] = true;

    }

    bool RawInputFilter::filterEvent(RAWINPUT* raw, bool blockRequested)
    {
        bool allowThrough = true;

        if (raw->header.dwType == RIM_TYPEKEYBOARD)
        {
            USHORT virtualKey = raw->data.keyboard.VKey;
                    
            // Flags contains PRESS(0)/BREAK(1) flags, as well as E0/E1 (additional mapping flags) key versions for some keys
            // e.g. arrows press will have FLAGS==PRESS|E0 and break FLAGS==BREAK|E0
            if ((raw->data.keyboard.Flags & 1) == RI_KEY_MAKE)
            {
                if (blockRequested)
                {
                    allowThrough = false;
                }
                else
                {
                    allowThrough = true;
                    m_keyDown[virtualKey / bitSizeOfUInt] |= (1ul << (virtualKey % bitSizeOfUInt));
                }
            }
            else if ((raw->data.keyboard.Flags & 1) == RI_KEY_BREAK)
            {
                allowThrough = !blockRequested || !!(m_keyDown[virtualKey / bitSizeOfUInt] & (1ul << (virtualKey % bitSizeOfUInt)));
                m_keyDown[virtualKey / bitSizeOfUInt] &= ~(1ul << (virtualKey % bitSizeOfUInt));
            }
        }
        else if (raw->header.dwType == RIM_TYPEMOUSE)
        {
            bool isDown = false, isUp = false;
            EMouseButton btn = numMouseButtons;

            if (!blockRequested)
            {
                allowThrough = true;
            }
            else if (((raw->data.mouse.usButtonFlags & RI_MOUSE_LEFT_BUTTON_UP) && m_buttonDown[kLeft]) ||
                ((raw->data.mouse.usButtonFlags & RI_MOUSE_MIDDLE_BUTTON_UP) && m_buttonDown[kMiddle]) ||
                ((raw->data.mouse.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_UP) && m_buttonDown[kRight]))
            {
                allowThrough = true;
            }
            else
            {
                allowThrough = false;
            }

            if (allowThrough)
            {
                if (raw->data.mouse.usButtonFlags & RI_MOUSE_LEFT_BUTTON_DOWN)
                    m_buttonDown[kLeft] = true;

                if (raw->data.mouse.usButtonFlags & RI_MOUSE_MIDDLE_BUTTON_DOWN)
                    m_buttonDown[kMiddle] = true;

                if (raw->data.mouse.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_DOWN)
                    m_buttonDown[kRight] = true;

                if (raw->data.mouse.usButtonFlags & RI_MOUSE_LEFT_BUTTON_UP)
                    m_buttonDown[kLeft] = false;

                if (raw->data.mouse.usButtonFlags & RI_MOUSE_MIDDLE_BUTTON_UP)
                    m_buttonDown[kMiddle] = false;

                if (raw->data.mouse.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_UP)
                    m_buttonDown[kRight] = false;
            }
        }
        else if (raw->header.dwType == RIM_TYPEHID)
        {
            allowThrough = blockRequested;
        
        }

        return allowThrough;
    }


    WMKeyDownFilter::WMKeyDownFilter()
    {
        for (int i = 0; i < 8; ++i)
        {
            m_keyDown[i] = 0;
        }

        m_firstRun = true;
    }

    void WMKeyDownFilter::initializeWithKeyState()
    {
        BYTE keys[256];
        GetKeyboardState(keys);

        for (int i = 0; i < 256; ++i)
        {
            if (keys[i] & 0x80)
                m_keyDown[i / bitSizeOfUInt] |= (1ul << (i % bitSizeOfUInt));
        }
    }

    bool WMKeyDownFilter::filterKey(USHORT vkey, bool down, bool blockRequested)
    {
        bool allowThrough = true;

        if (down)
        {
            if (blockRequested)
            {
                allowThrough = false;
            }
            else
            {
                allowThrough = true;
                m_keyDown[vkey / bitSizeOfUInt] |= (1ul << (vkey % bitSizeOfUInt));
            }
        }
        else
        {
            allowThrough = !blockRequested || !!(m_keyDown[vkey / bitSizeOfUInt] & (1ul << (vkey % bitSizeOfUInt)));
            m_keyDown[vkey / bitSizeOfUInt] &= ~(1ul << (vkey % bitSizeOfUInt));
        }
    
        return allowThrough;
    }

    DWORD WINAPI HooksAndThreadsManager::cleanupThreadProc(LPVOID lpParameter)
    {
        //AnselUIState* pUI = (AnselUIState*)lpParameter;

        const wchar_t* class_name = L"ANSEL_INPUT_CLEANUP_CLASS";
        HINSTANCE hInst = GetModuleHandle(nullptr);

        WNDCLASSEX wx = {};
        wx.cbSize = sizeof(WNDCLASSEX);
        wx.lpfnWndProc = cleanupWindowProc;        // function which will handle messages
        wx.hInstance = hInst;
        wx.lpszClassName = class_name;
        RegisterClassEx(&wx);

        HWND msgWindow = 0;

        msgWindow = CreateWindowEx(0, class_name, L"ansel_input_cleanup_window", 0, 0, 0, 0, 0, HWND_MESSAGE, NULL, NULL, NULL);
                
        if (msgWindow)
        {
            // Main message loop
            MSG msg = { 0 };
            while (WM_QUIT != msg.message)
            {
                if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
                {
                    if (msg.hwnd == NULL)
                        msg.hwnd = msgWindow;

                    TranslateMessage(&msg);
                    DispatchMessage(&msg);
                }
            }

            DestroyWindow(msgWindow);
        }

        UnregisterClass(class_name, hInst);

        return 0;
    }

    LRESULT CALLBACK HooksAndThreadsManager::pumpWindowProc(
        HWND   hwnd,
        UINT   uMsg,
        WPARAM wParam,
        LPARAM lParam
        )
    {
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }

    DWORD WINAPI HooksAndThreadsManager::pumpThreadProc(LPVOID lpParameter)
    {
        const wchar_t* class_name = L"ANSEL_INPUT_PUMP_CLASS";
        HINSTANCE hInst = GetModuleHandle(nullptr);

        WNDCLASSEX wx = {};
        wx.cbSize = sizeof(WNDCLASSEX);
        wx.lpfnWndProc = pumpWindowProc;        // function which will handle messages
        wx.hInstance = hInst;
        wx.lpszClassName = class_name;
        RegisterClassEx(&wx);

        HWND msgWindow = 0;
        msgWindow = CreateWindowEx(0, class_name, L"ansel_input_pump_window", 0, 0, 0, 0, 0, HWND_MESSAGE, NULL, NULL, NULL);

        HooksAndThreadsManager::getInstance().m_pumpWindow = msgWindow;
        SetEvent(HooksAndThreadsManager::getInstance().m_pumpWindowSetEvent);

        if (msgWindow)
        {
            // Main message loop
            MSG msg = { 0 };
            while (WM_QUIT != msg.message)
            {
                if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
                {
                    TranslateMessage(&msg);
                    DispatchMessage(&msg);
                }
            }

            DestroyWindow(msgWindow);
        }

        UnregisterClass(class_name, hInst);

        return 0;
    }

    LRESULT CALLBACK HooksAndThreadsManager::hookProc(int nCode, WPARAM wParam, LPARAM lParam)
    {
        HooksAndThreadsManager& mgr = getInstance();

        //this lock is mainly here to protect against the destruction of the HooksAndThreads manager while running a hook.
        //for the queue, a separate lock is used.
        //For most other variables that only influence whether the message should be blocked, thread safety can be ignored
        HooksBarrier::ScopedHookLock l(mgr.m_hooksBarrier);

        if (!l.hasEntered())
            return CallNextHookEx(0, nCode, wParam, lParam);
        
        auto it = mgr.m_threadContextsMap.find(GetLwrrentThreadId());

        if (it == mgr.m_threadContextsMap.end())
        {
            //assert(false); //debug 
            return CallNextHookEx(0, nCode, wParam, lParam);
        }
        
        LwrsorVisibility lwrsorVisibility = mgr.m_lwrsorVisibility;
        
        if (lwrsorVisibility != kLwrsorVisibilityUnmanaged) 
        { // Make sure system cursor is properly displayed with our own mouse trap
            LWRSORINFO lwrsorInfo;
            lwrsorInfo.cbSize = sizeof(LWRSORINFO);

            if (GetLwrsorInfo(&lwrsorInfo))
            {
                if ((lwrsorVisibility == kLwrsorVisibilityOff) && lwrsorInfo.flags == LWRSOR_SHOWING)
                {
                    while (ShowLwrsor(false) >= 0) {}
                }
                else if ((lwrsorVisibility == kLwrsorVisibilityOn) && lwrsorInfo.flags != LWRSOR_SHOWING)
                {
                    while (ShowLwrsor(true) < 0) {}
                }
            }
        }

        HookThreadContext* ctx = it->second;

        if (ctx->m_filter.isFirstRun())
        {
            ctx->m_filter.initializeWithKeyState();
            ctx->m_filter.setFirstRun(false);
        }

        if (ctx->m_syskeydownFilter.isFirstRun())
        {
            ctx->m_syskeydownFilter.initializeWithKeyState();
            ctx->m_syskeydownFilter.setFirstRun(false);
        }

        if (nCode == HC_ACTION)
        {
            bool blockThisMessage = false;

            MSG* msg = reinterpret_cast<MSG*>(lParam);

            switch (msg->message)
            {
                // determine if the input is for us
            case WM_INPUT:
            {
                
                UINT dwSize;
                GetRawInputData((HRAWINPUT)msg->lParam, RID_INPUT, NULL, &dwSize, sizeof(RAWINPUTHEADER));
                UINT alignedSize = (dwSize + 7) & ~7; //round up to 8 bytes to ensure alignment

                if (mgr.m_bHaveFolws)
                {
                    //lock
                    bool failed = false;
                    bool blockInput;

                    unsigned char* byteData = mgr.m_queue.beginInsertRawInput(alignedSize);
                    {
                        blockInput = mgr.m_blockInputToApp;


                        UINT dwSize2 = dwSize; //dummy 
                        UINT dwRet = GetRawInputData((HRAWINPUT)msg->lParam, RID_INPUT, byteData, &dwSize2, sizeof(RAWINPUTHEADER));

                        if (dwRet != dwSize)
                        {
                            LOG_WARN(LogChannel::kInput_RawInput, "GetRawInputData failed");
                            failed = true;
                        }
                        else
                        {
                            RAWINPUT* raw = (RAWINPUT*)byteData;

                            blockInput = !ctx->m_filter.filterEvent(raw, blockInput);
                        }
                    } //unlock
                    mgr.m_queue.endInsertRawInput(failed);
                
                    blockThisMessage = blockInput;
                }
                else
                {
                    blockThisMessage = true;
                }

                break;
            }
            case WM_LBUTTONDOWN:
            case WM_RBUTTONDOWN:
            case WM_MBUTTONDOWN:
            {
                blockThisMessage = mgr.m_blockInputToApp;
                break;
            }
            case WM_KEYDOWN:
            {
                blockThisMessage = false; //TODO come up with a way to distinguish between dialogs and non-dialogs
                blockThisMessage = !ctx->m_syskeydownFilter.filterKey((USHORT)msg->wParam, true, blockThisMessage);
                break;
            }
            case WM_KEYUP:
            {
                blockThisMessage = false; //TODO come up with a way to distinguish between dialogs and non-dialogs
                blockThisMessage = !ctx->m_syskeydownFilter.filterKey((USHORT)msg->wParam, false, blockThisMessage);
                break;
            }
            case WM_SYSKEYDOWN:
            {
                blockThisMessage = (msg->wParam == VK_MENU || msg->wParam == VK_F4 || msg->wParam == VK_RETURN) ? false : mgr.m_blockInputToApp; //TODO come up with a way to distinguish between dialogs and non-dialogs
                blockThisMessage = !ctx->m_syskeydownFilter.filterKey((USHORT) msg->wParam, true, blockThisMessage);
                break;
            }
            case WM_SYSKEYUP:
            {
                blockThisMessage = (msg->wParam == VK_MENU || msg->wParam == VK_F4 || msg->wParam == VK_RETURN) ? false : mgr.m_blockInputToApp; //TODO come up with a way to distinguish between dialogs and non-dialogs
                blockThisMessage = !ctx->m_syskeydownFilter.filterKey((USHORT) msg->wParam, false, blockThisMessage);
                break;
            }
            default:
                break;
            }

            if (blockThisMessage)
            {
                PostThreadMessage(mgr.getCleanupThreadID(), msg->message, msg->wParam, msg->lParam);

                msg->message = WM_NULL;
                return 0; //  CallNextHookEx(0, nCode, wParam, lParam);
            }
            else
            {
                return CallNextHookEx(0, nCode, wParam, lParam);
            }
        }
        else
        {
            return CallNextHookEx(0, nCode, wParam, lParam);
        }
    }
    
    void HooksAndThreadsManager::firstTimeInit(bool haveFolws, DWORD dwForegroundThreadId)
    {
        m_lwrsorVisibility = kLwrsorVisibilityUnmanaged;

        m_hooksBarrier.firstTimeInit();

        startCleanupThread();
            
        m_bLostFolwsThisTick = false;
        m_blockInputToApp = false;

        m_framesLeftToCheckHooks = framesToCheckHooks;
        m_bHaveFolws = haveFolws;

        std::set<DWORD> threadIds = determineThreadIds();
                
        if (m_bHaveFolws)
        {
            if (threadIds.find(dwForegroundThreadId) == threadIds.end())
                m_framesLeftToCheckHooks = 0; // force check hooks
        }

        m_RIMan.firstTimeInit(m_pumpWindow, m_bHaveFolws);
        
        for (auto& tid : threadIds)
        {
            m_threadContextsMap.insert(std::make_pair(tid, new HookThreadContext()));

            HHOOK hook = SetWindowsHookEx(WH_GETMESSAGE, hookProc, GetModuleHandle(NULL), tid);

            if (!hook)
            {
                LOG_WARN(LogChannel::kInput_Hooks, "SetWindowsHookEx  failed" );

                m_threadContextsMap.erase(tid);
            }
            else
            {
                m_threadToHooksMap.insert(std::make_pair(tid, hook));
            }
        }

        m_hooksBarrier.setOffHooks();
                
        m_outDataCounter = 0;
        m_outData.numEvents = 0;
        m_eventRange = m_RIParser.ilwalidRange();
    }

    void HooksAndThreadsManager::tick(bool haveFolws, DWORD dwForegroundThreadId, LwrsorVisibility lwrsorVisibility, bool allowCheckingRawInputDevices)
    {
        bool hadFolws = m_bHaveFolws;
        m_bHaveFolws = haveFolws;
        m_lwrsorVisibility = lwrsorVisibility;

        m_framesLeftToCheckHooks--;
        
        if ((hadFolws != haveFolws) && haveFolws)
        {
            if (m_threadToHooksMap.find(dwForegroundThreadId) == m_threadToHooksMap.end())
                m_framesLeftToCheckHooks = 0; // force check hooks
        }
                        

        if (m_framesLeftToCheckHooks <= 0)
        {
            checkHooks(true);

            m_framesLeftToCheckHooks = framesToCheckHooks;
        }

        m_RIMan.tick(m_bHaveFolws, allowCheckingRawInputDevices);
        
        if (hadFolws && !m_bHaveFolws)
        {
            m_bLostFolwsThisTick = true;
        }
        else
        {
            m_bLostFolwsThisTick = false;
        }
            
        m_outData = m_queue.swapAndConsume();
        m_outDataCounter = 0;
        m_eventRange = m_RIParser.ilwalidRange();
    }

    const InputEvent* HooksAndThreadsManager::popEvent()
    {
        while (true)
        {
            if (m_eventRange.first != m_eventRange.second)
            {
                const InputEvent* ev = m_eventRange.first;
                m_eventRange.first += 1;

                return ev;
            }

            if (m_outDataCounter > m_outData.numEvents)
                return nullptr;

            if (m_outDataCounter == m_outData.numEvents)
            {
                if (isFolwsLostThisTick())
                {
                    m_outDataCounter += 1;

                    return &m_killFolwsEvent;
                }
                else
                {
                    return nullptr;
                }
            }

            unsigned char* byteEvent = m_outData.eventsData + (m_outDataCounter ? m_outData.eventsOffsets[m_outDataCounter - 1] : 0);
            m_eventRange = m_RIParser.parseEvent((RAWINPUT*)byteEvent);

            ++m_outDataCounter;
        }

        return nullptr;
    }

    void HooksAndThreadsManager::checkHooks(bool forceRehook)
    {
        std::set<DWORD> threadIds = determineThreadIds();
        std::vector<DWORD> toErase;

        for (auto& ctxit : m_threadToHooksMap)
        {
            if (!forceRehook)
            {
                auto it = threadIds.find(ctxit.first);

                if (it != threadIds.end())
                {
                    continue;
                }
            }

            UnhookWindowsHookEx(ctxit.second);
            toErase.push_back(ctxit.first);
        }

        if (!toErase.empty())
        {
            m_hooksBarrier.enterMainThreadLock();

            for (auto it = toErase.begin(), end = toErase.end(); it != end; ++it)
            {
                m_threadToHooksMap.erase(*it);
                
                auto ctxit = m_threadContextsMap.find(*it);

                if (ctxit != m_threadContextsMap.end())
                {
                    delete ctxit->second;
                    m_threadContextsMap.erase(ctxit);
                }
                else
                {
                    //assert(false); //debug
                }
            }

            m_hooksBarrier.leaveMainThreadLock();
        }

        std::vector<DWORD> toAdd;

        for (auto& tid : threadIds)
        {
            auto it = m_threadToHooksMap.find(tid);

            if (it == m_threadToHooksMap.end())
            {
                toAdd.push_back(tid);
            }
        }

        if (!toAdd.empty())
        {
            m_hooksBarrier.enterMainThreadLock();

            for (auto& tid : toAdd)
            {
                HHOOK hook = SetWindowsHookEx(WH_GETMESSAGE, hookProc, GetModuleHandle(NULL), tid);
                
                if (!hook)
                {
                    LOG_WARN(LogChannel::kInput_Hooks, "SetWindowsHookEx  failed");
                }
                else
                {
                    m_threadToHooksMap.insert(std::make_pair(tid, hook));
                    m_threadContextsMap.insert(std::make_pair(tid, new HookThreadContext()));
                }
            }

            m_hooksBarrier.leaveMainThreadLock();
        }
    }

    void HooksAndThreadsManager::deinit()
    {
        for (auto& ctxit : m_threadToHooksMap)
        {
            UnhookWindowsHookEx(ctxit.second);
        }

        m_threadToHooksMap.clear();
        m_hooksBarrier.joinHooks();

        for (auto it = m_threadContextsMap.begin(), end = m_threadContextsMap.end(); it != end; ++it)
        {
            delete it->second;
        }

        m_threadContextsMap.clear();
                
        stopCleanupThread();

        m_lwrsorVisibility = kLwrsorVisibilityUnmanaged;

        m_bHaveFolws = false;
        m_blockInputToApp = false;
        m_bLostFolwsThisTick = false;

        m_RIMan.deinit();

        m_outDataCounter = 0;
        m_outData.numEvents = 0;
        m_eventRange = m_RIParser.ilwalidRange();
    }

    void HooksAndThreadsManager::startCleanupThread()
    {
        if (!m_pumpWindowSetEvent)
        {
            m_pumpWindowSetEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
        }

        if (!m_cleanupThread)
        {
            m_cleanupThread = CreateThread(nullptr, 0, cleanupThreadProc, (void*) this, 0, nullptr);
            m_cleanupThreadId = GetThreadId(m_cleanupThread);
        }

        if (!m_pumpThread)
        {
            m_pumpThread = CreateThread(nullptr, 0, pumpThreadProc, (void*) this, 0, nullptr);
            m_pumpThreadId = GetThreadId(m_pumpThread);
            WaitForSingleObject(m_pumpWindowSetEvent, INFINITE);
        }
    }

    void HooksAndThreadsManager::stopCleanupThread()
    {
        if (m_cleanupThread)
        {
            PostThreadMessage(m_cleanupThreadId, WM_QUIT, 0, 0);
            WaitForSingleObject(m_cleanupThread, INFINITE);
            CloseHandle(m_cleanupThread);
            m_cleanupThread = 0;
            m_cleanupThreadId = 0xFFffFFff;
        }

        if (m_pumpThread)
        {
            PostThreadMessage(m_pumpThreadId, WM_QUIT, 0, 0);
            WaitForSingleObject(m_pumpThread, INFINITE);
            CloseHandle(m_pumpThread);
            m_pumpThread = 0;
            m_pumpThreadId = 0xFFffFFff;
        }

        if (m_pumpWindowSetEvent)
        {
            CloseHandle(m_pumpWindowSetEvent);
            m_pumpWindowSetEvent = 0;
        }

        m_pumpWindow = 0;
    }

    BOOL __stdcall HooksAndThreadsManager::enumProcWindowsProc(HWND hwnd, LPARAM lParam)
    {
        DWORD pid;
        DWORD dwThreadId = GetWindowThreadProcessId(hwnd, &pid);
        if (pid == reinterpret_cast<HinstanceAndHwnd*>(lParam)->lwrrentPid && pid != reinterpret_cast<HinstanceAndHwnd*>(lParam)->hookCleanupThread)
        {
            reinterpret_cast<HinstanceAndHwnd*>(lParam)->threadsWithHwndIds.insert(dwThreadId);
        }

        return TRUE;
    }

    std::set<DWORD> HooksAndThreadsManager::determineThreadIds()
    {
        HinstanceAndHwnd hah = { std::set<DWORD>(), GetLwrrentProcessId(), getCleanupThreadID() };
        EnumWindows(enumProcWindowsProc, reinterpret_cast<LPARAM>(&hah));
        hah.threadsWithHwndIds.insert(m_pumpThreadId);//it's message only window, won't be enumerated
        return hah.threadsWithHwndIds;
    }
}
