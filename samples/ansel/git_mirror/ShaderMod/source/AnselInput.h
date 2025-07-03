#ifndef _ANSEL_INPUT_H_
#define _ANSEL_INPUT_H_

#include <set>
#include <vector>
#include <map>
#include <assert.h>
#include <stdint.h>

#include <windows.h>
#include <hidsdi.h> // Gamepad events processing
#include <Psapi.h>
#include <tlhelp32.h>

#define DEBUG_GAMEPAD 0

class AnselServer;
class AnselUI;

namespace input
{
    const float mouseSensititvityCamera = 0.05f;

    struct KeyDownEvent
    {
        USHORT vkey;
    };

    struct KeyUpEvent
    {
        USHORT vkey;
    };

    class KeyboardState
    {
    protected:
        static const unsigned int c_stateStorageSize = 8;
        static const unsigned int c_bitSizeOfUInt = sizeof(unsigned long)* 8;
        static_assert(sizeof(unsigned long) == 4, "sizeof(unsigned long) should be 4!");

    public:

        KeyboardState() { resetState(); }

        bool isKeyDown(USHORT vkey) const { return (m_keyDown[vkey / c_bitSizeOfUInt] & (1ul << (vkey % c_bitSizeOfUInt))) != 0; }
        
        static const unsigned int DoneIterating = 0xFFffFFff;
        static const unsigned int StartIterating = 0xFFffFFff;

        unsigned int getNumVKeys() const;

        unsigned int getNextKeyDown(unsigned int key) const;
        void resetState();
        
    protected:
        unsigned long m_keyDown[c_stateStorageSize];
    };

    class MomentaryKeyboardState: public KeyboardState
    {
    public:

        MomentaryKeyboardState() { resetThis(); }

        bool wasKeyDown(USHORT vkey) const { return (m_wasKeyDown[vkey / c_bitSizeOfUInt] & (1ul << (vkey % c_bitSizeOfUInt))) != 0; }
        bool isKeyStateChanged(USHORT vkey) const
        {
            return !!(m_keyDown[vkey / c_bitSizeOfUInt] & (1ul << (vkey % c_bitSizeOfUInt))) ^
                !!(m_wasKeyDown[vkey / c_bitSizeOfUInt] & (1ul << (vkey % c_bitSizeOfUInt)));
        }

        bool isKeyStateChangedToUp(USHORT vkey) const
        {
            return !(m_keyDown[vkey / c_bitSizeOfUInt] & (1ul << (vkey % c_bitSizeOfUInt))) &&
                (m_wasKeyDown[vkey / c_bitSizeOfUInt] & (1ul << (vkey % c_bitSizeOfUInt)));
        }

        bool isKeyStateChangedToDown(USHORT vkey) const
        {
            return (m_keyDown[vkey / c_bitSizeOfUInt] & (1ul << (vkey % c_bitSizeOfUInt))) &&
                !(m_wasKeyDown[vkey / c_bitSizeOfUInt] & (1ul << (vkey % c_bitSizeOfUInt)));
        }

        static const unsigned int DoneIterating = 0xFFffFFff;
        static const unsigned int StartIterating = 0xFFffFFff;

        unsigned int getNextKeyStateChangedToUp(unsigned int key) const;
        unsigned int getNextKeyStateChangedToDown(unsigned int key) const;
        void resetState();
        void consumeEvent(const KeyDownEvent& ev);
        void consumeEvent(const KeyUpEvent& ev);
        void updatePreviousState();
        void onKillFolws();

    protected:
        void resetThis();

        unsigned long m_wasKeyDown[c_stateStorageSize];
    };

    struct EMouseButton
    {
        enum Enum
        {
            kLButton = 0,
            kRButton,
            kMButton,
            numButtons
        } value;
    };


    struct MouseEvent
    {
        int lastCoordsX, lastCoordsY, lastCoordsWheel;
    };

    struct MouseMoveEvent : public MouseEvent
    {
    };

    struct MouseButtonDownEvent : public MouseEvent
    {
        EMouseButton btn;
    };

    struct MouseButtonUpEvent : public MouseEvent
    {
        EMouseButton btn;
    };

    class MouseState
    {
    public:

        MouseState() { resetState(); }

        bool isButtonDown(EMouseButton::Enum btn) const { return m_buttonDown[btn]; }
            
        int getAclwmulatedCoordX() const { return m_aclwmCoordsX; }
        int getAclwmulatedCoordY() const { return m_aclwmCoordsY; }
        int getAclwmulatedCoordWheel() const { return m_aclwmCoordsWheel; }
            
        void resetState();

    protected:

        bool m_buttonDown[EMouseButton::numButtons];
        int m_aclwmCoordsX, m_aclwmCoordsY, m_aclwmCoordsWheel;
    };


    class MomentaryMouseState: public MouseState
    {
    public:

        MomentaryMouseState() { resetThis(); }

        bool wasButtonDown(EMouseButton::Enum btn) const { return m_wasButtonDown[btn]; }

        bool isButtonStateChanged(EMouseButton::Enum btn) const
        {
            return m_buttonDown[btn] ^ m_wasButtonDown[btn];
        }

        bool isButtonStateChangedToUp(EMouseButton::Enum btn) const
        {
            return !m_buttonDown[btn] && m_wasButtonDown[btn];
        }

        bool isButtonStateChangedToDown(EMouseButton::Enum btn) const
        {
            return m_buttonDown[btn] && !m_wasButtonDown[btn];
        }

        int getLastCoordX() const { return m_lastCoordsX; }
        int getLastCoordY() const { return m_lastCoordsY; }
        int getLastCoordWheel() const { return m_lastCoordsWheel; }
            
        void resetCoordsAclwmulator()
        {
            m_aclwmCoordsX = m_aclwmCoordsY = m_aclwmCoordsWheel = 0;
        }

        void resetThis();
        void resetState();
        void consumeEvent(const MouseButtonDownEvent& ev);
        void consumeEvent(const MouseButtonUpEvent& ev);
        void consumeEvent(const MouseMoveEvent& ev);
        void updatePreviousState();
        void onKillFolws();

    protected:

        bool m_wasButtonDown[EMouseButton::numButtons];
        int m_lastCoordsX, m_lastCoordsY, m_lastCoordsWheel;
    };

    struct EDPadDirection
    {
        enum Enum
        {
            kUp = 0,
            kUpRight = 1,
            kRight = 2,
            kRightDown = 3,
            kDown = 4,
            kDownLeft = 5,
            kLeft = 6,
            kLeftUp = 7,
            kCenter = 8,
            numDirections
        } value;
    };

    struct EGamepadButton
    {
        enum Enum
        {
            kA = 0,
            kB = 1,
            kX = 2,
            kY = 3,
            kLeftShoulder = 4,
            kRightShoulder = 5,
            kLeftStickPress = 8,
            kRightStickPress = 9,
            numButtons
        } value;
    };

    struct GamepadStateUpdateEvent
    {
        bool buttonDown[EGamepadButton::numButtons];
        EDPadDirection dpadState;

        short axisLX;
        short axisLY;
        short axisZ;
        short axisRX;
        short axisRY;
    };

    class GamepadState
    {
    public:

        GamepadState() { resetState(); }

        bool isButtonDown(EGamepadButton::Enum btn) const { return m_buttonDown[btn]; }
        EDPadDirection::Enum getDpadDirection() const { return m_dpadState.value; }
    
        short getAxisLX() const { return m_axisLX; }
        short getAxisLY() const { return m_axisLY; }
        short getAxisRX() const { return m_axisRX; }
        short getAxisRY() const { return m_axisRY; }
        short getAxisZ() const { return m_axisZ; }

        // We might not need that if turns out that the axes are coded in one's complement, stupidly
        const float gamepadStickBacklashGap = 0.2f;

        static float axisToFloat(short val)
        {
            return float(val) / (val > 0 ? float(SHRT_MAX) : -float(SHRT_MIN));
        }

        static float removeBacklash(float value, float backlashGap);
        void resetState();
        
    protected:

        bool m_buttonDown[EGamepadButton::numButtons];
        EDPadDirection m_dpadState;
    
        short m_axisLX;
        short m_axisLY;
        short m_axisZ;
        short m_axisRX;
        short m_axisRY;
    };
    
    class MomentaryGamepadState: public GamepadState
    {
    public:

        MomentaryGamepadState() { resetThis(); }

        bool wasButtonDown(EGamepadButton::Enum btn) const { return m_wasButtonDown[btn]; }

        bool isButtonStateChanged(EGamepadButton::Enum btn) const
        {
            return m_buttonDown[btn] ^ m_wasButtonDown[btn];
        }

        bool isButtonStateChangedToUp(EGamepadButton::Enum btn) const
        {
            return !m_buttonDown[btn] && m_wasButtonDown[btn];
        }

        bool isButtonStateChangedToDown(EGamepadButton::Enum btn) const
        {
            return m_buttonDown[btn] && !m_wasButtonDown[btn];
        }

        EDPadDirection::Enum getDpadPrevDirection() const { return m_dpadStateWas.value; }
        bool isDpadDirectionChanged() const { return m_dpadState.value != m_dpadStateWas.value; }

        void resetThis();
        void resetState();
        void consumeEvent(const GamepadStateUpdateEvent& ev);
        void updatePreviousState();
        void onKillFolws();

    protected:

        bool m_wasButtonDown[EGamepadButton::numButtons];
        EDPadDirection m_dpadStateWas;
    };

    struct InputEvent
    {
        enum struct Type
        {
            kGamepadStateUpdate = 0,
            kMouseMove,
            kMouseButtonDown,
            kMouseButtonUp,
            kKeyDown,
            kKeyUp,
            kKillFolws,
            numTypes
        } type;

        union
        {
            GamepadStateUpdateEvent     gamepadStateUpdate;
            MouseMoveEvent  mouseMove;
            MouseButtonDownEvent mouseButtonDown;
            MouseButtonUpEvent mouseButtonUp;
            KeyDownEvent keyDown;
            KeyUpEvent keyUp;
        } event;
    };

    class InputEventQueue
    {
    public:
        InputEventQueue() : m_bproduce(&m_bqueue0), m_bconsume(&m_bqueue1), m_sproduce(&m_squeue0), m_sconsume(&m_squeue1)
        {
            initialize();
            m_bIsValid = true;
        }

        ~InputEventQueue()
        {
            destroy();
        }

        void initialize();
        void destroy();
        unsigned char* beginInsertRawInput(unsigned int size);
        void endInsertRawInput(bool undo);

        struct RawData
        {
            unsigned int numEvents;
            unsigned char* eventsData;
            unsigned int* eventsOffsets;
        };

        RawData swapAndConsume();

    protected:

        bool                        m_bIsValid;
        CRITICAL_SECTION            m_lock;

        std::vector<unsigned char>  m_bqueue0, m_bqueue1;
        std::vector<unsigned int>   m_squeue0, m_squeue1;

        std::vector<unsigned char>* m_bproduce;
        std::vector<unsigned char>* m_bconsume;
        std::vector<unsigned int>*  m_sproduce;
        std::vector<unsigned int>* m_sconsume;
    };

    class InputState
    {
    public:

        GamepadState& getGamepadState() { return m_gamepadState; }
        MouseState& getMouseState() { return m_mouseState; }
        KeyboardState& getKeyboardState() { return m_keyboardState; }

    protected:
        
        MomentaryKeyboardState              m_fakeKbdState;

        MomentaryGamepadState               m_gamepadState;
        MomentaryMouseState                 m_mouseState;
        MomentaryKeyboardState              m_keyboardState;

        void resetState()
        {
            m_gamepadState.resetState();
            m_mouseState.resetState();
            m_keyboardState.resetState();
        }

        void updatePreviousState()
        {
            m_gamepadState.updatePreviousState();
            m_mouseState.updatePreviousState();
            m_keyboardState.updatePreviousState();
        }

        void consumeEvent(const InputEvent& ev)
        {
            updatePreviousState();

            switch (ev.type)
            {
            case InputEvent::Type::kGamepadStateUpdate:
                m_gamepadState.consumeEvent(ev.event.gamepadStateUpdate);
                break;
            case InputEvent::Type::kMouseMove:
                m_mouseState.consumeEvent(ev.event.mouseMove);
                break;
            case InputEvent::Type::kMouseButtonDown:
                m_mouseState.consumeEvent(ev.event.mouseButtonDown);
                break;
            case InputEvent::Type::kMouseButtonUp:
                m_mouseState.consumeEvent(ev.event.mouseButtonUp);
                break;
            case InputEvent::Type::kKeyDown:
                m_keyboardState.consumeEvent(ev.event.keyDown);
                break;
            case InputEvent::Type::kKeyUp:
                m_keyboardState.consumeEvent(ev.event.keyUp);
                break;
            case InputEvent::Type::kKillFolws:
                m_keyboardState.onKillFolws();
                m_gamepadState.onKillFolws();
                m_mouseState.onKillFolws();
                break;
            default:
                break;
            }
        }
    };
    
    class GamepadDevice
    {
        static const int MAX_NUM_GAMEPAD_AXES = 7; //Lx, Ly, lt, rt, RX, RY, Dpad
    public:

        enum struct EGamepadDevice
        {
            kXboxOne = 0,
            kShield,
            kXbox360,
            kDualShock4,
            numDevices,
            kUnknown = numDevices
        };

        struct GamepadStats
        {
            EGamepadDevice type = EGamepadDevice::kUnknown;
            DWORD dwVendorId = 0;
            DWORD dwProductId = 0;
            DWORD dwVersionNumber = 0;
        };

        enum struct EGamepadAxis
        {
            kLeftStickX = 0,
            kLeftStickY,
            kRightStickX,
            kRightStickY,
            kLeftTrigger,
            kRightTrigger,
            kBothTriggers,
            kDpad,
            numAxes
        };

        GamepadDevice() : m_handle(0), m_deviceType(EGamepadDevice::kUnknown), m_dwVendorId(0), m_dwProductId(0),m_dwVersionNumber(0) {}
        
        void ilwalidate()
        {
            m_handle = 0; m_dwVendorId = 0; m_dwProductId = 0; m_dwVersionNumber = 0;
            m_deviceType = EGamepadDevice::kUnknown;
        }

        void getStats(GamepadStats& stats) const;

        HANDLE getHandle() const { return m_handle; }
        bool isInitialzied() const { return m_handle != 0; }
        unsigned int getTotalNumButtons() const { return m_numButtons; }

        USAGE getButtonUsagePage(unsigned int n) const { return m_buttonUsagePages[n]; }
        unsigned int getButtonUsagePageNumButtons(unsigned int n) const { return m_buttonUsagePageButtonCount[n]; }
        unsigned int getNumButtonUsagePages() const
        {
            assert(m_buttonUsagePages.size() == m_buttonUsagePageButtonCount.size());
            return (unsigned int)m_buttonUsagePageButtonCount.size();
        }

        const PHIDP_PREPARSED_DATA getHidPreparsedData() const { return (PHIDP_PREPARSED_DATA)&(m_hidPreparsedData[0]); }
        unsigned int getNumAxes() const { return m_numAxes; }
        USAGE getAxisUsagePage(unsigned int naxis) const { assert(naxis < getNumAxes()); return m_hidValueCaps[m_axisToValueCap[naxis]].UsagePage; }
        USAGE getAxisUsage(unsigned int naxis) const { assert(naxis < getNumAxes()); return m_axisUsages[naxis]; }
        EGamepadDevice getDeviceType() const { return m_deviceType; }

        EGamepadButton::Enum translateButton(unsigned long idx) const;
        void translateAxes(GamepadStateUpdateEvent& evt, const ULONG* values) const;
        void initialize(HANDLE hGamepad, DWORD dwVendorId, DWORD dwProductId, DWORD dwVersionNumber);

    protected:
        static EGamepadDevice deviceFromDeviceInfo(DWORD dwVendorId, DWORD dwProductId, DWORD dwVersionNumber);


        HANDLE m_handle;
        EGamepadDevice m_deviceType;
        DWORD m_dwVendorId;
        DWORD m_dwProductId;
        DWORD m_dwVersionNumber;

        unsigned int m_numAxes;
        EGamepadAxis m_axisType[MAX_NUM_GAMEPAD_AXES];
        int m_axisToValueCap[MAX_NUM_GAMEPAD_AXES];
        USAGE m_axisUsages[MAX_NUM_GAMEPAD_AXES];

        HIDP_CAPS m_hidCaps;
        std::vector<HIDP_VALUE_CAPS> m_hidValueCaps;
        std::vector<HIDP_BUTTON_CAPS> m_hidButtonCaps;
        std::vector<unsigned char> m_hidPreparsedData;

        unsigned int m_numButtons;
        std::vector<USAGE> m_buttonUsagePages;
        std::vector<unsigned int> m_buttonUsagePageButtonCount;

        GamepadStats m_statsForTelemetry; //we need this because we'd like this to be valid even if the gamepad is deinitialzied
    };

    class RawInputManager
    {
        static const int framesToCheckRawInput = 200;
        static const int framesToCheckDeviceChange = 150;

    public:

        void firstTimeInit(HWND pumpWnd, bool checkRIDs);
        void deinit();
        void tick(bool haveFolws, bool allowCheckingRawInputDevices);

        HANDLE getMouse() const { return m_hmouse; }
        HANDLE getKeyboard() const { return m_hkeyboard; }
        const GamepadDevice& getGamepad() const { return m_gamepad; }

        void setMouse(HANDLE mouse)
        {
            m_hmouse = mouse;
        }

        void setKeyboard(HANDLE kbd)
        {
            m_hkeyboard = kbd;
        }

        bool setGamepad(HANDLE gamepad);

        void notifySelectedMouseEvent()
        {
            m_selectedMouseEventRegistered = true;
        }

        void notifyUnselectedMouseEvent()
        {
            m_unselectedMouseEventRegistered = true;
        }

        void notifySelectedKeyboardEvent()
        {
            m_selectedKeyboardEventRegistered = true;
        }

        void notifyUnselectedKeyboardEvent()
        {
            m_unselectedKeyboardEventRegistered = true;
        }

        void notifySelectedGamepadEvent()
        {
            m_selectedGamepadEventRegistered = true;
        }

        void notifyUnselectedGamepadEvent()
        {
            m_unselectedGamepadEventRegistered = true;
        }

        void resetEventReceivedFlags();
        void checkDeviceHandles();

        bool isGameMouseSaved() const
        {
            return  m_bSavedGameMouse;
        }

        bool isGameKeyboardSaved() const
        {
            return  m_bSavedGameKeyboard;
        }

        bool isGameGamepadSaved() const
        {
            return  m_bSavedGameGamepad;
        }

        bool isGameJoystickSaved() const
        {
            return  m_bSavedGameJoystick;
        }

        bool isGameMouseInputsink() const
        {
            return  !!(m_gameRidMouse.dwFlags & RIDEV_INPUTSINK);
        }

        bool isGameKeyboardInputsink() const
        {
            return  !!(m_gameRidKeyboard.dwFlags & RIDEV_INPUTSINK);
        }

        bool isGameGamepadInputsink() const
        {
            return  !!(m_gameRidGamepad.dwFlags & RIDEV_INPUTSINK);
        }

        bool isGameJoystickInputsink() const
        {
            return  !!(m_gameRidJoystick.dwFlags & RIDEV_INPUTSINK);
        }
                
        HWND getGameMouseRawInputWindow() const
        {
            return m_gameRidMouse.hwndTarget;
        }

        HWND getGameKeyboardRawInputWindow() const
        {
            return m_gameRidKeyboard.hwndTarget;
        }

        HWND getGameGamepadRawInputWindow() const
        {
            return m_gameRidGamepad.hwndTarget;
        }

        HWND getGameJoystickRawInputWindow() const
        {
            return m_gameRidJoystick.hwndTarget;
        }

    protected:

        bool checkRawInputDevices();
        void removeRestoreRawInputDevices();

        HWND                m_pumpWindow = 0;

        HANDLE              m_hkeyboard;
        HANDLE              m_hmouse;
        GamepadDevice       m_gamepad;

        RAWINPUTDEVICE      m_gameRidMouse;
        RAWINPUTDEVICE      m_gameRidKeyboard;
        RAWINPUTDEVICE      m_gameRidGamepad;
        RAWINPUTDEVICE      m_gameRidJoystick;

        bool                m_bSavedGameMouse = false;
        bool                m_bSavedGameKeyboard = false;
        bool                m_bSavedGameGamepad = false;
        bool                m_bSavedGameJoystick = false;

        bool                m_bInstalledAnselMouse = false;
        bool                m_bInstalledAnselKeyboard = false;
        bool                m_bInstalledAnselGamepad = false;
        bool                m_bInstalledAnselJoystick = false;

        bool                m_bCheckRIDsNextFrame = false;
        int                 m_framesToCheckDeviceChangeRemaining;
        int                 m_framesToCheckRidsRemaining;


        bool                m_selectedMouseEventRegistered;
        bool                m_unselectedMouseEventRegistered;
        bool                m_selectedKeyboardEventRegistered;
        bool                m_unselectedKeyboardEventRegistered;
        bool                m_selectedGamepadEventRegistered;
        bool                m_unselectedGamepadEventRegistered;
    };


    class RawInputEventParser
    {
    public:

        RawInputEventParser(RawInputManager& riman) : m_RIMan(riman)
        {
        }

        typedef std::pair<const InputEvent*, const InputEvent*> EventRange;

        EventRange parseEvent(RAWINPUT* raw);

        static EventRange ilwalidRange()
        {
            return std::make_pair(nullptr, nullptr);
        }

    protected:

        RawInputManager&    m_RIMan;

        //temps:
        std::vector<USAGE> m_usageList;
        std::vector<ULONG> m_values;

        std::vector<InputEvent> m_parsedEvents;
    };

    class RawInputFilter
    {
    public:
        RawInputFilter();

        void initializeWithKeyState();

        //true if the event should go to the game
        bool filterEvent(RAWINPUT* raw, bool blockRequested);

        bool isFirstRun() const { return m_firstRun; }
        void setFirstRun(bool b) { m_firstRun = b; }
        
    protected:

        static const unsigned int bitSizeOfUInt = sizeof(unsigned long)* 8;
        static_assert(sizeof(unsigned long) == 4, "sizeof(unsigned long) should be 4!");
        
        unsigned long m_keyDown[8];

        enum EMouseButton
        {
            kLeft = 0,
            kMiddle,
            kRight,
            numMouseButtons
        };

        bool m_buttonDown[numMouseButtons];

        bool m_firstRun;
    };


    class WMKeyDownFilter
    {
    public:
        WMKeyDownFilter();

        void initializeWithKeyState();

        bool isFirstRun() const { return m_firstRun; }
        void setFirstRun(bool b) { m_firstRun = b; }
        
        //true if the event should go to the game
        bool filterKey(USHORT vkey, bool down, bool blockRequested);


    protected:

        static const unsigned int bitSizeOfUInt = sizeof(unsigned long)* 8;
        static_assert(sizeof(unsigned long) == 4, "sizeof(unsigned long) should be 4!");

        unsigned long m_keyDown[8];
        bool m_firstRun;
    };

    class HooksBarrier
    {
    public:
        HooksBarrier() : m_numHooksRunningInParallelNow(0), m_dontEnterHookEvent(0), m_mainThreadAcquiredLockEvent(0),
            m_dontTouchHookReadDataEvent(0), m_allowHooksRun(0)
        {}

        //don't rely on that, deinit manually at the proper place
        ~HooksBarrier()
        {
            deinit();
        }

        //assumption: no hooks running, called from the main thread
        void firstTimeInit()
        {
            m_allowHooksRun = 0;
            m_numHooksRunningInParallelNow = 0;
            m_dontEnterHookEvent = CreateEvent(NULL, TRUE, TRUE, NULL);
            m_dontTouchHookReadDataEvent = CreateEvent(NULL, TRUE, TRUE, NULL);
            m_mainThreadAcquiredLockEvent = CreateEvent(NULL, TRUE, TRUE, NULL);
        }

        //assumption: initialized, hooks set, but not allowed to run yet, called from the main thread
        void setOffHooks()
        {
            assert(!m_allowHooksRun);
            m_allowHooksRun = 1;
        }
        
        //assumption: no hooks running, called from the main thread
        void deinit()
        {
            m_allowHooksRun = 0;
            m_numHooksRunningInParallelNow = 0;
            
            if (m_dontEnterHookEvent)
                CloseHandle(m_dontEnterHookEvent), m_dontEnterHookEvent = 0;

            if (m_dontTouchHookReadDataEvent)
                CloseHandle(m_dontTouchHookReadDataEvent), m_dontTouchHookReadDataEvent = 0;

            if (m_mainThreadAcquiredLockEvent)
                CloseHandle(m_mainThreadAcquiredLockEvent), m_mainThreadAcquiredLockEvent = 0;
        }

        //assumptions: initialized, hooks running, call once at the top of every hook in the hook thread
        //returns true if hook LOCKED and can proceed touching the data. If false, the hook shouldn't access any non-local data and shouldn't unlock
        bool enterHookFunctionLock()
        {
            WaitForSingleObject(m_dontEnterHookEvent, INFINITE);
            
            if (!m_allowHooksRun)
                return false;

            ResetEvent(m_dontTouchHookReadDataEvent);
            InterlockedIncrement(&m_numHooksRunningInParallelNow);

            //this is a very rare situation that might occur if the main thread grabbed the  dontEnetrHookEvent lock after the hook waited for it, 
            //but before the hook grabbed the dontTouchHookreadDataEvent
            //this isn't a proper solution either, but the chances of a race condition are very small. TODO: come up with a truely atomic fix
            WaitForSingleObject(m_mainThreadAcquiredLockEvent, INFINITE);

            if (!m_allowHooksRun)
            {
                unsigned int numHooksRunning = InterlockedDecrement(&m_numHooksRunningInParallelNow);

                if (!numHooksRunning)
                    SetEvent(m_dontTouchHookReadDataEvent);

                return false;
            }

            return true;
        }

        //assumptions: initialized, hooks running, call once on return from  every hook in the hook thread
        void leaveHookFunctionLock()
        {
            unsigned int numHooksRunning = InterlockedDecrement(&m_numHooksRunningInParallelNow);

            if (!numHooksRunning)
                SetEvent(m_dontTouchHookReadDataEvent);

        }

        //assumptions: initialized, hooks running, call in the main thread before you touch data shared with the hooks
        void enterMainThreadLock()
        {
            ResetEvent(m_dontEnterHookEvent);
            WaitForSingleObject(m_dontTouchHookReadDataEvent, INFINITE);
            ResetEvent(m_mainThreadAcquiredLockEvent);
        }

        //assumptions: initialized, hooks running, call in the main thread after you touch data shared with the hooks
        void leaveMainThreadLock()
        {
            SetEvent(m_mainThreadAcquiredLockEvent);
            SetEvent(m_dontEnterHookEvent);
        }

        //assumption: initialized, hooks RESET, but may be running, call in the main thread to wait for hooks to finish
        void joinHooks()
        {
            ResetEvent(m_dontEnterHookEvent);
            m_allowHooksRun = 0;
            WaitForSingleObject(m_dontTouchHookReadDataEvent, INFINITE);
            ResetEvent(m_mainThreadAcquiredLockEvent);
            SetEvent(m_mainThreadAcquiredLockEvent);
            SetEvent(m_dontEnterHookEvent);
        }

        class ScopedHookLock
        {
        public:
            ScopedHookLock(HooksBarrier& bar) : m_bar(bar) { m_entered = m_bar.enterHookFunctionLock(); }
            ~ScopedHookLock() { if (m_entered) m_bar.leaveHookFunctionLock(); }
            
            bool hasEntered() const
            {
                return m_entered;
            }

        protected:
            HooksBarrier&   m_bar;
            bool m_entered;
        };

        class ScopedMainThreadLock
        {
        public:
            ScopedMainThreadLock(HooksBarrier& bar) : m_bar(bar) {m_bar.enterMainThreadLock(); }
            ~ScopedMainThreadLock() { m_bar.leaveMainThreadLock(); }
        protected:
            HooksBarrier&   m_bar;
        };

    protected:
        volatile bool m_allowHooksRun;
        volatile unsigned int m_numHooksRunningInParallelNow;
        HANDLE      m_dontEnterHookEvent;
        HANDLE      m_dontTouchHookReadDataEvent;
        HANDLE      m_mainThreadAcquiredLockEvent;
    };

    enum LwrsorVisibility
    {
        kLwrsorVisibilityUnmanaged,
        kLwrsorVisibilityOn,
        kLwrsorVisibilityOff
    };

    class HooksAndThreadsManager
    {
        static const int framesToCheckHooks = 250;

    public:
        
        void getGamepadStats(GamepadDevice::GamepadStats& stats) const
        {
            m_RIMan.getGamepad().getStats(stats);
        }

        static LRESULT CALLBACK cleanupWindowProc(
            HWND   hwnd,
            UINT   uMsg,
            WPARAM wParam,
            LPARAM lParam
            )
        {
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
        }

        static DWORD WINAPI cleanupThreadProc(LPVOID lpParameter);
    
        static LRESULT CALLBACK pumpWindowProc(
            HWND   hwnd,
            UINT   uMsg,
            WPARAM wParam,
            LPARAM lParam
            );

        static DWORD WINAPI pumpThreadProc(LPVOID lpParameter);
        static LRESULT CALLBACK hookProc(int nCode, WPARAM wParam, LPARAM lParam);

        static HooksAndThreadsManager& getInstance()
        {
            static HooksAndThreadsManager htm;

            return htm;
        }

        InputEventQueue& getEventQueue()
        {
            return m_queue;
        }

        void firstTimeInit(bool haveFolws, DWORD dwForegroundThreadId);
        void tick(bool haveFolws, DWORD dwForegroundThreadId, LwrsorVisibility lwrsorVisibility, bool allowCheckingRawInputDevices);
        void setBlockInput(bool doBlock)
        {
            m_blockInputToApp = doBlock;
        }
        const InputEvent* popEvent();
        void deinit();

    protected:
        void checkHooks(bool forceRehook);

        bool isFolwsLostThisTick()
        {
            return m_bLostFolwsThisTick;
        }

        HooksAndThreadsManager() : m_RIParser(m_RIMan)
        {
            m_killFolwsEvent.type = InputEvent::Type::kKillFolws;

            m_bHaveFolws = false;
            m_blockInputToApp = false;
            m_bLostFolwsThisTick = false;

            m_outDataCounter = 0;
            m_outData.numEvents = 0;
            m_eventRange = m_RIParser.ilwalidRange();
        }

        ~HooksAndThreadsManager()
        {
            deinit();
        }

        DWORD getCleanupThreadID() const
        {
            return m_cleanupThreadId;
        }

        DWORD getPumpThreadID() const
        {
            return m_pumpThreadId;
        }

        void startCleanupThread();
        void stopCleanupThread();

        struct HinstanceAndHwnd
        {
            std::set<DWORD> threadsWithHwndIds;
            DWORD lwrrentPid;
            DWORD hookCleanupThread;
        };

        static BOOL __stdcall enumProcWindowsProc(HWND hwnd, LPARAM lParam);
        std::set<DWORD> determineThreadIds();

        struct HookThreadContext
        {
            RawInputFilter m_filter;
            WMKeyDownFilter m_syskeydownFilter;
        };

        std::map<DWORD, HookThreadContext*> m_threadContextsMap;

        HooksBarrier    m_hooksBarrier;
        std::map<DWORD, HHOOK> m_threadToHooksMap;

        volatile bool m_blockInputToApp;
        bool m_bHaveFolws;
        bool m_bLostFolwsThisTick;

        volatile input::LwrsorVisibility m_lwrsorVisibility;

        RawInputManager m_RIMan;
        RawInputEventParser m_RIParser;
        InputEventQueue m_queue;
        HANDLE          m_cleanupThread = 0;
        DWORD           m_cleanupThreadId = 0xFFffFFff;
        HANDLE          m_pumpThread = 0;
        DWORD           m_pumpThreadId = 0xFFffFFff;
        HWND            m_pumpWindow = 0;
        HANDLE          m_pumpWindowSetEvent = 0;
        DWORD           m_readInputThreadId = 0xFFffFFff;

        int             m_framesLeftToCheckHooks;

        //iterating over outputs:

        InputEventQueue::RawData                    m_outData;
        unsigned int                                m_outDataCounter;
        RawInputEventParser::EventRange             m_eventRange;
        InputEvent                                  m_killFolwsEvent;
    };

}

#endif
