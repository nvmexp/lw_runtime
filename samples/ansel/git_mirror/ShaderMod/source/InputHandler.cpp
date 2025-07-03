#include "InputHandler.h"
#include "UI.h"
#include "AnselSDKState.h"
#include "AnselServer.h"

#if (ENABLE_XINPUT_SUPPORT == 1)
#   include <xinput.h>
#   pragma comment (lib, "xinput9_1_0.lib")

#   include <dinputd.h>
#   pragma comment (lib, "dinput8.lib")

LPDIRECTINPUT8          m_pDI = nullptr;
LPDIRECTINPUTDEVICE8    m_pDIController = nullptr;
bool m_filterOutXinputDevices = true;
struct DI_ENUM_CONTEXT
{
    DIJOYCONFIG * pPreferredJoyCfg;
    bool bPreferredJoyCfgValid;
};
#endif

#define DBG_ADDITIONAL_HOTKEYS  1

#define SAFE_DELETE_ARRAY(p)  { if(p) { delete[] (p);     (p)=nullptr; } }

namespace input
{
    bool FolwsChecker::checkFolws()
    {
        bool folwsUpdated = false;

        HWND foregroundWnd = GetForegroundWindow();

        if (foregroundWnd != m_foregroundWindow)
        {
            DWORD pid;
            DWORD dwThreadId = GetWindowThreadProcessId(foregroundWnd, &pid);
            DWORD lwrrentPid = GetLwrrentProcessId();

            bool foregroundWindowBelongsToThisApp = pid == lwrrentPid;

            if (!foregroundWindowBelongsToThisApp)
            {
                //UWP fallback
                HWND theRealForegroundWindow = FindWindowEx(foregroundWnd, NULL, L"Windows.UI.Core.CoreWindow", NULL);

                if (theRealForegroundWindow)
                {
                    foregroundWnd = theRealForegroundWindow;
                    dwThreadId = GetWindowThreadProcessId(foregroundWnd, &pid);
                    foregroundWindowBelongsToThisApp = pid == lwrrentPid;
                }
            }

            if (foregroundWindowBelongsToThisApp)
            {
                m_foregroundWindow = foregroundWnd;
                m_bHaveFolws = true;
            }
            else
            {
                m_foregroundWindow = 0;
                m_bHaveFolws = false;
            }

            m_foregroundThread = dwThreadId;
            folwsUpdated = true;
        }

        return folwsUpdated;
    }

#if (ENABLE_XINPUT_SUPPORT == 1)
    class XInputDeviceCheck
    {
    protected:

        RAWINPUTDEVICELIST * m_devices = nullptr;

        std::vector<DWORD> m_xInputVidPids;

    public:

        void rebuildDevicesList()
        {
            UINT numDevices, numDevicesTemp;
            UINT i;

            m_xInputVidPids.resize(0);

            if (GetRawInputDeviceList(NULL, &numDevicesTemp, sizeof(RAWINPUTDEVICELIST)) != 0)
            {
                return;
            }
            m_devices = new RAWINPUTDEVICELIST[numDevicesTemp];
            if (m_devices == nullptr)
            {
                return;
            }
            if ((numDevices = GetRawInputDeviceList(m_devices, &numDevicesTemp, sizeof(RAWINPUTDEVICELIST))) == (UINT)-1)
            {
                cleanup();
                return;
            }

            // Loop over all available devices
            for (i = 0; i < numDevices; ++i)
            {
                // Assuming that all possible XInput devices will be reported as generic HID devices and not as keyboards or mice
                if (m_devices[i].dwType == RIM_TYPEHID)
                {
                    RID_DEVICE_INFO rdi;
                    UINT cbSize;

                    cbSize = rdi.cbSize = sizeof(rdi);
                    if ((INT)GetRawInputDeviceInfoA(m_devices[i].hDevice, RIDI_DEVICEINFO, &rdi, &cbSize) >= 0)
                    {
                        const int nameBufSize = 256;
                        char nameBuf[nameBufSize];
                        UINT nameSize = nameBufSize;
                        UINT reslen;

                        reslen = GetRawInputDeviceInfoA(m_devices[i].hDevice, RIDI_DEVICENAME, nameBuf, &nameSize);
                        if (reslen != (UINT)-1)
                        {
                            // Check if the device ID contains "IG_". If it does, then it's an XInput device;
                            //  unfortunately this information can not be found by just using DirectInput 
                            if (strstr(nameBuf, "IG_") != NULL)
                            {
                                DWORD dwVidPid = MAKELONG(rdi.hid.dwVendorId, rdi.hid.dwProductId);

                                // Add the VID/PID to a list of XInput devices
                                m_xInputVidPids.push_back(dwVidPid);
                            }
                        }
                    }
                }
            }
        }

        void cleanup()
        {
            SAFE_DELETE_ARRAY(m_devices);
        }

        // Returns true if the DirectInput device is also an XInput device.
        bool isXInputDevice(const GUID & pGuidProductFromDirectInput)
        {
            for (size_t devIdx = 0, devIdxEnd = m_xInputVidPids.size(); devIdx < devIdxEnd; ++devIdx)
            {
                // Check each XInput device to see if this device's vid/pid matches
                if (m_xInputVidPids[devIdx] == pGuidProductFromDirectInput.Data1)
                    return true;
            }
            return false;
        }
    };
    XInputDeviceCheck g_xInputDeviceCheck;

    // This function is called once for each enumerated controller
    BOOL CALLBACK EnumControllersCallback(const DIDEVICEINSTANCE * pdidInstance, VOID * pContext)
    {
        auto pEnumContext = reinterpret_cast<DI_ENUM_CONTEXT *>(pContext);
        HRESULT hr;

        bool isXInputDevice = g_xInputDeviceCheck.isXInputDevice(pdidInstance->guidProduct);
        if (isXInputDevice)
        {
            LOG_VERBOSE("Device %d is XInput device (RI)", pdidInstance->guidProduct);
        }
        else
        {
            LOG_VERBOSE("Device %d is NOT an XInput device (RI)", pdidInstance->guidProduct);
        }

        if (m_filterOutXinputDevices && isXInputDevice)
            return DIENUM_CONTINUE;

        // Skip anything other than the perferred controller device as defined by the control panel,
        //  as alternative, user could pick desired controller from a list that we could provide
        if (pEnumContext->bPreferredJoyCfgValid &&
            !IsEqualGUID( pdidInstance->guidInstance, pEnumContext->pPreferredJoyCfg->guidInstance))
            return DIENUM_CONTINUE;

        // Obtain an interface to the enumerated controller
        hr = m_pDI->CreateDevice(pdidInstance->guidInstance, &m_pDIController, nullptr);

        // If it failed, then we can't use this controller
        //  e.g. user unplugged it in the middle of enumerating it
        if (FAILED(hr))
            return DIENUM_CONTINUE;

        // Taking the first controller we get
        return DIENUM_STOP;
    }

    // Callback function for enumerating objects (axes, buttons, POVs) on a controller
    BOOL CALLBACK EnumObjectsCallback(const DIDEVICEOBJECTINSTANCE * pdidoi, VOID * pContext)
    {
        // For axes that are returned, set the DIPROP_RANGE property for the
        //  enumerated axis in order to scale min/max values.
        if (pdidoi->dwType & DIDFT_AXIS)
        {
            DIPROPRANGE diprg;
            diprg.diph.dwSize = sizeof(DIPROPRANGE);
            diprg.diph.dwHeaderSize = sizeof(DIPROPHEADER);
            diprg.diph.dwHow = DIPH_BYID;
            diprg.diph.dwObj = pdidoi->dwType; // Specify the enumerated axis
            diprg.lMin = -1000;
            diprg.lMax = +1000;

            // Set the range for the axis
            if (FAILED(m_pDIController->SetProperty(DIPROP_RANGE, &diprg.diph)))
                return DIENUM_STOP;
        }

        return DIENUM_CONTINUE;
    }

    HRESULT InputHandlerForStandalone::InitDirectInput()
    {
        HRESULT hr;

        // Register with the DirectInput subsystem and get a pointer to a IDirectInput interface we can use
        // Create a DInput object
        hr = DirectInput8Create(
                GetModuleHandle(nullptr),
                DIRECTINPUT_VERSION,
                IID_IDirectInput8,
                (VOID**)&m_pDI,
                nullptr
                );
        if (FAILED(hr))
            return hr;

        if (m_filterOutXinputDevices)
        {
            g_xInputDeviceCheck.rebuildDevicesList();
        }

        DIJOYCONFIG PreferredJoyCfg = {0};
        DI_ENUM_CONTEXT enumContext;
        enumContext.pPreferredJoyCfg = &PreferredJoyCfg;
        enumContext.bPreferredJoyCfgValid = false;

        IDirectInputJoyConfig8 * pJoyConfig = nullptr;
        hr = m_pDI->QueryInterface(
                        IID_IDirectInputJoyConfig8,
                        (void **)&pJoyConfig
                        );
        if (FAILED(hr))
            return hr;

        PreferredJoyCfg.dwSize = sizeof( PreferredJoyCfg );
        
        // This function is expected to fail if no controller is attached
        hr = pJoyConfig->GetConfig(
                            0,
                            &PreferredJoyCfg,
                            DIJC_GUIDINSTANCE
                            );
        if (SUCCEEDED(hr))
        {
            enumContext.bPreferredJoyCfgValid = true;
        }

        SAFE_RELEASE(pJoyConfig);

        // Look for a simple controller we can use for this sample program.
        hr = m_pDI->EnumDevices(
                        DI8DEVCLASS_GAMECTRL,
                        EnumControllersCallback,
                        &enumContext,
                        DIEDFL_ATTACHEDONLY
                        );
        if (FAILED(hr))
            return hr;

        // Make sure we got a controller
        if (!m_pDIController)
        {
            return E_FAIL;
        }

        // Set the data format to "simple joystick" - a predefined data format.
        //  A data format specifies which controls on a device we are interested in,
        //  and how they should be reported. This tells DInput that we will be
        //  passing a DIJOYSTATE2 structure to IDirectInputDevice::GetDeviceState().
        hr = m_pDIController->SetDataFormat(&c_dfDIJoystick2);
        if (FAILED(hr))
            return hr;

        // Set the cooperative level to let DInput know how this device should
        //  interact with the system and with other DInput applications
        hr = m_pDIController->SetCooperativeLevel(
                                nullptr,
                                DISCL_NONEXCLUSIVE | DISCL_BACKGROUND
                                );
        if (FAILED(hr))
            return hr;

        // Enumerate the controller objects
        hr = m_pDIController->EnumObjects(
                            EnumObjectsCallback,
                            nullptr,
                            DIDFT_ALL
                            );
        if (FAILED(hr))
            return hr;

        return S_OK;
    }
#endif

    void InputHandler::init()
    {
        resetState();
        checkFolws();
    }

    void InputHandler::deinit()
    {
        if (m_bInputInitialized)
        {
            m_bInputInitialized = false;
            g_xInputDeviceCheck.cleanup();
#if (ENABLE_XINPUT_SUPPORT == 1)
            // Unacquire the device one last time just in case the app tried to exit while the device is still acquired
            if (m_pDIController)
                m_pDIController->Unacquire();

            SAFE_RELEASE(m_pDIController);
            SAFE_RELEASE(m_pDI);
#endif
        }
    }

    void InputHandler::addEventConsumer(InputEventsConsumerInterface* consumer)
    {
#ifdef _DEBUG
        for (auto i = m_eventConsumers.begin(), end = m_eventConsumers.end(); i != end; ++i)
        {
            if (*i == consumer)
            {
                assert(false && "Don't register keyboard event consumer twice!");
                return;
            }
        }
#endif
        m_eventConsumers.push_back(consumer);
    }

    bool InputHandler::removeEventConsumer(InputEventsConsumerInterface* consumer)
    {
        for (auto i = m_eventConsumers.begin(), end = m_eventConsumers.end(); i != end; ++i)
        {
            if (*i == consumer)
            {
                //Preserve the order. Could be more efficient if we filled the gap with the last one. Do we need to keep the order?
                m_eventConsumers.erase(i);

                return true;
            }
        }

        return false;
    }

    bool MouseTrapper::trapMouse(HWND hGameWnd, input::LwrsorVisibility& lwrsorVisib)
    {
        bool mouseInClientArea = true; //if we don't know, we assume it is
        lwrsorVisib = input::kLwrsorVisibilityUnmanaged;

        if (hGameWnd != 0)
        {
            mouseInClientArea = false;
            lwrsorVisib = input::kLwrsorVisibilityOn;

            const BOOL isInFolws = GetForegroundWindow() == hGameWnd;
            const BOOL isMinimized = IsIconic(hGameWnd);

            // Here we try to force the cursor position to stay
            //    within the current window (hwnd).  This still allows
            //    us to sample the mouse delta positions to update our 
            //    Ansel cursor - but prevent the ghost cursor issue
            //    from popping up.
            if (isInFolws && !isMinimized)
            {
                RECT clientrc;
                BOOL ok = GetClientRect(hGameWnd, &clientrc);

                POINT pos;
                ok = ok && GetLwrsorPos(&pos);

                if (ok)
                {
                    ok = ScreenToClient(hGameWnd, &pos);
                }

                if (ok)
                {
                    ok = PtInRect(&clientrc, pos);
                }

                if (ok)
                {
                    mouseInClientArea = true;

                    RECT rc;
                    GetWindowRect(hGameWnd, &rc);
                    SetLwrsorPos((rc.left + rc.right) / 2, (rc.top + rc.bottom) / 2);

                    lwrsorVisib = input::kLwrsorVisibilityOff;
                }
            }
        }

        m_mouseInClientArea = mouseInClientArea;

        return mouseInClientArea;
    }

    void InputHandlerForStandalone::update(const AnselSDKState & sdkState, AnselUI * UI)
    {
        if (!isInitialized())
            return;

        bool needToTrapMouse = UI->m_isAnselActive && sdkState.isDetected() && UI->isEnabled();
        const HWND gameWnd = needToTrapMouse ? static_cast<HWND>(sdkState.getConfiguration().gameWindowHandle) : 0;

        m_mouseState.resetCoordsAclwmulator();

        bool folwsChanged = checkFolws();
        LwrsorVisibility lwrsorVis;
        bool isMouseInClientArea = trapMouse(gameWnd, lwrsorVis);

        input::HooksAndThreadsManager::getInstance().setBlockInput(UI->isActive());
        input::HooksAndThreadsManager::getInstance().tick(hasFolws(), getForegroundThread(), lwrsorVis, true);

        const bool useKeyStateBackend = false;
        if (useKeyStateBackend)
        {
            input::InputEvent fakeInputEvent;
            m_fakeKbdState.updatePreviousState();
            m_fakeKbdState.KeyboardState::resetState();

            for (int i = 0; i < (int)m_fakeKbdState.getNumVKeys(); ++i)
            {
                USHORT vkey = (USHORT)i;
                if (0 != (0x8000 & GetAsyncKeyState(i)))
                {
                    input::KeyDownEvent ev;
                    ev.vkey = vkey;
                    m_fakeKbdState.consumeEvent(ev);
                }

                if (m_fakeKbdState.isKeyStateChanged(vkey))
                {
                    if (m_fakeKbdState.isKeyDown(vkey))
                    {
                        fakeInputEvent.type = InputEvent::Type::kKeyDown;
                        fakeInputEvent.event.keyDown.vkey = vkey;
                    }
                    else
                    {
                        fakeInputEvent.type = InputEvent::Type::kKeyUp;
                        fakeInputEvent.event.keyUp.vkey = vkey;
                    }

                    consumeEvent(fakeInputEvent);

                    m_eventConsumersLocal = m_eventConsumers; //some callbacks might change the state of the input handler, e.g. deinitialize it
                    for (auto hit = m_eventConsumersLocal.begin(), hend = m_eventConsumersLocal.end(); hit != hend; ++hit)
                    {
                        (*hit)->onInputEvent(fakeInputEvent, m_keyboardState, m_mouseState, m_gamepadState, *this, *this);
                    }
                    m_keyboardState.updatePreviousState();
                }
            }
        }

#if (ENABLE_XINPUT_SUPPORT == 1)
        const bool useXInputBackend = true;
        if (useXInputBackend)
        {
            // XInput: primary gamepad input
            {
                input::InputEvent fakeInputEvent;
                std::memset(&fakeInputEvent, 0, sizeof(fakeInputEvent));
                DWORD dwResult;
                for (DWORD i = 0; i < XUSER_MAX_COUNT; ++i)
                {
                    XINPUT_STATE xi_state;
                    ZeroMemory(&xi_state, sizeof(XINPUT_STATE));

                    dwResult = XInputGetState(i, &xi_state);

                    if (dwResult == ERROR_SUCCESS)
                    {
                        fakeInputEvent.type = InputEvent::Type::kGamepadStateUpdate;

                        input::GamepadStateUpdateEvent & gpadEvent = fakeInputEvent.event.gamepadStateUpdate;

                        // We need to ilwert Y-axes in order to match RI behavior
                        gpadEvent.axisLX = xi_state.Gamepad.sThumbLX;
                        if (xi_state.Gamepad.sThumbLY < 0)
                            gpadEvent.axisLY = -(xi_state.Gamepad.sThumbLY + 1);
                        else
                            gpadEvent.axisLY = -xi_state.Gamepad.sThumbLY;
                        gpadEvent.axisRX = xi_state.Gamepad.sThumbRX;
                        if (xi_state.Gamepad.sThumbRY < 0)
                            gpadEvent.axisRY = -(xi_state.Gamepad.sThumbRY + 1);
                        else
                            gpadEvent.axisRY = -xi_state.Gamepad.sThumbRY;

                        gpadEvent.axisZ = 0;

                        // Scale triggers to match RI behavior
                        long scaleToShort = 65536l / 255;
                        gpadEvent.axisZ += (short)(xi_state.Gamepad.bLeftTrigger * scaleToShort / 2);
                        gpadEvent.axisZ -= (short)(xi_state.Gamepad.bRightTrigger * scaleToShort / 2);

                        gpadEvent.buttonDown[EGamepadButton::kA] = ((xi_state.Gamepad.wButtons & XINPUT_GAMEPAD_A) != 0);
                        gpadEvent.buttonDown[EGamepadButton::kB] = ((xi_state.Gamepad.wButtons & XINPUT_GAMEPAD_B) != 0);
                        gpadEvent.buttonDown[EGamepadButton::kX] = ((xi_state.Gamepad.wButtons & XINPUT_GAMEPAD_X) != 0);
                        gpadEvent.buttonDown[EGamepadButton::kY] = ((xi_state.Gamepad.wButtons & XINPUT_GAMEPAD_Y) != 0);

                        gpadEvent.buttonDown[EGamepadButton::kLeftShoulder] = ((xi_state.Gamepad.wButtons & XINPUT_GAMEPAD_LEFT_SHOULDER) != 0);
                        gpadEvent.buttonDown[EGamepadButton::kRightShoulder] = ((xi_state.Gamepad.wButtons & XINPUT_GAMEPAD_RIGHT_SHOULDER) != 0);
                        gpadEvent.buttonDown[EGamepadButton::kLeftStickPress] = ((xi_state.Gamepad.wButtons & XINPUT_GAMEPAD_LEFT_THUMB) != 0);
                        gpadEvent.buttonDown[EGamepadButton::kRightStickPress] = ((xi_state.Gamepad.wButtons & XINPUT_GAMEPAD_RIGHT_THUMB) != 0);

                        bool xi_dpadLeft = ((xi_state.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_LEFT) != 0);
                        bool xi_dpadRight = ((xi_state.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_RIGHT) != 0);
                        bool xi_dpadUp = ((xi_state.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_UP) != 0);
                        bool xi_dpadDown = ((xi_state.Gamepad.wButtons & XINPUT_GAMEPAD_DPAD_DOWN) != 0);

                        // Probably EDPadDirection enum should be changed to a more logical group of states
                        gpadEvent.dpadState.value = EDPadDirection::kCenter;
                        if (xi_dpadLeft && xi_dpadRight && xi_dpadUp && xi_dpadDown)
                        {
                            gpadEvent.dpadState.value = EDPadDirection::kCenter;
                        }
                        else if (xi_dpadUp)
                        {
                            if (xi_dpadLeft)
                                gpadEvent.dpadState.value = EDPadDirection::kLeftUp;
                            else if (xi_dpadRight)
                                gpadEvent.dpadState.value = EDPadDirection::kUpRight;
                            else
                                gpadEvent.dpadState.value = EDPadDirection::kUp;
                        }
                        else if (xi_dpadDown)
                        {
                            if (xi_dpadLeft)
                                gpadEvent.dpadState.value = EDPadDirection::kDownLeft;
                            else if (xi_dpadRight)
                                gpadEvent.dpadState.value = EDPadDirection::kRightDown;
                            else
                                gpadEvent.dpadState.value = EDPadDirection::kDown;
                        }
                        else if (xi_dpadLeft)
                        {
                            gpadEvent.dpadState.value = EDPadDirection::kLeft;
                        }
                        else if (xi_dpadRight)
                        {
                            gpadEvent.dpadState.value = EDPadDirection::kRight;
                        }

                        consumeEvent(fakeInputEvent);

                        m_eventConsumersLocal = m_eventConsumers; //some callbacks might change the state of the input handler, e.g. deinitialize it
                        for (auto hit = m_eventConsumersLocal.begin(), hend = m_eventConsumersLocal.end(); hit != hend; ++hit)
                        {
                            (*hit)->onInputEvent(fakeInputEvent, m_keyboardState, m_mouseState, m_gamepadState, *this, *this);
                        }
                    }
                    else
                    {
                        // Controller is not connected
                        // TODO: we shouldn't actually query empty slots each frame, probably add deferring counter or something like that
                    }
                }
            }

            // DInput: alternative to support legacy controllers and DS3/4 on PC (as they are emulated as legacy)
            {
                input::InputEvent fakeInputEvent;

                // DInput controller state 
                DIJOYSTATE2 js;

                HRESULT hr;
                if (m_pDIController && SUCCEEDED(hr = m_pDIController->Poll()))
                {
                    if (SUCCEEDED(hr = m_pDIController->GetDeviceState( sizeof( DIJOYSTATE2 ), &js )))
                    {
                        fakeInputEvent.type = InputEvent::Type::kGamepadStateUpdate;

                        input::GamepadStateUpdateEvent & gpadEvent = fakeInputEvent.event.gamepadStateUpdate;

                        const short deadZone = 255;
                        auto scaleDInputAxis = [](LONG dinputAxis) -> short
                        {
                            short result;
                            if (dinputAxis > 0)
                            {
                                result = (short)( dinputAxis / 1000.f * SHRT_MAX);
                            }
                            else
                            {
                                result = (short)(-dinputAxis / 1000.f * SHRT_MIN);
                            }

                            return result;
                        };

                        // We need to ilwert Y-axes in order to match RI behavior
                        gpadEvent.axisLX = scaleDInputAxis(js.lX);
                        gpadEvent.axisLY = scaleDInputAxis(js.lY);

                        // DInput: right thumb maps to ZAxis(X) and ZRot(Y, negated)
                        gpadEvent.axisRX = scaleDInputAxis(js.lZ);
                        gpadEvent.axisRY = scaleDInputAxis(js.lRz);

                        gpadEvent.axisZ = 0;

                        // Scale triggers to match RI behavior
                        long scaleToShort = 65536l / 255;

                        // DInput: triggers map to XRot(Left Trigger) and YRot(Right Trigger)
                        //  lRx/lRy are in range [-1000; 1000], we need to pu them into [0; 1000]

                        // TODO avoroshilov: check triggers direction
                        short axisZpos = scaleDInputAxis((js.lRx + 1000) / 2);
                        short axisZneg = scaleDInputAxis((js.lRy + 1000) / 2);

                        gpadEvent.axisZ += axisZpos;
                        gpadEvent.axisZ -= axisZneg;

                        gpadEvent.buttonDown[EGamepadButton::kA] = ((js.rgbButtons[1] & 0x80) != 0);    // 01
                        gpadEvent.buttonDown[EGamepadButton::kB] = ((js.rgbButtons[2] & 0x80) != 0);    // 02
                        gpadEvent.buttonDown[EGamepadButton::kX] = ((js.rgbButtons[0] & 0x80) != 0);    // 00
                        gpadEvent.buttonDown[EGamepadButton::kY] = ((js.rgbButtons[3] & 0x80) != 0);    // 03

                        gpadEvent.buttonDown[EGamepadButton::kLeftShoulder] = ((js.rgbButtons[4] & 0x80) != 0);
                        gpadEvent.buttonDown[EGamepadButton::kRightShoulder] = ((js.rgbButtons[5] & 0x80) != 0);
                        gpadEvent.buttonDown[EGamepadButton::kLeftStickPress] = ((js.rgbButtons[10] & 0x80) != 0);
                        gpadEvent.buttonDown[EGamepadButton::kRightStickPress] = ((js.rgbButtons[11] & 0x80) != 0);

                        // DInput: D-pad maps to POV (u/r/d/l): 0/9000/18000/27000
                        //  combinations go half-way like this: u+r is 4500
                        switch (js.rgdwPOV[0])
                        {
                        case 0:
                            gpadEvent.dpadState.value = EDPadDirection::kUp;
                            break;
                        case 4500:
                            gpadEvent.dpadState.value = EDPadDirection::kUpRight;
                            break;
                        case 9000:
                            gpadEvent.dpadState.value = EDPadDirection::kRight;
                            break;
                        case 13500:
                            gpadEvent.dpadState.value = EDPadDirection::kRightDown;
                            break;
                        case 18000:
                            gpadEvent.dpadState.value = EDPadDirection::kDown;
                            break;
                        case 22500:
                            gpadEvent.dpadState.value = EDPadDirection::kDownLeft;
                            break;
                        case 27000:
                            gpadEvent.dpadState.value = EDPadDirection::kLeft;
                            break;
                        case 31500:
                            gpadEvent.dpadState.value = EDPadDirection::kLeftUp;
                            break;
                        default:
                            gpadEvent.dpadState.value = EDPadDirection::kCenter;
                            break;
                        }

                        consumeEvent(fakeInputEvent);

                        m_eventConsumersLocal = m_eventConsumers; //some callbacks might change the state of the input handler, e.g. deinitialize it
                        for (auto hit = m_eventConsumersLocal.begin(), hend = m_eventConsumersLocal.end(); hit != hend; ++hit)
                        {
                            (*hit)->onInputEvent(fakeInputEvent, m_keyboardState, m_mouseState, m_gamepadState, *this, *this);
                        }
                    }
                }
                else if (m_pDIController)
                {
                    // DInput is telling us that the input stream has been interrupted. We aren't tracking any state between polls,
                    //  so we don't have any special reset that needs to be done. We just re-acquire and try again.
                    hr = m_pDIController->Acquire();
                    while (hr == DIERR_INPUTLOST)
                        hr = m_pDIController->Acquire();

                    // hr may be DIERR_OTHERAPPHASPRIO or other errors. This may occur when the app is minimized or in the process of 
                    //  switching, so just try again later 
                }
            }
        }
#else
        const bool useXInputBackend = false;
#endif
        input::InputEvent mouseMoveEvent;
        mouseMoveEvent.type = InputEvent::Type::kMouseMove;

        bool mouseEventSequenceStarted = false;
        // accumulate events here
        while (const input::InputEvent* it = input::HooksAndThreadsManager::getInstance().popEvent())
        {
            if (useKeyStateBackend && (it->type == input::InputEvent::Type::kKeyDown || it->type == input::InputEvent::Type::kKeyUp))
                continue;

            if (useXInputBackend && (it->type == input::InputEvent::Type::kGamepadStateUpdate))
                continue;

            m_eventConsumersLocal = m_eventConsumers; //some callbacks might change the state of the input handler, e.g. deinitialize it
            if (it->type != InputEvent::Type::kMouseMove)
            {
                if (mouseEventSequenceStarted)
                {
                    consumeEvent(mouseMoveEvent);
                    for (auto hit = m_eventConsumersLocal.begin(), hend = m_eventConsumersLocal.end(); hit != hend; ++hit)
                        (*hit)->onInputEvent(mouseMoveEvent, m_keyboardState, m_mouseState, m_gamepadState, *this, *this);
                    mouseEventSequenceStarted = false;
                }
                consumeEvent(*it);
                for (auto hit = m_eventConsumersLocal.begin(), hend = m_eventConsumersLocal.end(); hit != hend; ++hit)
                    (*hit)->onInputEvent(*it, m_keyboardState, m_mouseState, m_gamepadState, *this, *this);
            }
            else
            {
                if (!mouseEventSequenceStarted)
                {
                    mouseEventSequenceStarted = true;
                    mouseMoveEvent.event.mouseMove.lastCoordsWheel = it->event.mouseMove.lastCoordsWheel;
                    mouseMoveEvent.event.mouseMove.lastCoordsX = it->event.mouseMove.lastCoordsX;
                    mouseMoveEvent.event.mouseMove.lastCoordsY = it->event.mouseMove.lastCoordsY;
                }
                else
                {
                    mouseMoveEvent.event.mouseMove.lastCoordsWheel += it->event.mouseMove.lastCoordsWheel;
                    mouseMoveEvent.event.mouseMove.lastCoordsX += it->event.mouseMove.lastCoordsX;
                    mouseMoveEvent.event.mouseMove.lastCoordsY += it->event.mouseMove.lastCoordsY;
                }
            }
        }

        // in case there was no non-mouse-move event in the queue but mouse move sequence was started, feed it 
        // to consumers
        if (mouseEventSequenceStarted)
        {
            consumeEvent(mouseMoveEvent);
            m_eventConsumersLocal = m_eventConsumers; //some callbacks might change the state of the input handler, e.g. deinitialize it
            for (auto hit = m_eventConsumersLocal.begin(), hend = m_eventConsumersLocal.end(); hit != hend; ++hit)
                (*hit)->onInputEvent(mouseMoveEvent, m_keyboardState, m_mouseState, m_gamepadState, *this, *this);
        }

        return;
    }

    void InputHandlerForStandalone::dormantUpdate()
    {
        if (!isInitialized())
            return;

        bool folwsChanged = checkFolws();
        LwrsorVisibility lwrsorVis;
        bool isMouseInClientArea = trapMouse(nullptr, lwrsorVis);

        input::HooksAndThreadsManager::getInstance().setBlockInput(false);
        input::HooksAndThreadsManager::getInstance().tick(hasFolws(), getForegroundThread(), lwrsorVis, false);
    }

    void InputHandlerForStandalone::init()
    {
        InputHandler::init();

        if (!m_bInputInitialized)
        {

#if (ENABLE_XINPUT_SUPPORT == 1)
            InitDirectInput();
#endif
            m_bInputInitialized = true;
        }


        if (!m_isHookBasedInputInitialized)
        {
            input::HooksAndThreadsManager::getInstance().firstTimeInit(hasFolws(), getForegroundThread());
            m_isHookBasedInputInitialized = true;
        }
    }

    void InputHandlerForStandalone::deinit()
    {
        if (m_isHookBasedInputInitialized)
        {
            input::HooksAndThreadsManager::getInstance().deinit();
            m_isHookBasedInputInitialized = false;
        }

        InputHandler::deinit();
    }

    void InputHandlerForStandalone::getGamepadStats(GamepadDevice::GamepadStats& stats) const
    {
        input::HooksAndThreadsManager::getInstance().getGamepadStats(stats);
    }

    void InputHandlerForIPC::update()
    {
        if (!isInitialized())
            return;

        m_mouseState.resetCoordsAclwmulator();

        bool folwsChanged = checkFolws();
        LwrsorVisibility lwrsorVis;
        bool isMouseInClientArea = trapMouse(NULL, lwrsorVis);

        for (auto it = m_ipcEventsQueue.begin(), end = m_ipcEventsQueue.end(); it != end; ++it)
        {
            consumeEvent(*it);

            m_eventConsumersLocal = m_eventConsumers; //some callbacks might change the state of the input handler, e.g. deinitialize it

            for (auto hit = m_eventConsumersLocal.begin(), hend = m_eventConsumersLocal.end(); hit != hend; ++hit)
            {
                (*hit)->onInputEvent(*it, m_keyboardState, m_mouseState, m_gamepadState, *this, *this);
            }
        }

        m_ipcEventsQueue.resize(0);

        return;
    }

    void InputHandlerForIPC::init()
    {
        InputHandler::init();
        if (!m_bInputInitialized)
            m_bInputInitialized = true;

        if (!m_isIpcBasedInputInitialized)
        {
            m_ipcEventsQueue.resize(0);
            m_isIpcBasedInputInitialized = true;
        }
    }

    void InputHandlerForIPC::deinit()
    {
        if (m_isIpcBasedInputInitialized)
        {
            m_ipcEventsQueue.resize(0);
            m_isIpcBasedInputInitialized = false;
        }

        InputHandler::deinit();
    }

    void InputHandlerForIPC::pushBackMouseMoveEvent(int dx, int dy, int dz)
    {
        long lastX = (long)dx;
        long lastY = (long)dy;
        long lastWheel = (long)dz;

        InputEvent ev;
        ev.type = InputEvent::Type::kMouseMove;
        ev.event.mouseMove.lastCoordsX = lastX;
        ev.event.mouseMove.lastCoordsY = lastY;
        ev.event.mouseMove.lastCoordsWheel = lastWheel;

        pushBackIpcInputEvent(ev);
    }

    void InputHandlerForIPC::pushBackMouseLButtonDownEvent(int dx, int dy, int dz)
    {
        long lastX = (long)dx;
        long lastY = (long)dy;
        long lastWheel = (long)dz;

        InputEvent ev;
        ev.type = InputEvent::Type::kMouseButtonDown;
        ev.event.mouseButtonDown.btn.value = EMouseButton::kLButton;
        ev.event.mouseButtonDown.lastCoordsX = lastX;
        ev.event.mouseButtonDown.lastCoordsY = lastY;
        ev.event.mouseButtonDown.lastCoordsWheel = lastWheel;


        pushBackIpcInputEvent(ev);
    }

    void InputHandlerForIPC::pushBackMouseLButtonUpEvent(int dx, int dy, int dz)
    {
        long lastX = (long)dx;
        long lastY = (long)dy;
        long lastWheel = (long)dz;

        InputEvent ev;
        ev.type = InputEvent::Type::kMouseButtonUp;
        ev.event.mouseButtonUp.btn.value = EMouseButton::kLButton;
        ev.event.mouseButtonUp.lastCoordsX = lastX;
        ev.event.mouseButtonUp.lastCoordsY = lastY;
        ev.event.mouseButtonUp.lastCoordsWheel = lastWheel;

        pushBackIpcInputEvent(ev);
    }

    void InputHandlerForIPC::pushBackMouseRButtonDownEvent(int dx, int dy, int dz)
    {
        long lastX = (long)dx;
        long lastY = (long)dy;
        long lastWheel = (long)dz;

        InputEvent ev;
        ev.type = InputEvent::Type::kMouseButtonDown;
        ev.event.mouseButtonDown.btn.value = EMouseButton::kRButton;
        ev.event.mouseButtonDown.lastCoordsX = lastX;
        ev.event.mouseButtonDown.lastCoordsY = lastY;
        ev.event.mouseButtonDown.lastCoordsWheel = lastWheel;

        pushBackIpcInputEvent(ev);
    }

    void InputHandlerForIPC::pushBackMouseRButtonUpEvent(int dx, int dy, int dz)
    {
        long lastX = (long)dx;
        long lastY = (long)dy;
        long lastWheel = (long)dz;

        InputEvent ev;
        ev.type = InputEvent::Type::kMouseButtonUp;
        ev.event.mouseButtonUp.btn.value = EMouseButton::kRButton;
        ev.event.mouseButtonUp.lastCoordsX = lastX;
        ev.event.mouseButtonUp.lastCoordsY = lastY;
        ev.event.mouseButtonUp.lastCoordsWheel = lastWheel;

        pushBackIpcInputEvent(ev);
    }

    void InputHandlerForIPC::pushBackMouseMButtonDownEvent(int dx, int dy, int dz)
    {
        long lastX = (long)dx;
        long lastY = (long)dy;
        long lastWheel = (long)dz;

        InputEvent ev;
        ev.type = InputEvent::Type::kMouseButtonDown;
        ev.event.mouseButtonDown.btn.value = EMouseButton::kMButton;
        ev.event.mouseButtonDown.lastCoordsX = lastX;
        ev.event.mouseButtonDown.lastCoordsY = lastY;
        ev.event.mouseButtonDown.lastCoordsWheel = lastWheel;

        pushBackIpcInputEvent(ev);
    }

    void InputHandlerForIPC::pushBackMouseMButtonUpEvent(int dx, int dy, int dz)
    {
        long lastX = (long)dx;
        long lastY = (long)dy;
        long lastWheel = (long)dz;

        InputEvent ev;
        ev.type = InputEvent::Type::kMouseButtonUp;
        ev.event.mouseButtonUp.btn.value = EMouseButton::kMButton;
        ev.event.mouseButtonUp.lastCoordsX = lastX;
        ev.event.mouseButtonUp.lastCoordsY = lastY;
        ev.event.mouseButtonUp.lastCoordsWheel = lastWheel;

        pushBackIpcInputEvent(ev);
    }

    void InputHandlerForIPC::pushBackKeyDownEvent(unsigned long vkey)
    {
        InputEvent ev;
        ev.type = InputEvent::Type::kKeyDown;
        ev.event.keyDown.vkey = (USHORT)vkey;

        pushBackIpcInputEvent(ev);
    }

    void InputHandlerForIPC::pushBackKeyUpEvent(unsigned long vkey)
    {
        InputEvent ev;

        ev.type = InputEvent::Type::kKeyUp;
        ev.event.keyUp.vkey = (USHORT)vkey;

        pushBackIpcInputEvent(ev);
    }
}
