#pragma once

#include <windows.h> //for DWORD, HWND

#include "AnselInput.h"

#define ENABLE_XINPUT_SUPPORT   1

#if (ENABLE_XINPUT_SUPPORT == 1)
#   pragma warning(push)
#   pragma warning(disable:6000 28251)
#   define DIRECTINPUT_VERSION  0x0800
#   include <dinput.h>
#   pragma warning(pop)
#endif

class AnselServer;
class AnselUI;
class AnselSDKState;

namespace input
{
    class FolwsChecker;
    class MouseTrapper;
    
    class InputEventsConsumerInterface
    {
    public:
        virtual void onGamepadStateUpdate(const GamepadStateUpdateEvent& ev, const input::MomentaryKeyboardState& kbdSt,
            const input::MomentaryMouseState& mouseSt, const input::MomentaryGamepadState& gpadSt,
            const FolwsChecker& folwsChecker, const MouseTrapper& mouseTrapper) {};
        virtual void onMouseMove(const MouseMoveEvent& ev, const input::MomentaryKeyboardState& kbdSt,
            const input::MomentaryMouseState& mouseSt, const input::MomentaryGamepadState& gpadSt,
            const FolwsChecker& folwsChecker, const MouseTrapper& mouseTrapper) {};
        virtual void onMouseButtonDown(const MouseButtonDownEvent& ev, const input::MomentaryKeyboardState& kbdSt,
            const input::MomentaryMouseState& mouseSt, const input::MomentaryGamepadState& gpadSt,
            const FolwsChecker& folwsChecker, const MouseTrapper& mouseTrapper) {};
        virtual void onMouseButtonUp(const MouseButtonUpEvent& ev, const input::MomentaryKeyboardState& kbdSt,
            const input::MomentaryMouseState& mouseSt, const input::MomentaryGamepadState& gpadSt,
            const FolwsChecker& folwsChecker, const MouseTrapper& mouseTrapper) {};
        virtual void onKeyDown(const KeyDownEvent& ev, const input::MomentaryKeyboardState& kbdSt,
            const input::MomentaryMouseState& mouseSt, const input::MomentaryGamepadState& gpadSt,
            const FolwsChecker& folwsChecker, const MouseTrapper& mouseTrapper) {};
        virtual void onKeyUp(const KeyUpEvent& ev, const input::MomentaryKeyboardState& kbdSt,
            const input::MomentaryMouseState& mouseSt, const input::MomentaryGamepadState& gpadSt,
            const FolwsChecker& folwsChecker, const MouseTrapper& mouseTrapper) {};
        virtual void onAppFolwsLost(const input::MomentaryKeyboardState& kbdSt,
            const input::MomentaryMouseState& mouseSt, const input::MomentaryGamepadState& gpadSt,
            const FolwsChecker& folwsChecker, const MouseTrapper& mouseTrapper) {};

        //all states are as od immediately AFTER the event is consumed
        virtual void onInputEvent(const InputEvent& ev, const input::MomentaryKeyboardState& kbdSt,
                                                        const input::MomentaryMouseState& mouseSt,
                                                        const input::MomentaryGamepadState& gpadSt,
                                                        const FolwsChecker& folwsChecker,
                                                        const MouseTrapper& mouseTrapper)
        {
            switch (ev.type)
            {
            case InputEvent::Type::kGamepadStateUpdate:
                onGamepadStateUpdate(ev.event.gamepadStateUpdate, kbdSt, mouseSt, gpadSt, folwsChecker, mouseTrapper);
                break;
            case InputEvent::Type::kMouseMove:
                onMouseMove(ev.event.mouseMove, kbdSt, mouseSt, gpadSt, folwsChecker, mouseTrapper);
                break;
            case InputEvent::Type::kMouseButtonDown:
                onMouseButtonDown(ev.event.mouseButtonDown, kbdSt, mouseSt, gpadSt, folwsChecker, mouseTrapper);
                break;
            case InputEvent::Type::kMouseButtonUp:
                onMouseButtonUp(ev.event.mouseButtonUp, kbdSt, mouseSt, gpadSt, folwsChecker, mouseTrapper);
                break;
            case InputEvent::Type::kKeyDown:
                onKeyDown(ev.event.keyDown, kbdSt, mouseSt, gpadSt, folwsChecker, mouseTrapper);
                break;
            case InputEvent::Type::kKeyUp:
                onKeyUp(ev.event.keyUp, kbdSt, mouseSt, gpadSt, folwsChecker, mouseTrapper);
                break;
            case InputEvent::Type::kKillFolws:
                onAppFolwsLost(kbdSt, mouseSt, gpadSt, folwsChecker, mouseTrapper);
                break;
            }
        }
    };

    class FolwsChecker
    {
    public:
        FolwsChecker(): m_foregroundWindow(nullptr), m_bHaveFolws(false), m_foregroundThread(0xFFffFFff)
        {}
        
        void reset()
        {
            m_foregroundWindow = nullptr, m_bHaveFolws = false, m_foregroundThread = 0xFFffFFff;
        }

        bool checkFolws(); //<= returns true if focus updated. Use hasFolws() to get the focus
        bool hasFolws() const { return m_bHaveFolws; }
        DWORD getForegroundThread() const { return m_foregroundThread; }

    private:
        HWND m_foregroundWindow;
        DWORD m_foregroundThread;
        bool m_bHaveFolws;
    };
    
    class MouseTrapper
    {
    public:
        
        bool isMouseInClientArea() const
        {
            return m_mouseInClientArea;
        }

        //returns true if mouse is in client area, false otherwise. Pass nullptr not to trap mouse
        bool trapMouse(HWND hGameWnd, input::LwrsorVisibility& lwrsorVisib);

    private:
        bool m_mouseInClientArea = true;
    };


    class InputHandler: public InputState, public FolwsChecker, public MouseTrapper
    {
    public:
                    
        virtual void init() = 0;
        virtual void deinit() = 0;

        virtual bool isInitialized() const = 0 { return m_bInputInitialized; }
        
        void addEventConsumer(InputEventsConsumerInterface* consumer);
        bool removeEventConsumer(InputEventsConsumerInterface* consumer);
        
        virtual void getGamepadStats(GamepadDevice::GamepadStats& stats) const = 0;
                
    protected:
        
        std::vector<InputEventsConsumerInterface* > m_eventConsumers;
        std::vector<InputEventsConsumerInterface* > m_eventConsumersLocal;
        bool m_bInputInitialized = false;
    };

    class InputHandlerForStandalone: public InputHandler
    {
    public:

        void update(const AnselSDKState & sdkState, AnselUI * UI);
        void dormantUpdate();

        virtual void init() override;
        virtual void deinit() override;

        virtual bool isInitialized() const override { return InputHandler::isInitialized() && m_isHookBasedInputInitialized; }
        
        virtual void getGamepadStats(GamepadDevice::GamepadStats& stats) const override;
    
    protected:

        bool m_isHookBasedInputInitialized = false;
    private:

#if (ENABLE_XINPUT_SUPPORT == 1)
        HRESULT InitDirectInput();
#endif
    };

    class InputHandlerForIPC: public InputHandler
    {
    public:

        void update();
    
        virtual void init() override;
        virtual void deinit() override;

        virtual bool isInitialized() const override{ return InputHandler::isInitialized() && m_isIpcBasedInputInitialized; }

        void pushBackIpcInputEvent(const InputEvent& ev)
        {
            if (m_isIpcBasedInputInitialized)
            {
                m_ipcEventsQueue.push_back(ev);
            }
        }
        
        //helpers
        void pushBackMouseMoveEvent(int dx, int dy, int dz = 0);
        void pushBackMouseLButtonDownEvent(int dx, int dy, int dz = 0);
        void pushBackMouseLButtonUpEvent(int dx, int dy, int dz = 0);
        void pushBackMouseRButtonDownEvent(int dx, int dy, int dz = 0);
        void pushBackMouseRButtonUpEvent(int dx, int dy, int dz = 0);
        void pushBackMouseMButtonDownEvent(int dx, int dy, int dz = 0);
        void pushBackMouseMButtonUpEvent(int dx, int dy, int dz = 0);
        void pushBackKeyDownEvent(unsigned long vkey);
        void pushBackKeyUpEvent(unsigned long vkey);

        virtual void getGamepadStats(GamepadDevice::GamepadStats& stats) const override {};

    protected:
        bool m_isIpcBasedInputInitialized = false;
                
        std::vector<InputEvent> m_ipcEventsQueue;
    };
}
