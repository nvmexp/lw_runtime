#pragma once

/*
** Copyright (c) 2020 LWPU CORPORATION.  All rights reserved.
**
** LWPU CORPORATION and its licensors retain all intellectual property
** and proprietary rights in and to this software, related documentation
** and any modifications thereto.  Any use, reproduction, disclosure or
** distribution of this software and related documentation without an express
** license agreement from LWPU CORPORATION is strictly prohibited.
*/

class EventInfo
{
public:
    bool active;        // Whether mouse is on window or not

    double mouseX;      // Last position of the mouse
    double mouseY;

    bool leftPressed;
    double leftMouseX;
    double leftMouseY;

    bool rightPressed;
    double rightMouseX;
    double rightMouseY;

    bool quit;          // Signal to quit the program ("pressing  'q' key)

    EventInfo()
    {
        leftMouseX = 100.0f;     // Arbitrary initial position
        leftMouseY = 100.0f;

        active = false;
        quit   = false;

        leftPressed  = false;
        rightPressed = false;
    }
};

// Method to install our app specific keyboard/mouse event functions
void InitializeSeatListener();

// Global structure that encapsulates a subset of keyboard/mouse events
extern EventInfo eventInfo;
