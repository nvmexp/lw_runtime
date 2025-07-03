////////////////////////////////////////////////////////////////////////////////
// Filename: main.cpp
////////////////////////////////////////////////////////////////////////////////

// Please refer to http://web.archive.org/web/20140625070410/http:/rastertek.com/tutindex.html for more detailed documentation about this app. 

#include "systemclass.h"

int WINAPI WinMain/*main*/(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR pScmdline, int iCmdshow)
{
    SystemClass* System;
    bool result;
    
    int initialTris = 0, presentModel = 0, numParams = 0;
    float incrementRatio = 0, fpsLwtoff = 0, dxgiFormat = 0;
    bool FULL_SCREEN = true, bWindowedFullScreenTransition = false, bDXTLAutomationTesting = false;
    numParams = sscanf_s(pScmdline, "%d %f %f %d %f %d %d %d", &initialTris, &incrementRatio, &fpsLwtoff, &presentModel, &dxgiFormat, &FULL_SCREEN, &bWindowedFullScreenTransition, &bDXTLAutomationTesting);  

    if( (numParams == 0) || (numParams == EOF) )
    {
        initialTris = 1;
        incrementRatio = 1.0f;
        fpsLwtoff = 10.f;
        presentModel = 0;
        FULL_SCREEN = 1;
        dxgiFormat = 28; //DXGI_FORMAT_R8G8B8A8_UNORM     
        bWindowedFullScreenTransition = false;
        bDXTLAutomationTesting = false;
    }
    else if( numParams != 8)
    {
        return 0;
    }

    // Create the system object.
    System = new SystemClass;
    if(!System)
    {
        return 0;
    }

    // Initialize and run the system object.
    result = System->Initialize( initialTris, incrementRatio, fpsLwtoff, presentModel, FULL_SCREEN, (DWORD)dxgiFormat, bWindowedFullScreenTransition, bDXTLAutomationTesting);

    if(result)
    {
        System->Run();
    }

    // Shutdown and release the system object.
    System->Shutdown(FULL_SCREEN);
    delete System;
    System = 0;

    return 0;
}