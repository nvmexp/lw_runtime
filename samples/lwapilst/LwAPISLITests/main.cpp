////////////////////////////////////////////////////////////////////////////////
// Filename: main.cpp
////////////////////////////////////////////////////////////////////////////////
#include "systemclass.h"

// LwAPISLITests
// Version : 1.0


int WINAPI WinMain/*main*/(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR pScmdline, int iCmdshow)
{
    SystemClass* System;
    bool result;
    
    FILE* fp;
    errno_t err;
    err = fopen_s(&fp,"main_log.txt","w");
    
    DWORD numParams = 0;
    bool FULL_SCREEN = true;
    //bool FULL_SCREEN = false;
    bool bSetResourceHint = false, bUseBeginEndCalls = false;
    unsigned long dwBeginFlags = 32;
    char option1[3], option2[3];
    unsigned int offset[2] = {0,0};
    
    numParams = sscanf_s(pScmdline, "%s %n %s %u %n",option1,_countof(option1),&offset[0],option2,_countof(option2),&dwBeginFlags,&offset[1]);
    
    for (int i=1;i>=0;i--)
    {
        if (offset[i])
        {
            break;
        }    
    }

    if (offset[0])
    {
        numParams += sscanf_s((pScmdline + offset[0]), "\n %u %s \n",&dwBeginFlags,option2,_countof(option2));
    }

    bool strcmpRes1B = (strcmp(option1,"/b") == 0);
    bool strcmpRes1S = (strcmp(option1,"/s") == 0);
    bool strcmpRes2B = (strcmp(option2,"/b") == 0);
    bool strcmpRes2S = (strcmp(option2,"/s") == 0);
    
    if (numParams == 1)
    {
        if (strcmpRes1S)
        {
            bSetResourceHint = true;
            fprintf(fp, "\n Option to use LwAPI_D3D_SetResourceHint provided \n"); 
        }
        else if (strcmpRes1B)
        {
            bUseBeginEndCalls = true;
            fprintf(fp, "\n Option to use LwAPI_D3D_BeginResourceRendering provided. \n");
            fprintf(fp, "\n No flags specified, hence the LWAPI_D3D_RR_FLAG_DEFAULTS flag shall be used. \n");
            dwBeginFlags = 0; // LWAPI_D3D_RR_FLAG_DEFAULTS
        }
        else
        {
            fprintf(fp, "\n Incorrect input format!! \n Supported options : \n /s - Use LwAPI_D3D_SetResourceHint \n /b [Flags] - Use LwAPI_D3D_BeginResourceRendering \n");
        }
    }
    
    if (numParams == 2)
    {
        if (strcmpRes1B)
        {
            bUseBeginEndCalls = true;
            if (strcmpRes2S)
            {
                bSetResourceHint = true;
                fprintf(fp,"\n option to use LwAPI_D3D_SetResourceHint and LwAPI_D3D_BeginResourceRendering provided \n");
                fprintf(fp, "\n No flags specified, hence the LWAPI_D3D_RR_FLAG_DEFAULTS flag shall be used. \n");
                dwBeginFlags = 0;
            }
            else if (offset[0])
            {
                fprintf(fp,"\n option to use LwAPI_D3D_BeginResourceRendering with flag value %u provided \n",dwBeginFlags);
            }
            else
            {
                fprintf(fp, "\n Incorrect input format!! \n Supported options : \n /s - Use LwAPI_D3D_SetResourceHint \n /b [Flags] - Use LwAPI_D3D_BeginResourceRendering \n");
            }
        }
        else if (strcmpRes1S)
        {
            bSetResourceHint = true;
            if (strcmpRes2B)
            {
                bUseBeginEndCalls = 0;
                fprintf(fp,"\n option to use LwAPI_D3D_SetResourceHint and LwAPI_D3D_BeginResourceRendering provided \n");
                fprintf(fp, "\n No flags specified, hence the LWAPI_D3D_RR_FLAG_DEFAULTS flag shall be used. \n");
                dwBeginFlags = 0;
            }
            else
            {
                fprintf(fp, "\n Incorrect input format!! \n Supported options : \n /s - Use LwAPI_D3D_SetResourceHint \n /b [Flags] - Use LwAPI_D3D_BeginResourceRendering \n");
            }
        }
    }

    if (numParams == 3)
    {
        if (((strcmpRes1S)||(strcmpRes1B)) && ((strcmpRes2S)||(strcmpRes2B)) && (strcmp(option1,option2)))
        {
            bSetResourceHint = true;
            bUseBeginEndCalls = true;
            fprintf(fp,"\n option to use LwAPI_D3D_SetResourceHint and LwAPI_D3D_BeginResourceRendering with flag value %u provided \n",dwBeginFlags);
        }
        else
        {
            fprintf(fp, "\n Incorrect input format!! \n Supported options : \n /s - Use LwAPI_D3D_SetResourceHint \n /b [Flags] - Use LwAPI_D3D_BeginResourceRendering \n");
        }
    }
    // Create the system object.
    System = new SystemClass;
    if(!System)
    {
        fclose(fp);
        return 0;
    }

    // Initialize and run the system object.
    result = System->Initialize(FULL_SCREEN);
    if(result)
    {
        System->Run(bSetResourceHint,bUseBeginEndCalls,dwBeginFlags);
    }

    // Shutdown and release the system object.
    System->Shutdown(FULL_SCREEN);
    delete System;
    System = 0;
    
    fprintf(fp,"\n Ops completed \n");
    fclose(fp);

    return 0;
}