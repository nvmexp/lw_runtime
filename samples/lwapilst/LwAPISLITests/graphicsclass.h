////////////////////////////////////////////////////////////////////////////////
// Filename: graphicsclass.h
////////////////////////////////////////////////////////////////////////////////
#ifndef _GRAPHICSCLASS_H_
#define _GRAPHICSCLASS_H_


/////////////
// GLOBALS //
/////////////
const bool VSYNC_ENABLED = false;
const float SCREEN_DEPTH = 1000.0f;
const float SCREEN_NEAR = 0.1f;
//bool FULL_SCREEN ;
///////////////////////
// MY CLASS INCLUDES //
///////////////////////
#include "d3dclass.h"
#include "cameraclass.h"
#include "modelclass.h"
#include "lwapi.h"


////////////////////////////////////////////////////////////////////////////////
// Class name: GraphicsClass
////////////////////////////////////////////////////////////////////////////////
class GraphicsClass
{
public:
    GraphicsClass();
    GraphicsClass(const GraphicsClass&);
    ~GraphicsClass();

    bool Initialize(int, int, HWND, DWORD, bool);
    void Shutdown();
    void Frame();
    void Render();
    void Render_TextureCopy(bool bIsOptionSetResourceHint,bool bIsOptionTrackResource, LwU32 dwFlags);
    bool Verify_Texture_Copy();
    
private:
    D3DClass* m_D3D;
    CameraClass* m_Camera;
    ModelClass* m_Model;
};

#endif