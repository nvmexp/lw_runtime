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


////////////////////////////////////////////////////////////////////////////////
// Class name: GraphicsClass
////////////////////////////////////////////////////////////////////////////////
class GraphicsClass
{
public:
    GraphicsClass();
    GraphicsClass(const GraphicsClass&);
    ~GraphicsClass();

    bool Initialize(int, int, HWND, DWORD presentModel, bool FULL_SCREEN, DWORD dxgiFormat = 28, bool bWindowedFullScreenTransition = false, bool bDXTLAutomationTesting = false);
    void Shutdown();
    void Frame();
    void Render(unsigned int numTris);
    void WasteMemory(HWND hwnd, unsigned int numMB);
	D3DClass* getD3DObject() {return m_D3D;}
private:
    D3DClass* m_D3D;
    CameraClass* m_Camera;
    ModelClass* m_Model;
    ID3D10Texture3D* m_UnusedTexture;
};

#endif