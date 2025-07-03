////////////////////////////////////////////////////////////////////////////////
// Filename: modelclass.h
////////////////////////////////////////////////////////////////////////////////
#ifndef _MODELCLASS_H_
#define _MODELCLASS_H_


//////////////
// INCLUDES //
//////////////
#include <d3d10.h>
#include <d3dx10.h>
#include <fstream>
using namespace std;

#define NUM_INSTANCES 1000

////////////////////////////////////////////////////////////////////////////////
// Class name: ModelClass
////////////////////////////////////////////////////////////////////////////////
class ModelClass
{
private:
    struct VertexType
    {
        D3DXVECTOR3 position;
        D3DXVECTOR2 texcoord;
        unsigned int texarrayindex;
    };

public:
    ModelClass();
    ModelClass(const ModelClass&);
    ~ModelClass();

    bool Initialize(ID3D10Device*, HWND, ID3D10ShaderResourceView**);
    void Shutdown();
    void Render(ID3D10Device*);

    void SetMatrices(D3DXMATRIX, D3DXMATRIX, D3DXMATRIX);

private:
    bool InitializeShader(ID3D10Device*, HWND, ID3D10ShaderResourceView**);
    void ShutdownShader();

    bool InitializeBuffers(ID3D10Device*);
    void ShutdownBuffers();

    void RenderBuffers(ID3D10Device*);
    void RenderShader(ID3D10Device*);

private:
    ID3D10Effect* m_effect;
    ID3D10EffectTechnique* m_technique;
    ID3D10InputLayout* m_layout;

    ID3D10EffectMatrixVariable* m_worldMatrixPtr;
    ID3D10EffectMatrixVariable* m_viewMatrixPtr;
    ID3D10EffectMatrixVariable* m_projectionMatrixPtr;

    ID3D10EffectVectorVariable* m_meshColorPtr;
    D3DXVECTOR4 m_vMeshColor;

    ID3D10EffectShaderResourceVariable* m_DiffuseVariable;

    ID3D10Buffer* m_vertexBuffer;
    ID3D10Buffer* m_indexBuffer;

    int m_vertexCount;
    int m_indexCount;
};

#endif