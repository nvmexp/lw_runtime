////////////////////////////////////////////////////////////////////////////////
// Filename: modelclass.cpp
////////////////////////////////////////////////////////////////////////////////
#include "modelclass.h"


ModelClass::ModelClass()
{
    m_effect = 0;
    m_technique = 0;
    m_layout = 0;

    m_worldMatrixPtr = 0;
    m_viewMatrixPtr = 0;
    m_projectionMatrixPtr = 0;

    m_meshColorPtr = 0;
    m_vMeshColor.x = 0.5f;
    m_vMeshColor.y = 0.5f;
    m_vMeshColor.z = 0.5f;
    m_vMeshColor.w = 0.5f;

    m_DiffuseVariable = NULL;

    m_vertexBuffer = 0;
    m_indexBuffer = 0;
}


ModelClass::ModelClass(const ModelClass& other)
{
}


ModelClass::~ModelClass()
{
}


bool ModelClass::Initialize(ID3D10Device* device, HWND hwnd,ID3D10ShaderResourceView** textureSRV)
{
    bool result;


    // Initialize the shader that will be used to draw the triangle.
    result = InitializeShader(device, hwnd, textureSRV);
    if(!result)
    {
        return false;
    }

    // Initialize the vertex and index buffer that hold the geometry for the triangle.
    result = InitializeBuffers(device);
    if(!result)
    {
        return false;
    }

    return true;
}


void ModelClass::Shutdown()
{
    // Release the vertex and index buffers.
    ShutdownBuffers();

    // Shutdown the shader effect.
    ShutdownShader();

    return;
}


void ModelClass::Render(ID3D10Device* device)
{
    // Put the vertex and index buffers on the graphics pipeline to prepare them for drawing.
    RenderBuffers(device);

    // Now render the prepared buffers with the shader.
    RenderShader(device);

    return;
}


bool ModelClass::InitializeShader(ID3D10Device* device, HWND hwnd,ID3D10ShaderResourceView** textureSRV)
{
    HRESULT result;
    ID3D10Blob* errorMessage;
    char* compileErrors;
    unsigned long long bufferSize, i;
    ofstream fout;
    D3D10_INPUT_ELEMENT_DESC lineLayout[3];
    unsigned int numElements;
    D3D10_PASS_DESC passDesc;


    // Initialize the error message.
    errorMessage = 0;

    // Load the shader in from the file.
    result = D3DX10CreateEffectFromFile(L"shader001.fx", NULL, NULL, "fx_4_0", D3D10_SHADER_ENABLE_STRICTNESS, 0, 
                                        device, NULL, NULL, &m_effect, &errorMessage, NULL);
    if(FAILED(result))
    {
        // If the shader failed to compile it should have writen something to the error message.
        if(errorMessage)
        {
            // Get a pointer to the error message text buffer.
            compileErrors = (char*)(errorMessage->GetBufferPointer());

            // Get the length of the message.
            bufferSize = errorMessage->GetBufferSize();

            // Open a file to write the error message to.
            fout.open("shader-error.txt");

            // Write out the error message.
            for(i=0; i<bufferSize; i++)
            {
                fout << compileErrors[i];
            }

            // Close the file.
            fout.close();

            // Release the error message.
            errorMessage->Release();
            errorMessage = 0;

            // Pop a message up on the screen to notify the user to check the text file for compile errors.
            MessageBox(hwnd, L"Error compiling shader.  Check shader-error.txt for message.", L"Shader Error", MB_OK);
        }
        else
        {
            MessageBox(hwnd, L"Could not find file: shader001.fx", L"Shader Error", MB_OK);
        }

        return false;
    }

    // Get a pointer to the technique inside the shader.
    m_technique = m_effect->GetTechniqueByName("LineTechnique");
    if(!m_technique)
    {
        return false;
    }

    // Now setup the layout of the data that goes into the shader.
    // This setup needs to match the VertexType stucture in this class and in the shader.
    lineLayout[0].SemanticName = "POSITION";
    lineLayout[0].SemanticIndex = 0;
    lineLayout[0].Format = DXGI_FORMAT_R32G32B32_FLOAT;
    lineLayout[0].InputSlot = 0;
    lineLayout[0].AlignedByteOffset = 0;
    lineLayout[0].InputSlotClass = D3D10_INPUT_PER_VERTEX_DATA;
    lineLayout[0].InstanceDataStepRate = 0;

    lineLayout[1].SemanticName = "TEXCOORD";
    lineLayout[1].SemanticIndex = 0;
    lineLayout[1].Format = DXGI_FORMAT_R32G32_FLOAT;
    lineLayout[1].InputSlot = 0;
    lineLayout[1].AlignedByteOffset = 12;
    lineLayout[1].InputSlotClass = D3D10_INPUT_PER_VERTEX_DATA;
    lineLayout[1].InstanceDataStepRate = 0;

    lineLayout[2].SemanticName = "TEXARRAYINDEX";
    lineLayout[2].SemanticIndex = 0;
    lineLayout[2].Format = DXGI_FORMAT_R32_UINT;
    lineLayout[2].InputSlot = 0;
    lineLayout[2].AlignedByteOffset = 20;
    lineLayout[2].InputSlotClass = D3D10_INPUT_PER_VERTEX_DATA;
    lineLayout[2].InstanceDataStepRate = 0;

    // Get a count of the elements in the layout.
    numElements = sizeof(lineLayout) / sizeof(lineLayout[0]);

    // Get the description of the first pass described in the shader technique.
    m_technique->GetPassByIndex(0)->GetDesc(&passDesc);

    // Create the input layout.
    result = device->CreateInputLayout(lineLayout, numElements, passDesc.pIAInputSignature, passDesc.IAInputSignatureSize, &m_layout);
    if(FAILED(result))
    {
        return false;
    }

    // Get pointers to the three matrices inside the shader so we can update them from this class.
    m_worldMatrixPtr = m_effect->GetVariableByName("worldMatrix")->AsMatrix();
    m_viewMatrixPtr = m_effect->GetVariableByName("viewMatrix")->AsMatrix();
    m_projectionMatrixPtr = m_effect->GetVariableByName("projectionMatrix")->AsMatrix();
    
    m_meshColorPtr = m_effect->GetVariableByName("vMeshColor")->AsVector();
    m_meshColorPtr->SetFloatVector((float *)m_vMeshColor);
    
    m_DiffuseVariable = m_effect->GetVariableByName("txArray0")->AsShaderResource();
    m_DiffuseVariable->SetResource(textureSRV[0]);
    m_DiffuseVariable = m_effect->GetVariableByName("txArray1")->AsShaderResource();
    m_DiffuseVariable->SetResource(textureSRV[1]);
    m_DiffuseVariable = m_effect->GetVariableByName("txArray2")->AsShaderResource();
    m_DiffuseVariable->SetResource(textureSRV[2]);

    return true;
}


void ModelClass::ShutdownShader()
{
    // Release the pointers to the matrices inside the shader.
    m_worldMatrixPtr = 0;
    m_viewMatrixPtr = 0;
    m_projectionMatrixPtr = 0;

    // Release the pointer to the shader layout.
    if(m_layout)
    {
        m_layout->Release();
        m_layout = 0;
    }

    // Release the pointer to the shader technique.
    m_technique = 0;

    // Release the pointer to the shader.
    if(m_effect)
    {
        m_effect->Release();
        m_effect = 0;
    }

    return;
}


bool ModelClass::InitializeBuffers(ID3D10Device* device)
{
    VertexType* vertices;
    D3D10_BUFFER_DESC vertexBufferDesc;
    D3D10_SUBRESOURCE_DATA vertexData;
    unsigned long* indices;
    D3D10_BUFFER_DESC indexBufferDesc;
    D3D10_SUBRESOURCE_DATA indexData;
    HRESULT result;

    
    // Set the number of vertices in the vertex array.
    //m_vertexCount = 3;
    m_vertexCount = 12;

    // Create the vertex array.
    vertices = new VertexType[m_vertexCount];
    if(!vertices)
    {
        return false;
    }

    // Set the number of indices in the index array.
    //m_indexCount = 3*NUM_INSTANCES;
    m_indexCount = 18;

    // Create the index array.
    indices = new unsigned long[m_indexCount];
    if(!indices)
    {
        return false;
    }

    // First quad
    vertices[0].position      = D3DXVECTOR3(-1.0f, -2.0f, 0.0f);
    vertices[0].texcoord      = D3DXVECTOR2(0.0f, 0.0f);
    vertices[0].texarrayindex = (unsigned int)1;
    
    vertices[1].position      = D3DXVECTOR3(1.0f, -2.0f, 0.0f);
    vertices[1].texcoord      = D3DXVECTOR2(1.0f, 0.0f);
    vertices[1].texarrayindex = (unsigned int)1;
    
    vertices[2].position      = D3DXVECTOR3(-1.0f, 2.0f, 0.0f);
    vertices[2].texcoord      = D3DXVECTOR2(0.0f, 1.0f);
    vertices[2].texarrayindex = (unsigned int) 1;
    
    vertices[3].position      = D3DXVECTOR3(1.0f, 2.0f, 0.0f);
    vertices[3].texcoord      = D3DXVECTOR2(1.0f, 1.0f);
    vertices[3].texarrayindex = (unsigned int) 1;

    // Second quad
    vertices[4].position      = D3DXVECTOR3(-4.0f, -2.0f, 0.0f);
    vertices[4].texcoord      = D3DXVECTOR2(0.0f, 0.0f);
    vertices[4].texarrayindex = 0;
    
    vertices[5].position      = D3DXVECTOR3(-2.0f, -2.0f, 0.0f);
    vertices[5].texcoord      = D3DXVECTOR2(1.0f, 0.0f);
    vertices[5].texarrayindex = 0;
    
    vertices[6].position      = D3DXVECTOR3(-4.0f, 2.0f, 0.0f);
    vertices[6].texcoord      = D3DXVECTOR2(0.0f, 1.0f);
    vertices[6].texarrayindex = 0;
    
    vertices[7].position      = D3DXVECTOR3(-2.0f, 2.0f, 0.0f);
    vertices[7].texcoord      = D3DXVECTOR2(1.0f, 1.0f);
    vertices[7].texarrayindex = 0;

    // Third quad
    vertices[8].position      = D3DXVECTOR3(2.0f, -2.0f, 0.0f);
    vertices[8].texcoord      = D3DXVECTOR2(0.0f, 0.0f);
    vertices[8].texarrayindex = 2;
    
    vertices[9].position      = D3DXVECTOR3(4.0f, -2.0f, 0.0f);
    vertices[9].texcoord      = D3DXVECTOR2(1.0f, 0.0f);
    vertices[9].texarrayindex = 2;
    
    vertices[10].position      = D3DXVECTOR3(2.0f, 2.0f, 0.0f);
    vertices[10].texcoord      = D3DXVECTOR2(0.0f, 1.0f);
    vertices[10].texarrayindex = 2;
    
    vertices[11].position      = D3DXVECTOR3(4.0f, 2.0f, 0.0f);
    vertices[11].texcoord      = D3DXVECTOR2(1.0f, 1.0f);
    vertices[11].texarrayindex = 2;

    indices[0] = 0;
    indices[1] = 2;
    indices[2] = 1;
    indices[3] = 1;
    indices[4] = 2;
    indices[5] = 3;

    indices[6] = 4;
    indices[7] = 6;
    indices[8] = 5;
    indices[9] = 5;
    indices[10] = 6;
    indices[11] = 7;

    indices[12] = 8;
    indices[13] = 10;
    indices[14] = 9;
    indices[15] = 9;
    indices[16] = 10;
    indices[17] = 11;
    
    // Set up the description of the vertex buffer.
    vertexBufferDesc.Usage = D3D10_USAGE_DEFAULT;
    vertexBufferDesc.ByteWidth = sizeof(VertexType) * m_vertexCount;
    vertexBufferDesc.BindFlags = D3D10_BIND_VERTEX_BUFFER;
    vertexBufferDesc.CPUAccessFlags = 0;
    vertexBufferDesc.MiscFlags = 0;

    // Give the subresource structure a pointer to the vertex data.
    vertexData.pSysMem = vertices;

    // Now finally create the vertex buffer.
    result = device->CreateBuffer(&vertexBufferDesc, &vertexData, &m_vertexBuffer);
    if(FAILED(result))
    {
        return false;
    }

    // Set up the description of the index buffer.
    indexBufferDesc.Usage = D3D10_USAGE_DEFAULT;
    indexBufferDesc.ByteWidth = sizeof(unsigned long) * m_indexCount;
    indexBufferDesc.BindFlags = D3D10_BIND_INDEX_BUFFER;
    indexBufferDesc.CPUAccessFlags = 0;
    indexBufferDesc.MiscFlags = 0;

    // Give the subresource structure a pointer to the index data.
    indexData.pSysMem = indices;

    // Create the index buffer.
    result = device->CreateBuffer(&indexBufferDesc, &indexData, &m_indexBuffer);
    if(FAILED(result))
    {
        return false;
    }

    // Release the arrays now that the vertex and index buffers have been created and loaded.
    delete [] vertices;
    vertices = 0;

    delete [] indices;
    indices = 0;

    return true;
}


void ModelClass::ShutdownBuffers()
{
    // Release the index buffer.
    if(m_indexBuffer)
    {
        m_indexBuffer->Release();
        m_indexBuffer = 0;
    }

    // Release the vertex buffer.
    if(m_vertexBuffer)
    {
        m_vertexBuffer->Release();
        m_vertexBuffer = 0;
    }

    return;
}


void ModelClass::RenderBuffers(ID3D10Device* device)
{
    unsigned int stride;
    unsigned int offset;


    // Set vertex buffer stride and offset.
    stride = sizeof(VertexType); 
    offset = 0;
    
    // Set the vertex buffer to active in the input assembler so it can be rendered.
    device->IASetVertexBuffers(0, 1, &m_vertexBuffer, &stride, &offset);

    // Set the index buffer to active in the input assembler so it can be rendered.
    device->IASetIndexBuffer(m_indexBuffer, DXGI_FORMAT_R32_UINT, 0);

    // Set the type of primitive that should be rendered from this vertex buffer, in this case lines.
    device->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    return;
}


void ModelClass::RenderShader(ID3D10Device* device)
{
    D3D10_TECHNIQUE_DESC techniqueDesc;
    unsigned int i;
    

    // Set the input layout.
    device->IASetInputLayout(m_layout);

    // Get the description structure of the technique from inside the shader so it can be used for rendering.
    m_technique->GetDesc(&techniqueDesc);

    
    // Go through each pass in the technique (should be just one lwrrently) and render the triangles.
    for(i=0; i<techniqueDesc.Passes; ++i)
    {
        m_technique->GetPassByIndex(i)->Apply(0);
        //DWORD numIndices = 3 * numTris;
        //numIndices = 18;
        /*while( numIndices )
        {
            if( numIndices > (unsigned int)m_indexCount )
            {
                device->DrawIndexed(m_indexCount, 0, 0);
                numIndices -= m_indexCount;
            }
            else
            {
                device->DrawIndexed(numIndices, 0, 0);
                numIndices = 0;
            }
        }*/

        device->DrawIndexed(m_indexCount, 0, 0);
    }
    
    return;
}


void ModelClass::SetMatrices(D3DXMATRIX worldMatrix, D3DXMATRIX viewMatrix, D3DXMATRIX projectionMatrix)
{
    // Set the world matrix variable inside the shader.
    m_worldMatrixPtr->SetMatrix((float*)&worldMatrix);

    // Set the view matrix variable inside the shader.
    m_viewMatrixPtr->SetMatrix((float*)&viewMatrix);

    // Set the projection matrix variable inside the shader.
    m_projectionMatrixPtr->SetMatrix((float*)&projectionMatrix);

    return;
}