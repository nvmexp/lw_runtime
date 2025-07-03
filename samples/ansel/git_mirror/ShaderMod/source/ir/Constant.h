#pragma once

#include "Defines.h"
#include "TypeEnums.h"

namespace shadermod
{
    class CmdProcConstantBufDesc;

namespace ir
{

    class Constant
    {
    public:
        static const int OffsetAuto = -1;

        //system constant, bound by pack offset
        Constant::Constant(ConstType type, int packOffsetInComponents):
            m_type(type), m_constantOffsetInComponents(packOffsetInComponents)
        {
            m_userConstName[0] = '\0';
            m_constBindName[0] = '\0';
        }

        //system constant, bound by name
        Constant::Constant(ConstType type, const char* name) :
            m_type(type), m_constantOffsetInComponents(OffsetAuto)
        {
            m_userConstName[0] = '\0';
            sprintf_s(m_constBindName, IR_RESOURCENAME_MAX*sizeof(char), "%s", name);
        }

        //user constant, bound by packoffset
        Constant::Constant(const char* name, int packOffsetInComponents) :
            m_type(ConstType::kUserConstantBase), m_constantOffsetInComponents(packOffsetInComponents)
        {
            m_constBindName[0] = '\0';
            sprintf_s(m_userConstName, IR_RESOURCENAME_MAX*sizeof(char), "%s", name);
        }

        //user constant, bound by name
        Constant::Constant(const char* name, const char* bindName) :
            m_type(ConstType::kUserConstantBase), m_constantOffsetInComponents(OffsetAuto)
        {
            sprintf_s(m_constBindName, IR_RESOURCENAME_MAX*sizeof(char), "%s", bindName);
            sprintf_s(m_userConstName, IR_RESOURCENAME_MAX*sizeof(char), "%s", name);
        }
        
        ConstType       m_type;
        char            m_userConstName[IR_RESOURCENAME_MAX];
        int             m_constantOffsetInComponents;
        char            m_constBindName[IR_RESOURCENAME_MAX];
        
    protected:

    };
    
    class ConstantBuf
    {
    public:
        struct Layout
        {
            struct ConstantDesc
            {
                unsigned int m_constHandle;
                int m_offsetInComponents;
            };

            std::vector<ConstantDesc> m_constantsAndOffsets;
            ID3D11Buffer*   m_D3DBuf = nullptr;
        };
                        
        void addConstant(Constant * constant)
        {
            m_constants.push_back(constant);
        }
        
        std::vector<Constant *> m_constants;
        std::vector<Layout> m_layouts;

    protected:

    };

}
}
