#include "ir/UserConstantManager.h"

namespace shadermod
{
namespace ir
{
            
    UserConstantManager::~UserConstantManager()
    {
        for (auto& uc : m_validUserConstants)
            m_userConstants.deleteElement(uc);
    }

    void UserConstantManager::destroyAllUserConstants()
    {
        for (auto rit = m_validUserConstants.rbegin(); rit != m_validUserConstants.rend(); ++rit)
        {
            m_userConstantsByName.erase((*rit)->getName());
            m_userConstantsById.erase((*rit)->getUid());
            m_freeUIDs.push((*rit)->getUid());

            m_userConstants.deleteElement(*rit);
        }
        m_validUserConstants.resize(0);
    }

    void UserConstantManager::destroyUserConstant(UserConstant* uc)
    {
        unsigned int idx = uc->getPosition();
        assert(idx >= 0 && idx < m_validUserConstants.size());
        m_validUserConstants.erase(m_validUserConstants.begin() + idx);
        m_userConstantsByName.erase(uc->getName());
        m_userConstantsById.erase(uc->getUid());
        m_freeUIDs.push(uc->getUid());

        m_userConstants.deleteElement(uc);
    }

    const UserConstant* UserConstantManager::findByName(const std::string& name) const
    {
        auto it = m_userConstantsByName.find(name);

        if (it == m_userConstantsByName.end())
            return nullptr;

        return it->second;
    }

    UserConstant* UserConstantManager::findByName(const std::string& name)
    {
        auto it = m_userConstantsByName.find(name);

        if (it == m_userConstantsByName.end())
            return nullptr;

        return it->second;
    }

    const UserConstant* UserConstantManager::findByUid(unsigned int uid) const
    {
        auto it = m_userConstantsById.find(uid);

        if (it == m_userConstantsById.end())
            return nullptr;

        return it->second;
    }

    UserConstant* UserConstantManager::findByUid(unsigned int uid)
    {
        auto it = m_userConstantsById.find(uid);

        if (it == m_userConstantsById.end())
            return nullptr;

        return it->second;
    }
}
}
