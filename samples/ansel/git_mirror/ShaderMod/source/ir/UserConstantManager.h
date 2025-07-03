#pragma once

#include <vector>
#include <map>
#include <stack>
#include <string>
#include <assert.h>

#include "ir/SpecializedPool.h"
#include "ir/UserConstant.h"

namespace shadermod
{
namespace ir
{
    class UserConstantManager
    {
    public:

        ~UserConstantManager();

        template<typename ... Ts>
        UserConstant* insertUserConstantAt(int position, Ts&& ... constructorArgs)
        {
            if (position < 0)
                position = (unsigned int)m_validUserConstants.size();

            assert((unsigned int) position <= m_validUserConstants.size());

            UserConstant* uc = m_userConstants.newElement(std::forward<Ts>(constructorArgs) ...);
            auto nit = m_userConstantsByName.insert(std::make_pair(uc->getName(), uc));

            if (!nit.second) //duplicate name
            {
                m_userConstants.deleteElement(uc);

                return nullptr;
            }

            unsigned int newUid;

            if (m_freeUIDs.size())
            {
                newUid = m_freeUIDs.top();
                m_freeUIDs.pop();
            }
            else
            {
                newUid = m_firstUnassignedUid++;
            }

            auto uit = m_userConstantsById.insert(std::make_pair(newUid, uc));
            assert(uit.second);

            uc->setPosition(position);
            uc->setUid(newUid);
            m_validUserConstants.insert(m_validUserConstants.begin() + position, uc);

            return uc;
        }

        template<typename ... Ts>
        UserConstant* pushBackUserConstant(Ts&& ... constructorArgs)
        {
            return insertUserConstantAt(-1, std::forward<Ts>(constructorArgs)...);
        }

        void destroyAllUserConstants();
        void destroyUserConstant(UserConstant* uc);
        const UserConstant* findByName(const std::string& name) const;
        UserConstant* findByName(const std::string& name);
        const UserConstant* findByUid(unsigned int uid) const;
        UserConstant* findByUid(unsigned int uid);

        unsigned int getNumUserConstants() const
        {
            return (unsigned int)m_validUserConstants.size();
        }

        UserConstant* getUserConstantByIndex(unsigned int idx)
        {
            return m_validUserConstants[idx];
        }

        const UserConstant* getUserConstantByIndex(unsigned int idx) const
        {
            return m_validUserConstants[idx];
        }

        struct ConstRange
        {
            const UserConstant*const* begin;
            const UserConstant*const* end;
        };

        struct Range
        {
            UserConstant*const* begin;
            UserConstant*const* end;
        };

        ConstRange getPointersToAllUserConstants() const
        {
            return{ m_validUserConstants.data(), m_validUserConstants.data() + m_validUserConstants.size() };
        }

        Range getPointersToAllUserConstants()
        {
            return{ m_validUserConstants.data(), m_validUserConstants.data() + m_validUserConstants.size() };
        }

        //thise methods are for the shadermod internals only
        bool isConstantUpdated(unsigned int idx) const
        {
            return m_validUserConstants[idx]->isUpdated();
        }

        void markConstantClean(unsigned int idx)
        {
            m_validUserConstants[idx]->markClean();
        }

        bool getConstantValue(unsigned int idx, void* buf, size_t bufSz) const
        {
            return m_validUserConstants[idx]->getValue(buf, bufSz);
        }

    protected:

        std::map<std::string, UserConstant*>  m_userConstantsByName;
        std::map<unsigned int, UserConstant*>  m_userConstantsById;
        std::vector<UserConstant*>        m_validUserConstants;
        std::stack<unsigned int>        m_freeUIDs;
        unsigned int              m_firstUnassignedUid = 0;
        ir::Pool<UserConstant>          m_userConstants;
    };
}
}