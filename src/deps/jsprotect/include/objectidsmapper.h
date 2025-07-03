/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2014 by LWPU Corporation. All rights reserved. All information
 * contained herein is proprietary and confidential to LWPU Corporation. Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifdef PRAGMA_ONCE_SUPPORTED
#pragma once
#endif

#ifndef OBJECTIDSMAPPER_H
#define OBJECTIDSMAPPER_H

#include <map>

#include "jscntxt.h"
#include "jsobj.h"
#include "jstypes.h"

#include "jsobjectid.h"

class ObjectIdsMapper
{
public:
    JSObject *& operator[](JSObjectId id)
    {
        return m_idToObjectMap[id];
    }

    JSObject* GetObjectById(JSObjectId id)
    {
        IdToObjectMap::const_iterator it = m_idToObjectMap.find(id);
        if (m_idToObjectMap.end() != it)
        {
            return it->second;
        }

        return nullptr;
    }

    JSObjectId GetIdByObject(JSObject *obj)
    {
        ObjectToIdMap::const_iterator it = m_objectToIdMap.find(obj);
        if (m_objectToIdMap.end() != it)
        {
            return it->second;
        }

        return 0;
    }

    static
    void JS_DLL_CALLBACK ObjectIdsMapperCallback(JSContext *cx, JSObject *obj, JSBool isNew, void *closure)
    {
        ObjectIdsMapper *This = reinterpret_cast<ObjectIdsMapper *>(closure);
        static JSObjectId idCount = 0;

        if (JS_TRUE == isNew)
        {
            This->m_objectToIdMap[obj] = ++idCount;
        }
        else
        {
            This->m_objectToIdMap.erase(obj);
        }
    }

private:
    typedef std::map<JSObject*, JSObjectId> ObjectToIdMap;
    typedef std::map<JSObjectId, JSObject*> IdToObjectMap;

    ObjectToIdMap m_objectToIdMap;
    IdToObjectMap m_idToObjectMap;
};

#endif
