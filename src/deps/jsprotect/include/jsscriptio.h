/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2014-2019 by LWPU Corporation. All rights reserved. All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation. Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifdef PRAGMA_ONCE_SUPPORTED
#pragma once
#endif

#ifndef JSSCRIPTIO_H
#define JSSCRIPTIO_H

#include <vector>

#include <boost/type_traits.hpp>
#include <boost/utility/enable_if.hpp>

#include "jsapi.h"
#include "jsarena.h"
#include "jsatom.h"
#include "jsemit.h"
#include "jsfun.h"
#include "jsinterp.h"
#include "jsobj.h"
#include "jspubtd.h"
#include "jsregexp.h"
#include "jsscope.h"
#include "jsscript.h"
#include "jsstr.h"
#include "jsutil.h"

#include "align.h"

#include "jsobjectid.h"
#include "inbitstream.h"
#include "outbitstream.h"

enum ObjectType
{
    OBJ_TYPE_BLOCK,
    OBJ_TYPE_FUNCTION,
    OBJ_TYPE_REGEXP,
    OBJ_TYPE_UNKNOWN
};

enum FunctionProperyType
{
    FUN_PROP_LOCAL_VAR,
    FUN_PROP_ARG,
    FUN_PROP_UNKNOWN
};

template <class OutputIterator, class JSObjectMap>
class JSScriptSaver
{
public:
    JSScriptSaver(const BitsIO::OutBitStream<OutputIterator> &os,
                  JSContext *cx, JSObjectMap *objMap)
      : m_os(os)
      , m_cx(cx)
      , m_objMap(objMap)
    {}

    void Save(JSScript *jsScript)
    {
        jssrcnote *notes = SCRIPT_NOTES(jsScript);
        jssrcnote *sn;

        for (sn = notes; !SN_IS_TERMINATOR(sn); sn = SN_NEXT(sn));
        uint32 srcNotesLength = static_cast<uint32>((sn - notes + 1) * sizeof(*sn));

        Save(jsScript->length);
        Save(srcNotesLength);

        uint32 numTryNotes = 0;
        if (nullptr != jsScript->trynotes)
        {
            numTryNotes = 1;
            for (JSTryNote *tn = jsScript->trynotes; 0 != tn->catchStart; ++tn, ++numTryNotes);
        }

        Save(numTryNotes);

        SaveBytecode(jsScript->code, jsScript->length);
        Save(PTRDIFF(jsScript->main, jsScript->code, jsbytecode));
        Save(jsScript->version);
        Save(jsScript->numGlobalVars);
        Save(jsScript->depth);
        Save(jsScript->lineno);

        SaveSrcNotes(notes, srcNotesLength);

        if (0 != numTryNotes)
        {
            JSTryNote *tn = jsScript->trynotes;
            do
            {
                Save(*tn);
            } while ((tn++)->catchStart != 0);
        }

        Save(jsScript->atomMap, jsScript->atomMap.length);
    }

private:
    template <class T>
    void Save(
        T t,
        typename boost::enable_if_c<
            boost::is_integral<T>::value && boost::is_unsigned<T>::value
          , T
          >::type *dummy = nullptr
    )
    {
        UINT64 v = t + 1;
        UINT32 numBits = 64;
        if (v <= 0x00000000FFFFFFFFULL) { numBits -= 32; v <<= 32; }
        if (v <= 0x0000FFFFFFFFFFFFULL) { numBits -= 16; v <<= 16; }
        if (v <= 0x00FFFFFFFFFFFFFFULL) { numBits -= 8;  v <<= 8; }
        if (v <= 0x0FFFFFFFFFFFFFFFULL) { numBits -= 4;  v <<= 4; }
        if (v <= 0x3FFFFFFFFFFFFFFFULL) { numBits -= 2;  v <<= 2; }
        if (v <= 0x7FFFFFFFFFFFFFFFULL) { numBits -= 1; }
        m_os.Write(0, numBits - 1);
        m_os.Write(t + 1, numBits);
    }

    template <class T>
    void Save(
        T t,
        typename boost::enable_if_c<
            boost::is_integral<T>::value && !boost::is_unsigned<T>::value
          , T
          >::type *dummy = nullptr
    )
    {
        UINT64 v = 2 * abs(t);
        if (t > 0) --v;
        Save(v);
    }

    void Save(double v)
    {
        static_assert(sizeof(double) == sizeof(UINT64), "size of double is not 64 bits");
        union { UINT64 ui; double d; } u;
        u.d = v;
        Save(u.ui);
    }

    void Save(const JSAtom *atom)
    {
        if (ATOM_IS_OBJECT(atom))
        {
            JSObject *jsObj = ATOM_TO_OBJECT(atom);
            JSClass *jsClass = OBJ_GET_CLASS(m_cx, jsObj);

#if defined(JS_USE_STRONGLY_TYPED_JSVAL_AND_JSID)
            Save(static_cast<jsval::TagType>(JSVAL_OBJECT));
#else
            Save(static_cast<jsval>(JSVAL_OBJECT));
#endif
            if (&js_BlockClass == jsClass)
            {
                Save(static_cast<uint8>(OBJ_TYPE_BLOCK));
                JSObject *parent = JSVAL_TO_OBJECT(jsObj->slots[JSSLOT_PARENT]);

                JSObjectId thisId = m_objMap->GetIdByObject(jsObj);
                JSObjectId parentId = 0;
                if (0 != parent)
                {
                    parentId = m_objMap->GetIdByObject(parent);
                }
                Save(thisId);
                Save(parentId);

                Save(OBJ_BLOCK_DEPTH(m_cx, jsObj));

                UINT32 count = 0;
                for (JSScopeProperty *p = OBJ_SCOPE(jsObj)->lastProp; p; p = p->parent, ++count);
                Save(count);
                for (JSScopeProperty *p = OBJ_SCOPE(jsObj)->lastProp; p; p = p->parent)
                {
                    Save(p->shortid);
                    Save(JSID_TO_ATOM(p->id));
                }
            }
            else if (&js_FunctionClass == jsClass)
            {
                Save(static_cast<uint8>(OBJ_TYPE_FUNCTION));
                JSFunction *fun = static_cast<JSFunction *>(JS_GetPrivate(m_cx, jsObj));

                JSObject *parent = JSVAL_TO_OBJECT(jsObj->slots[JSSLOT_PARENT]);
                JSObjectId thisId = m_objMap->GetIdByObject(jsObj);
                JSObjectId parentId = 0;
                if (0 != parent)
                {
                    parentId = m_objMap->GetIdByObject(parent);
                }
                Save(thisId);
                Save(parentId);

                Save(fun->nargs);
                Save(fun->flags);

                uint8 funHasName = nullptr != fun->atom ? 1 : 0;
                Save(funHasName);
                if (0 != funHasName)
                {
                    Save(fun->atom);
                }

                Save(fun->u.i.lwars);
                Save(fun->u.i.nregexps);

                UINT32 count = fun->u.i.lwars + fun->nargs;
                if (0 < count)
                {
                    for (JSScopeProperty *p = OBJ_SCOPE(jsObj)->lastProp; p; p = p->parent)
                    {
                        FunctionProperyType propType = FUN_PROP_UNKNOWN;
                        if (js_GetLocalVariable == p->getter)
                        {
                            propType = FUN_PROP_LOCAL_VAR;
                        }
                        else if (js_GetArgument == p->getter)
                        {
                            propType = FUN_PROP_ARG;
                        }
                        Save(static_cast<uint8>(propType));
                        Save(p->shortid);
                        Save(p->flags);
                        Save(JSID_TO_ATOM(p->id));
                    }
                }

                Save(fun->u.i.script);
            }
            else if (&js_RegExpClass == jsClass)
            {
                Save(static_cast<uint8>(OBJ_TYPE_REGEXP));
                JSRegExp *re = static_cast<JSRegExp *>(JS_GetPrivate(m_cx, jsObj));
                Save(re->source);
                Save(re->flags);
                Save(re->cloneIndex);
            }
        }
        else if (ATOM_IS_INT(atom))
        {
#if defined(JS_USE_STRONGLY_TYPED_JSVAL_AND_JSID)
            Save(static_cast<jsval::TagType>(JSVAL_INT));
#else
            Save(static_cast<jsval>(JSVAL_INT));
#endif
            Save(ATOM_TO_INT(atom));
        }
        else if (ATOM_IS_DOUBLE(atom))
        {
#if defined(JS_USE_STRONGLY_TYPED_JSVAL_AND_JSID)
            Save(static_cast<jsval::TagType>(JSVAL_DOUBLE));
#else
            Save(static_cast<jsval>(JSVAL_DOUBLE));
#endif
            Save(*ATOM_TO_DOUBLE(atom));
        }
        else if (ATOM_IS_STRING(atom))
        {
#if defined(JS_USE_STRONGLY_TYPED_JSVAL_AND_JSID)
            Save(static_cast<jsval::TagType>(JSVAL_STRING));
#else
            Save(static_cast<jsval>(JSVAL_STRING));
#endif
            Save(ATOM_TO_STRING(atom));
        }
        else if (ATOM_IS_BOOLEAN(atom))
        {
#if defined(JS_USE_STRONGLY_TYPED_JSVAL_AND_JSID)
            Save(static_cast<jsval::TagType>(JSVAL_BOOLEAN));
#else
            Save(static_cast<jsval>(JSVAL_BOOLEAN));
#endif
            Save(ATOM_TO_BOOLEAN(atom));
        }
    }

    void SaveBytecode(const jsbytecode *bc, uint32 size)
    {
        const jsbytecode *const start = bc;
        const jsbytecode *const finish = start + size;
        for (const jsbytecode *pc = start; finish > pc; ++pc)
        {
            Save(*pc);
        }
    }

    void SaveSrcNotes(const jssrcnote *sn, uint32 size)
    {
        const jssrcnote *const start = sn;
        const jssrcnote *const finish = start + size;
        for (const jssrcnote *it = start; finish > it; ++it)
        {
            Save(*it);
        }
    }

    void Save(const JSTryNote &tryNote)
    {
        Save(tryNote.start);
        Save(tryNote.length);
        Save(tryNote.catchStart);
    }

    void Save(const JSAtomMap &atomMap, JSAtomMap::size_type size)
    {
        Save(atomMap.length);
        for (jsatomid i = 0; size > i; ++i)
        {
            Save(atomMap.vector[i]);
        }
    }

    void Save(const JSString *str)
    {
        JSString::size_type length = JSSTRING_LENGTH(str);
        const JSString::value_type *chars = JSSTRING_CONSTCHARS(str);

        Save(length);
        std::vector<JSString::value_type> buf(length);
        std::copy(&chars[0], &chars[0] + length, &buf[0]);
        for (JSString::size_type i = 0; length > i; ++i)
        {
            Save(buf[i]);
        }
    }

    BitsIO::OutBitStream<OutputIterator> m_os;
    JSContext                           *m_cx;
    JSObjectMap                         *m_objMap;
};

template <class InputIterator, class JSObjectMap>
class JSScriptLoader
{
public:
    JSScriptLoader(const BitsIO::InBitStream<InputIterator> &is,
                   JSContext *cx, JSObjectMap *objMap)
      : m_is(is)
      , m_cx(cx)
      , m_objMap(objMap)
    {}

    void Load(JSScript **jsScript)
    {
        uint32 srcNotesLength;
        uint32 numTryNotes;
        uint32 byteCodeLength;

        Load(&byteCodeLength);
        Load(&srcNotesLength);
        Load(&numTryNotes);

        *jsScript = js_NewScript(m_cx, byteCodeLength, srcNotesLength, numTryNotes);

        if (nullptr == jsScript)
        {
            return;
        }

        LoadBytecode((*jsScript)->code, byteCodeLength);

        ptrdiff_t main;
        Load(&main);
        (*jsScript)->main = (*jsScript)->code + main;
        Load(&(*jsScript)->version);
        Load(&(*jsScript)->numGlobalVars);
        Load(&(*jsScript)->depth);
        Load(&(*jsScript)->lineno);

        jssrcnote *notes = SCRIPT_NOTES(*jsScript);
        LoadSrcNotes(notes, srcNotesLength);

        for (size_t i = 0; numTryNotes > i; ++i)
        {
            Load(&(*jsScript)->trynotes[i]);
        }

        Load(&(*jsScript)->atomMap.length);

        if (0 < (*jsScript)->atomMap.length)
        {
            // It will be destroyed in js_DestroyScript
            (*jsScript)->atomMap.vector = static_cast<JSAtom **>(JS_malloc(
                m_cx,
                static_cast<size_t>((*jsScript)->atomMap.length) * sizeof(*(*jsScript)->atomMap.vector)
            ));
            Load(&(*jsScript)->atomMap, (*jsScript)->atomMap.length);
        }
        else
        {
            (*jsScript)->atomMap.vector = nullptr;
        }
    }

private:
    template <class T>
    void Load(
        T *t,
        typename boost::enable_if_c<
            boost::is_integral<T>::value && boost::is_unsigned<T>::value
          , T
          >::type *dummy = nullptr
    )
    {
        UINT32 zeroBits;
        UINT64 v;
        for (zeroBits = 0; 0 == m_is.Read(1) && 0 < m_is.BitsLeft(); ++zeroBits);
        if (0 == zeroBits)
        {
            v = 0;
        }
        else
        {
            v = (1ULL << zeroBits) + static_cast<UINT64>(m_is.Read(zeroBits)) - 1;
        }
        *t = static_cast<T>(v);
    }

    template <class T>
    void Load(
        T *t,
        typename boost::enable_if_c<
            boost::is_integral<T>::value && !boost::is_unsigned<T>::value
          , T
          >::type* dummy = nullptr
    )
    {
        typename boost::make_unsigned<T>::type u;
        Load(&u);

        if (0 == (u & 0x1))
        {
            u >>= 1;
            *t = 0 - u;
        }
        else
        {
            *t = (u + 1) >> 1;
        }
    }

    void Load(double *v)
    {
        static_assert(sizeof(double) == sizeof(UINT64), "size of double is not 64 bits");
        union { UINT64 ui; double d; } u;
        Load(&u.ui);
        *v = u.d;
    }

    void Load(JSAtom **atom)
    {
#if defined(JS_USE_STRONGLY_TYPED_JSVAL_AND_JSID)
        jsval::TagType tag;
#else
        jsval tag;
#endif
        Load(&tag);
        switch (tag)
        {
            case JSVAL_OBJECT:
            {
                uint8 objType;
                Load(&objType);

                switch (objType)
                {
                    case OBJ_TYPE_BLOCK:
                    {
                        JSObjectId parentId = 0;
                        JSObjectId thisId = 0;

                        Load(&thisId);
                        Load(&parentId);

                        JSObject *jsObj = js_NewBlockObject(m_cx);
                        *atom = js_AtomizeObject(m_cx, jsObj, 0);
                        (*m_objMap)[thisId] = jsObj;
                        if (0 != parentId)
                        {
                            JSObject *parent = m_objMap->GetObjectById(parentId);
                            if (nullptr != parent)
                            {
                                OBJ_SET_SLOT(m_cx, jsObj, JSSLOT_PARENT, OBJECT_TO_JSVAL(parent));
                            }
                        }
                        else
                        {
                            OBJ_SET_SLOT(m_cx, jsObj, JSSLOT_PARENT, OBJECT_TO_JSVAL(0));
                        }

                        jsint depth;
                        Load(&depth);
                        OBJ_SET_BLOCK_DEPTH(m_cx, jsObj, depth);

                        UINT32 count;
                        Load(&count);
                        for (UINT32 i = 0; count > i; ++i)
                        {
                            int16 shortid;
                            JSAtom *name;
                            Load(&shortid);
                            Load(&name);
                            js_DefineNativeProperty(m_cx, jsObj, ATOM_TO_JSID(name),
                                JSVAL_VOID, nullptr, nullptr,
                                JSPROP_ENUMERATE | JSPROP_PERMANENT,
                                SPROP_HAS_SHORTID,
                                shortid,
                                nullptr);
                        }

                        break;
                    }
                    case OBJ_TYPE_FUNCTION:
                    {
                        JSObjectId parentId = 0;
                        JSObjectId thisId = 0;

                        Load(&thisId);
                        Load(&parentId);

                        uint16 nargs;
                        uint16 flags;
                        JSAtom *funAtom = nullptr;
                        Load(&nargs);
                        Load(&flags);

                        uint8 funHasName = 0;
                        Load(&funHasName);
                        if (0 != funHasName)
                        {
                            Load(&funAtom);
                        }

                        JSObject *parent = nullptr;
                        if (0 != parentId)
                        {
                            parent = m_objMap->GetObjectById(parentId);
                        }

                        JSFunction *fun = js_NewFunction(
                            m_cx,
                            nullptr,
                            nullptr,
                            nargs,
                            flags,
                            parent,
                            funAtom
                        );
                        JSObject *funObj = fun->object;
                        *atom = js_AtomizeObject(m_cx, funObj, 0);
                        (*m_objMap)[thisId] = funObj;
                        funObj->slots[JSSLOT_PARENT] = OBJECT_TO_JSVAL(parent);

                        // reassign flags, since js_NewFunction resets
                        // interpreted flag
                        fun->flags = flags;

                        Load(&fun->u.i.lwars);
                        Load(&fun->u.i.nregexps);

                        UINT32 count = fun->u.i.lwars + fun->nargs;
                        for (UINT32 i = 0; count > i; ++i)
                        {
                            int16 shortid;
                            JSAtom *name = nullptr;
                            uint8 flags;
                            uint8 propType;
                            JSPropertyOp getter = nullptr;
                            JSPropertyOp setter = nullptr;

                            Load(&propType);
                            if (FUN_PROP_LOCAL_VAR == propType)
                            {
                                getter = js_GetLocalVariable;
                                setter = js_SetLocalVariable;
                            }
                            else if (FUN_PROP_ARG == propType)
                            {
                                getter = js_GetArgument;
                                setter = js_SetArgument;
                            }

                            Load(&shortid);
                            Load(&flags);
                            Load(&name);
                            js_AddHiddenProperty(m_cx, funObj, ATOM_TO_JSID(name),
                                getter, setter,
                                SPROP_ILWALID_SLOT,
                                JSPROP_PERMANENT | JSPROP_SHARED,
                                flags | SPROP_HAS_SHORTID,
                                shortid);
                        }
                        Load(&fun->u.i.script);

                        break;
                    }
                    case OBJ_TYPE_REGEXP:
                    {
                        JSString *source;
                        uint16 flags;

                        Load(&source);
                        JSString::size_type length = JSSTRING_LENGTH(source);
                        JSString::value_type *chars = JSSTRING_CHARS(source);
                        Load(&flags);

                        JSObject *jsObj = js_NewRegExpObject(m_cx, nullptr, chars, length, flags);
                        *atom = js_AtomizeObject(m_cx, jsObj, 0);
                        JSRegExp * re = static_cast<JSRegExp *>(JS_GetPrivate(m_cx, jsObj));
                        Load(&re->cloneIndex);

                        break;
                    }
                    default:
                        break;
                }

                break;
            }
            case JSVAL_INT:
            {
                jsint val;
                Load(&val);
                *atom = js_AtomizeInt(m_cx, val, 0);

                break;
            }
            case JSVAL_DOUBLE:
            {
                jsdouble val;
                Load(&val);
                *atom = js_AtomizeDouble(m_cx, val, 0);

                break;
            }
            case JSVAL_STRING:
            {
                JSString::size_type length;
                Load(&length);
                std::vector<JSString::value_type> buf(length);
                for (JSString::size_type i = 0; length > i; ++i)
                {
                    Load(&buf[i]);
                }
                *atom = js_AtomizeChars(m_cx, &buf[0], length, 0);

                break;
            }
            case JSVAL_BOOLEAN:
            {
                JSBool val;
                Load(&val);
                *atom = js_AtomizeBoolean(m_cx, val, 0);

                break;
            }
            default:
                break;
        }
    }

    void Load(JSString **jsString)
    {
        JSString::size_type length;
        Load(&length);
        std::vector<JSString::value_type> buf(length);
        for (JSString::size_type i = 0; length > i; ++i)
        {
            Load(&buf[i]);
        }
        JSAtom *atom = js_AtomizeChars(m_cx, &buf[0], length, 0);
        *jsString = ATOM_TO_STRING(atom);
    }

    void LoadBytecode(jsbytecode *bc, uint32 size)
    {
        jsbytecode *const start = bc;
        jsbytecode *const finish = start + size;
        for (jsbytecode *pc = start; finish > pc; ++pc)
        {
            Load(pc);
        }
    }

    void LoadSrcNotes(jssrcnote *sn, uint32 size)
    {
        jssrcnote *const start = sn;
        jssrcnote *const finish = start + size;
        for (jssrcnote *it = start; finish > it; ++it)
        {
            Load(it);
        }
    }

    void Load(JSTryNote *tryNote)
    {
        Load(&tryNote->start);
        Load(&tryNote->length);
        Load(&tryNote->catchStart);
    }

    void Load(JSAtomMap *atomMap, JSAtomMap::size_type size)
    {
        for (jsatomid i = 0; size > i; ++i)
        {
            Load(&atomMap->vector[i]);
        }
    }

    BitsIO::InBitStream<InputIterator> m_is;
    JSContext                         *m_cx;
    JSObjectMap                       *m_objMap;
};

#endif
