/* -*- Mode: C; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 4 -*-
 *
 * ***** BEGIN LICENSE BLOCK *****
 * Version: MPL 1.1/GPL 2.0/LGPL 2.1
 *
 * The contents of this file are subject to the Mozilla Public License Version
 * 1.1 (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 * http://www.mozilla.org/MPL/
 *
 * Software distributed under the License is distributed on an "AS IS" basis,
 * WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
 * for the specific language governing rights and limitations under the
 * License.
 *
 * The Original Code is Mozilla Communicator client code, released
 * March 31, 1998.
 *
 * The Initial Developer of the Original Code is
 * Netscape Communications Corporation.
 * Portions created by the Initial Developer are Copyright (C) 1998
 * the Initial Developer. All Rights Reserved.
 *
 * Contributor(s):
 *
 * Alternatively, the contents of this file may be used under the terms of
 * either of the GNU General Public License Version 2 or later (the "GPL"),
 * or the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
 * in which case the provisions of the GPL or the LGPL are applicable instead
 * of those above. If you wish to allow use of your version of this file only
 * under the terms of either the GPL or the LGPL, and not to allow others to
 * use your version of this file under the terms of the MPL, indicate your
 * decision by deleting the provisions above and replace them with the notice
 * and other provisions required by the GPL or the LGPL. If you do not delete
 * the provisions above, a recipient may use your version of this file under
 * the terms of any one of the MPL, the GPL or the LGPL.
 *
 * ***** END LICENSE BLOCK ***** */

#ifndef jspubtd_h___
#define jspubtd_h___
/*
 * JS public API typedefs.
 */
#include "jstypes.h"
#include "jscompat.h"

/* Scalar typedefs. */
#if defined(__cplusplus) && (__cplusplus > 199711L || \
                             (defined(_MSC_VER) && _MSC_VER >= 1900))
                             // Before _MSC_VER 1900 char16_t in MSVC was
                             // a typedef.
typedef char16_t  jschar;
#else
typedef uint16    jschar;
#endif

typedef int32     jsint;
typedef uint32    jsuint;
typedef float64   jsdouble;
typedef int32     jsrefcount;   /* PRInt32 if JS_THREADSAFE, see jslock.h */

#if defined(JS_USE_STRONGLY_TYPED_JSVAL_AND_JSID)
#include <type_traits>
#ifdef _WIN32
#   pragma warning(push)
#   pragma warning(disable: 4265) // class has virtual functions, but destructor is not virtual
#endif
#include <functional>
#ifdef _WIN32
#   pragma warning(pop)
#endif

#if defined(_MSC_VER)
// C-linkage specified, but returns UDT 'jsval' which is incompatible with C
#pragma warning(disable:4190)
#endif

template <typename Integral, typename Tag>
class StrongInt
{
    template <typename AnotherIntegral, typename AnotherTag>
    friend class StrongInt;

private:
    Integral m_Int;

public:
    typedef Integral value_type;

    Integral GetValue() const { return m_Int; }
    void SetValue(Integral v) { m_Int = v; }

    StrongInt() = default;

    template <
        typename T
      , typename Foo = std::enable_if_t<
            // is_colwertible would be more appropriate, but MSVC issues
            // an internal error in jsdbgapi.cpp
            (std::is_integral<T>::value && sizeof(T) <= sizeof(Integral)) ||
            std::is_enum<T>::value
          >
    >
    constexpr StrongInt(T t)
      : m_Int(static_cast<Integral>(t))
    {}

    template <typename AnotherTag>
    StrongInt(StrongInt<Integral, AnotherTag> si)
      : m_Int(si.m_Int)
    {}

    template <
        typename T
      , typename Foo = std::enable_if_t<sizeof(Integral) >= sizeof(T*)>
      >
    StrongInt(T *p)
      : m_Int(reinterpret_cast<Integral>(p))
    {}

    operator Integral () const { return m_Int; }

    template <
        typename T
      , typename Foo = std::enable_if_t<sizeof(Integral) >= sizeof(T*)>
      >
    operator T* () const { return reinterpret_cast<T*>(m_Int); }

    template <
        typename T
      , typename AnotherTag
      , typename Foo = std::enable_if_t<sizeof(Integral) >= sizeof(T)>
      >
    StrongInt& operator = (const StrongInt<T, AnotherTag> si)
    {
        m_Int = si.m_Int;

        return *this;
    }

#define JS_STRONG_INT_DEF_COMPARE_OP(op)                                         \
    template <                                                                   \
        typename T                                                               \
      , typename Foo = std::enable_if_t<std::is_integral<T>::value>              \
      >                                                                          \
    constexpr bool operator op (T t) const { return m_Int op t; }                \
                                                                                 \
    template <                                                                   \
        typename T                                                               \
      , typename Foo = std::enable_if_t<std::is_integral<T>::value>              \
      >                                                                          \
    friend constexpr bool operator op (T t, const StrongInt si)                  \
    {                                                                            \
        return t op Integral(si);                                                \
    }                                                                            \
                                                                                 \
    template <typename AnotherIntegral, typename AnotherTag>                     \
    constexpr bool operator op (StrongInt<AnotherIntegral, AnotherTag> si) const \
    {                                                                            \
        return m_Int op si.m_Int;                                                \
    }

    JS_STRONG_INT_DEF_COMPARE_OP(==);
    JS_STRONG_INT_DEF_COMPARE_OP(!=);
    JS_STRONG_INT_DEF_COMPARE_OP(<);
    JS_STRONG_INT_DEF_COMPARE_OP(<=);
    JS_STRONG_INT_DEF_COMPARE_OP(>);
    JS_STRONG_INT_DEF_COMPARE_OP(>=);
#undef JS_STRONG_INT_DEF_COMPARE_OP

#define JS_STRONG_INT_DEF_OP(op)                                               \
    template <                                                                 \
        typename T                                                             \
      , typename Foo = std::enable_if_t<std::is_integral<T>::value>            \
      >                                                                        \
    auto operator op (T t) const -> StrongInt<decltype(this->m_Int op t), Tag> \
    {                                                                          \
        return m_Int op t;                                                     \
    }                                                                          \
                                                                               \
    template <                                                                 \
        typename T                                                             \
      , typename Foo = std::enable_if_t<std::is_integral<T>::value>            \
      >                                                                        \
    friend auto operator op (T t, const StrongInt si) -> decltype(si op t)     \
    {                                                                          \
        return t op Integral(si);                                              \
    }                                                                          \
                                                                               \
    template <typename AnotherIntegral>                                        \
    auto operator op (StrongInt<AnotherIntegral, Tag> si) const                \
        -> StrongInt<decltype(this->m_Int op si.m_Int), Tag>                   \
    {                                                                          \
        return m_Int op si.m_Int;                                              \
    }

    JS_STRONG_INT_DEF_OP(^);
    JS_STRONG_INT_DEF_OP(|);
    JS_STRONG_INT_DEF_OP(&);
    JS_STRONG_INT_DEF_OP(<<);
#undef JS_STRONG_INT_DEF_OP

#define JS_STRONG_INT_DEF_ASSIGNMENT_OP(op)                               \
    template <                                                            \
        typename T                                                        \
      , typename Foo = std::enable_if_t<std::is_integral<T>::value>       \
      >                                                                   \
    StrongInt& operator op (T t)                                          \
    {                                                                     \
        m_Int op t;                                                       \
                                                                          \
        return *this;                                                     \
    }

    JS_STRONG_INT_DEF_ASSIGNMENT_OP(=);
    JS_STRONG_INT_DEF_ASSIGNMENT_OP(+=);
    JS_STRONG_INT_DEF_ASSIGNMENT_OP(-=);
    JS_STRONG_INT_DEF_ASSIGNMENT_OP(|=);
    JS_STRONG_INT_DEF_ASSIGNMENT_OP(&=);
    JS_STRONG_INT_DEF_ASSIGNMENT_OP(<<=);
#undef JS_STRONG_INT_DEF_ASSIGNMENT_OP
};

struct JsValTag {};
struct JsIdTag {};

struct JSObject;
struct JSString;
struct JSContext;
struct JSAtom;
class jsval : public StrongInt<jsword, JsValTag>
{
public:
    typedef StrongInt<jsword, JsValTag> Base;
    typedef Base::value_type value_type;
    typedef JSUint8 TagType;

    using Base::Base;
    using Base::operator =;

    // jsval can store values of different types: objects, integers, doubles,
    // strings and Booleans. In order to distinguish them jsval stores a tag in
    // the least significant bits of its binary representation. The tag has a
    // variable length: one bit for integers and three for all other types. What
    // makes it possible is the fact that three types (objects, strings and
    // doubles) are represented by pointers and pointers are aligned. Booleans
    // don't require too many bits. Finally, since for integers only one bit is
    // spared for a tag, the size of integers is simply limited by 31 bits
    // instead of 32.
    enum Tags : TagType
    {
        Object  = 0,
        Integer = 1,
        Double  = 2,
        String  = 4,
        Boolean = 6
    };

    static constexpr int tagBits = 3;
    static constexpr int intBits = 31;

    static constexpr value_type Null{ 0 };
    static constexpr value_type Void{ -value_type(0x7fffffff) };

private:
    static constexpr value_type tagMask = (1 << tagBits) - 1;

    void SetTag(TagType tag)
    {
        *this |= tag;
    }

    void ClearTag()
    {
        *this &= ~tagMask;
    }

    void Align()
    {
        *this <<= tagBits;
    }

public:
    static constexpr value_type intMin = 1 - (1 << 30);
    static constexpr value_type intMax = (1 << 30) - 1;
    static constexpr value_type align = value_type(1) << tagBits;

    TagType GetTag() const
    {
        return static_cast<TagType>(*this & tagMask);
    }

    template <
        typename T
      , typename Foo = std::enable_if_t<
            (std::is_integral<T>::value && sizeof(T) <= sizeof(value_type)) ||
            std::is_enum<T>::value
          >
      >
    static constexpr jsval FromInt(T t)
    {
        return (value_type(t) << 1) | Tags::Integer;
    }

    static jsval FromBoolean(JSBool b)
    {
        jsval v = b << tagBits;
        v.SetTag(Tags::Boolean);
        return v;
    }

    static jsval FromObject(JSObject *obj)
    {
        return obj;
    }

    static jsval FromDouble(jsdouble *dp)
    {
        jsval v = dp;
        v.SetTag(Tags::Double);
        return v;
    }

    static jsval FromString(JSString *str)
    {
        jsval v = str;
        v.SetTag(Tags::String);
        return v;
    }

    static jsval FromPrivate(void *p)
    {
        jsval v = p;
        v.SetTag(Tags::Integer);
        return v;
    }

    template <
        typename T
      , typename Foo = std::enable_if_t<
            (std::is_integral<T>::value && sizeof(T) <= sizeof(value_type)) ||
            std::is_enum<T>::value
          >
      >
    static constexpr bool IntFits(T t)
    {
        return t + intMax <= 2 * intMax;
    }

    bool IsObject() const { return GetTag() == Tags::Object; }
    bool IsNumber() const { return IsDouble() || IsInt(); }
    // integer is the only type that has the least significant bit set, don't
    // need to extract the full tag
    bool IsInt() const { return 0 != (*this & value_type(Tags::Integer)) && !IsVoid(); }
    bool IsDouble() const { return GetTag() == Tags::Double; }
    bool IsString() const { return GetTag() == Tags::String; }
    bool IsBoolean() const { return GetTag() == Tags::Boolean; }
    bool IsNull() const { return *this == Null; }
    bool IsVoid() const { return *this == Void; }
    bool IsPrimitive() const { return !IsObject() || IsNull(); }
    bool IsGcThing() const { return 0 == (*this & value_type(Tags::Integer)) && !IsBoolean(); }
    
    void* ToGcThing() const
    {
        return *this & ~tagMask;
    }
    
    JSObject * ToObject() const
    {
        return *this & ~tagMask;
    }

    jsdouble * ToDouble() const
    {
        return *this & ~tagMask;;
    }

    JSString * ToString() const
    {
        return *this & ~tagMask;
    }

    jsint ToInt() const
    {
        return static_cast<jsint>(*this >> 1);
    }

    JSBool ToBoolean() const
    {
        return static_cast<JSBool>(*this >> tagBits);
    }

    // A private data pointer (2-byte-aligned) can be stored as an int jsval
    void * ToPrivate() const
    {
        return *this & ~value_type(1);
    }

    JSBool Lock(JSContext *cx);
    JSBool UnLock(JSContext *cx);
};

class jsid : public StrongInt<jsword, JsIdTag>
{
public:
    typedef StrongInt<jsword, JsIdTag> Base;
    typedef Base::value_type value_type;
    typedef JSUint8 TagType;

    using Base::Base;

    enum Tags : TagType
    {
        Atom    = 0,
        Integer = 1,
        Object  = 2
    };

private:
    void SetTag(TagType tag)
    {
        *this |= tag;
    }

    void ClearTag()
    {
        *this &= ~tagMask;
    }

public:
    static constexpr value_type tagMask = 3;

    TagType GetTag() const
    {
        return static_cast<TagType>(*this & tagMask);
    }

    static jsid FromAtom(const JSAtom *a)
    {
        return a;
    }

    static jsid FromObject(JSObject *obj)
    {
        jsid id = obj;
        id.SetTag(Tags::Object);
        return id;
    }

    template <
        typename T
      , typename Foo = std::enable_if_t<
            (std::is_integral<T>::value && sizeof(T) <= sizeof(value_type)) ||
            std::is_enum<T>::value
          >
      >
    static constexpr jsid FromInt(T t)
    {
        return (value_type(t) << 1) | Tags::Integer;
    }

    bool IsAtom() const { return GetTag() == Tags::Atom; }
    bool IsInt() const { return 0 != (*this & value_type(Tags::Integer)); }
    bool IsObject() const { return GetTag() == Tags::Object; }

    JSAtom * ToAtom() const
    {
        return *this & ~tagMask;
    }

    jsint ToInt() const
    {
        return static_cast<jsint>(*this >> 1);
    }

    JSObject * ToObject() const
    {
        return *this & ~tagMask;
    }
};

namespace std
{
    // inject a hash function into std namespace to allow storing jsval in
    // unordered_map
    template <>
    struct hash<jsval>
    {
        typedef jsval argument_type;
        typedef std::size_t result_type;
        
        result_type operator()(argument_type const& s) const
        {
            // reuse an existing hash for integral types
            return hash<jsval::value_type>()(jsval::value_type(s));
        }
    };

    template <typename Integral, typename Tag>
    inline
    constexpr const Integral min(Integral x, StrongInt<Integral, Tag> y)
    {
        return x < y ? x : Integral(y);
    }

    template <typename Integral, typename Tag>
    inline
    constexpr const Integral min(StrongInt<Integral, Tag> x, Integral y)
    {
        return x < y ? Integral(x) : y;
    }
}
#else
typedef jsword    jsval;
typedef jsword    jsid;
#endif

JS_BEGIN_EXTERN_C

/*
 * Run-time version enumeration.  See jsconfig.h for compile-time counterparts
 * to these values that may be selected by the JS_VERSION macro, and tested by
 * #if expressions.
 */
typedef enum JSVersion {
    JSVERSION_1_0     = 100,
    JSVERSION_1_1     = 110,
    JSVERSION_1_2     = 120,
    JSVERSION_1_3     = 130,
    JSVERSION_1_4     = 140,
    JSVERSION_ECMA_3  = 148,
    JSVERSION_1_5     = 150,
    JSVERSION_1_6     = 160,
    JSVERSION_1_7     = 170,
    JSVERSION_DEFAULT = 0,
    JSVERSION_UNKNOWN = -1
} JSVersion;

#define JSVERSION_IS_ECMA(version) \
    ((version) == JSVERSION_DEFAULT || (version) >= JSVERSION_1_3)

/* Result of typeof operator enumeration. */
typedef enum JSType {
    JSTYPE_VOID,                /* undefined */
    JSTYPE_OBJECT,              /* object */
    JSTYPE_FUNCTION,            /* function */
    JSTYPE_STRING,              /* string */
    JSTYPE_NUMBER,              /* number */
    JSTYPE_BOOLEAN,             /* boolean */
    JSTYPE_NULL,                /* null */
    JSTYPE_XML,                 /* xml object */
    JSTYPE_LIMIT
} JSType;

/* Dense index into cached prototypes and class atoms for standard objects. */
typedef enum JSProtoKey {
#define JS_PROTO(name,idname,code,init) JSProto_##name = code,
#include "jsproto.tbl"
#undef JS_PROTO
    JSProto_LIMIT
} JSProtoKey;

/* JSObjectOps.checkAccess mode enumeration. */
typedef enum JSAccessMode {
    JSACC_PROTO  = 0,           /* XXXbe redundant w.r.t. id */
    JSACC_PARENT = 1,           /* XXXbe redundant w.r.t. id */
    JSACC_IMPORT = 2,           /* import foo.bar */
    JSACC_WATCH  = 3,           /* a watchpoint on object foo for id 'bar' */
    JSACC_READ   = 4,           /* a "get" of foo.bar */
    JSACC_WRITE  = 8,           /* a "set" of foo.bar = baz */
    JSACC_LIMIT
} JSAccessMode;

#define JSACC_TYPEMASK          (JSACC_WRITE - 1)

/*
 * This enum type is used to control the behavior of a JSObject property
 * iterator function that has type JSNewEnumerate.
 */
typedef enum JSIterateOp {
    JSENUMERATE_INIT,       /* Create new iterator state */
    JSENUMERATE_NEXT,       /* Iterate once */
    JSENUMERATE_DESTROY     /* Destroy iterator state */
} JSIterateOp;

/* Struct typedefs. */
typedef struct JSClass           JSClass;
typedef struct JSExtendedClass   JSExtendedClass;
typedef struct JSConstDoubleSpec JSConstDoubleSpec;
typedef struct JSContext         JSContext;
typedef struct JSErrorReport     JSErrorReport;
typedef struct JSFunction        JSFunction;
typedef struct JSFunctionSpec    JSFunctionSpec;
typedef struct JSIdArray         JSIdArray;
typedef struct JSProperty        JSProperty;
typedef struct JSPropertySpec    JSPropertySpec;
typedef struct JSObject          JSObject;
typedef struct JSObjectMap       JSObjectMap;
typedef struct JSObjectOps       JSObjectOps;
typedef struct JSXMLObjectOps    JSXMLObjectOps;
typedef struct JSRuntime         JSRuntime;
typedef struct JSRuntime         JSTaskState;   /* XXX deprecated name */
typedef struct JSScript          JSScript;
typedef struct JSStackFrame      JSStackFrame;
typedef struct JSString          JSString;
typedef struct JSXDRState        JSXDRState;
typedef struct JSExceptionState  JSExceptionState;
typedef struct JSLocaleCallbacks JSLocaleCallbacks;

/* JSClass (and JSObjectOps where appropriate) function pointer typedefs. */

/*
 * Add, delete, get or set a property named by id in obj.  Note the jsval id
 * type -- id may be a string (Unicode property identifier) or an int (element
 * index).  The *vp out parameter, on success, is the new property value after
 * an add, get, or set.  After a successful delete, *vp is JSVAL_FALSE iff
 * obj[id] can't be deleted (because it's permanent).
 */
typedef JSBool
(* JS_DLL_CALLBACK JSPropertyOp)(JSContext *cx, JSObject *obj, jsval id,
                                 jsval *vp);

/*
 * This function type is used for callbacks that enumerate the properties of
 * a JSObject.  The behavior depends on the value of enum_op:
 *
 *  JSENUMERATE_INIT
 *    A new, opaque iterator state should be allocated and stored in *statep.
 *    (You can use PRIVATE_TO_JSVAL() to tag the pointer to be stored).
 *
 *    The number of properties that will be enumerated should be returned as
 *    an integer jsval in *idp, if idp is non-null, and provided the number of
 *    enumerable properties is known.  If idp is non-null and the number of
 *    enumerable properties can't be computed in advance, *idp should be set
 *    to JSVAL_ZERO.
 *
 *  JSENUMERATE_NEXT
 *    A previously allocated opaque iterator state is passed in via statep.
 *    Return the next jsid in the iteration using *idp.  The opaque iterator
 *    state pointed at by statep is destroyed and *statep is set to JSVAL_NULL
 *    if there are no properties left to enumerate.
 *
 *  JSENUMERATE_DESTROY
 *    Destroy the opaque iterator state previously allocated in *statep by a
 *    call to this function when enum_op was JSENUMERATE_INIT.
 *
 * The return value is used to indicate success, with a value of JS_FALSE
 * indicating failure.
 */
typedef JSBool
(* JS_DLL_CALLBACK JSNewEnumerateOp)(JSContext *cx, JSObject *obj,
                                     JSIterateOp enum_op,
                                     jsval *statep, jsid *idp);

/*
 * The old-style JSClass.enumerate op should define all lazy properties not
 * yet reflected in obj.
 */
typedef JSBool
(* JS_DLL_CALLBACK JSEnumerateOp)(JSContext *cx, JSObject *obj);

/*
 * Resolve a lazy property named by id in obj by defining it directly in obj.
 * Lazy properties are those reflected from some peer native property space
 * (e.g., the DOM attributes for a given node reflected as obj) on demand.
 *
 * JS looks for a property in an object, and if not found, tries to resolve
 * the given id.  If resolve succeeds, the engine looks again in case resolve
 * defined obj[id].  If no such property exists directly in obj, the process
 * is repeated with obj's prototype, etc.
 *
 * NB: JSNewResolveOp provides a cheaper way to resolve lazy properties.
 */
typedef JSBool
(* JS_DLL_CALLBACK JSResolveOp)(JSContext *cx, JSObject *obj, jsval id);

/*
 * Like JSResolveOp, but flags provide contextual information as follows:
 *
 *  JSRESOLVE_QUALIFIED   a qualified property id: obj.id or obj[id], not id
 *  JSRESOLVE_ASSIGNING   obj[id] is on the left-hand side of an assignment
 *  JSRESOLVE_DETECTING   'if (o.p)...' or similar detection opcode sequence
 *  JSRESOLVE_DECLARING   var, const, or function prolog declaration opcode
 *  JSRESOLVE_CLASSNAME   class name used when constructing
 *
 * The *objp out parameter, on success, should be null to indicate that id
 * was not resolved; and non-null, referring to obj or one of its prototypes,
 * if id was resolved.
 *
 * This hook instead of JSResolveOp is called via the JSClass.resolve member
 * if JSCLASS_NEW_RESOLVE is set in JSClass.flags.
 *
 * Setting JSCLASS_NEW_RESOLVE and JSCLASS_NEW_RESOLVE_GETS_START further
 * extends this hook by passing in the starting object on the prototype chain
 * via *objp.  Thus a resolve hook implementation may define the property id
 * being resolved in the object in which the id was first sought, rather than
 * in a prototype object whose class led to the resolve hook being called.
 *
 * When using JSCLASS_NEW_RESOLVE_GETS_START, the resolve hook must therefore
 * null *objp to signify "not resolved".  With only JSCLASS_NEW_RESOLVE and no
 * JSCLASS_NEW_RESOLVE_GETS_START, the hook can assume *objp is null on entry.
 * This is not good practice, but enough existing hook implementations count
 * on it that we can't break compatibility by passing the starting object in
 * *objp without a new JSClass flag.
 */
typedef JSBool
(* JS_DLL_CALLBACK JSNewResolveOp)(JSContext *cx, JSObject *obj, jsval id,
                                   uintN flags, JSObject **objp);

/*
 * Colwert obj to the given type, returning true with the resulting value in
 * *vp on success, and returning false on error or exception.
 */
typedef JSBool
(* JS_DLL_CALLBACK JSColwertOp)(JSContext *cx, JSObject *obj, JSType type,
                                jsval *vp);

/*
 * Finalize obj, which the garbage collector has determined to be unreachable
 * from other live objects or from GC roots.  Obviously, finalizers must never
 * store a reference to obj.
 */
typedef void
(* JS_DLL_CALLBACK JSFinalizeOp)(JSContext *cx, JSObject *obj);

/*
 * Used by JS_AddExternalStringFinalizer and JS_RemoveExternalStringFinalizer
 * to extend and reduce the set of string types finalized by the GC.
 */
typedef void
(* JS_DLL_CALLBACK JSStringFinalizeOp)(JSContext *cx, JSString *str);

/*
 * The signature for JSClass.getObjectOps, used by JS_NewObject's internals
 * to discover the set of high-level object operations to use for new objects
 * of the given class.  All native objects have a JSClass, which is stored as
 * a private (int-tagged) pointer in obj->slots[JSSLOT_CLASS].  In contrast,
 * all native and host objects have a JSObjectMap at obj->map, which may be
 * shared among a number of objects, and which contains the JSObjectOps *ops
 * pointer used to dispatch object operations from API calls.
 *
 * Thus JSClass (which pre-dates JSObjectOps in the API) provides a low-level
 * interface to class-specific code and data, while JSObjectOps allows for a
 * higher level of operation, which does not use the object's class except to
 * find the class's JSObjectOps struct, by calling clasp->getObjectOps, and to
 * finalize the object.
 *
 * If this seems backwards, that's because it is!  API compatibility requires
 * a JSClass *clasp parameter to JS_NewObject, etc.  Most host objects do not
 * need to implement the larger JSObjectOps, and can share the common JSScope
 * code and data used by the native (js_ObjectOps, see jsobj.c) ops.
 *
 * Further extension to preserve API compatibility: if this function returns
 * a pointer to JSXMLObjectOps.base, not to JSObjectOps, then the engine calls
 * extended hooks needed for E4X.
 */
typedef JSObjectOps *
(* JS_DLL_CALLBACK JSGetObjectOps)(JSContext *cx, JSClass *clasp);

/*
 * JSClass.checkAccess type: check whether obj[id] may be accessed per mode,
 * returning false on error/exception, true on success with obj[id]'s last-got
 * value in *vp, and its attributes in *attrsp.  As for JSPropertyOp above, id
 * is either a string or an int jsval.
 *
 * See JSCheckAccessIdOp, below, for the JSObjectOps counterpart, which takes
 * a jsid (a tagged int or aligned, unique identifier pointer) rather than a
 * jsval.  The native js_ObjectOps.checkAccess simply forwards to the object's
 * clasp->checkAccess, so that both JSClass and JSObjectOps implementors may
 * specialize access checks.
 */
typedef JSBool
(* JS_DLL_CALLBACK JSCheckAccessOp)(JSContext *cx, JSObject *obj, jsval id,
                                    JSAccessMode mode, jsval *vp);

/*
 * Encode or decode an object, given an XDR state record representing external
 * data.  See jsxdrapi.h.
 */
typedef JSBool
(* JS_DLL_CALLBACK JSXDRObjectOp)(JSXDRState *xdr, JSObject **objp);

/*
 * Check whether v is an instance of obj.  Return false on error or exception,
 * true on success with JS_TRUE in *bp if v is an instance of obj, JS_FALSE in
 * *bp otherwise.
 */
typedef JSBool
(* JS_DLL_CALLBACK JSHasInstanceOp)(JSContext *cx, JSObject *obj, jsval v,
                                    JSBool *bp);

/*
 * Function type for JSClass.mark and JSObjectOps.mark, called from the GC to
 * scan live GC-things reachable from obj's private data structure.  For each
 * such thing, a mark implementation must call
 *
 *    JS_MarkGCThing(cx, thing, name, arg);
 *
 * The trailing name and arg parameters are used for GC_MARK_DEBUG-mode heap
 * dumping and ref-path tracing.  The mark function should pass a (typically
 * literal) string naming the private data member for name, and it must pass
 * the opaque arg parameter through from its caller.
 *
 * For the JSObjectOps.mark hook, the return value is the number of slots at
 * obj->slots to scan.  For JSClass.mark, the return value is ignored.
 *
 * NB: JSMarkOp implementations cannot allocate new GC-things (JS_NewObject
 * called from a mark function will fail silently, e.g.).
 */
typedef uint32
(* JS_DLL_CALLBACK JSMarkOp)(JSContext *cx, JSObject *obj, void *arg);

/*
 * The optional JSClass.reserveSlots hook allows a class to make computed
 * per-instance object slots reservations, in addition to or instead of using
 * JSCLASS_HAS_RESERVED_SLOTS(n) in the JSClass.flags initializer to reserve
 * a constant-per-class number of slots.  Implementations of this hook should
 * return the number of slots to reserve, not including any reserved by using
 * JSCLASS_HAS_RESERVED_SLOTS(n) in JSClass.flags.
 *
 * NB: called with obj locked by the JSObjectOps-specific mutual exclusion
 * mechanism appropriate for obj, so don't nest other operations that might
 * also lock obj.
 */
typedef uint32
(* JS_DLL_CALLBACK JSReserveSlotsOp)(JSContext *cx, JSObject *obj);

/* JSObjectOps function pointer typedefs. */

/*
 * Create a new subclass of JSObjectMap (see jsobj.h), with the nrefs and ops
 * members initialized from the same-named parameters, and with the nslots and
 * freeslot members initialized according to ops and clasp.  Return null on
 * error, non-null on success.
 *
 * JSObjectMaps are reference-counted by generic code in the engine.  Usually,
 * the nrefs parameter to JSObjectOps.newObjectMap will be 1, to count the ref
 * returned to the caller on success.  After a successful construction, some
 * number of js_HoldObjectMap and js_DropObjectMap calls ensue.  When nrefs
 * reaches 0 due to a js_DropObjectMap call, JSObjectOps.destroyObjectMap will
 * be called to dispose of the map.
 */
typedef JSObjectMap *
(* JS_DLL_CALLBACK JSNewObjectMapOp)(JSContext *cx, jsrefcount nrefs,
                                     JSObjectOps *ops, JSClass *clasp,
                                     JSObject *obj);

/*
 * Generic type for an infallible JSObjectMap operation, used lwrrently by
 * JSObjectOps.destroyObjectMap.
 */
typedef void
(* JS_DLL_CALLBACK JSObjectMapOp)(JSContext *cx, JSObjectMap *map);

/*
 * Look for id in obj and its prototype chain, returning false on error or
 * exception, true on success.  On success, return null in *propp if id was
 * not found.  If id was found, return the first object searching from obj
 * along its prototype chain in which id names a direct property in *objp, and
 * return a non-null, opaque property pointer in *propp.
 *
 * If JSLookupPropOp succeeds and returns with *propp non-null, that pointer
 * may be passed as the prop parameter to a JSAttributesOp, as a short-cut
 * that bypasses id re-lookup.  In any case, a non-null *propp result after a
 * successful lookup must be dropped via JSObjectOps.dropProperty.
 *
 * NB: successful return with non-null *propp means the implementation may
 * have locked *objp and added a reference count associated with *propp, so
 * callers should not risk deadlock by nesting or interleaving other lookups
 * or any obj-bearing ops before dropping *propp.
 */
typedef JSBool
(* JS_DLL_CALLBACK JSLookupPropOp)(JSContext *cx, JSObject *obj, jsid id,
                                   JSObject **objp, JSProperty **propp);

/*
 * Define obj[id], a direct property of obj named id, having the given initial
 * value, with the specified getter, setter, and attributes.  If the propp out
 * param is non-null, *propp on successful return contains an opaque property
 * pointer usable as a speedup hint with JSAttributesOp.  But note that propp
 * may be null, indicating that the caller is not interested in recovering an
 * opaque pointer to the newly-defined property.
 *
 * If propp is non-null and JSDefinePropOp succeeds, its caller must be sure
 * to drop *propp using JSObjectOps.dropProperty in short order, just as with
 * JSLookupPropOp.
 */
typedef JSBool
(* JS_DLL_CALLBACK JSDefinePropOp)(JSContext *cx, JSObject *obj,
                                   jsid id, jsval value,
                                   JSPropertyOp getter, JSPropertyOp setter,
                                   uintN attrs, JSProperty **propp);

/*
 * Get, set, or delete obj[id], returning false on error or exception, true
 * on success.  If getting or setting, the new value is returned in *vp on
 * success.  If deleting without error, *vp will be JSVAL_FALSE if obj[id] is
 * permanent, and JSVAL_TRUE if id named a direct property of obj that was in
 * fact deleted, or if id names no direct property of obj (id could name a
 * prototype property, or no property in obj or its prototype chain).
 */
typedef JSBool
(* JS_DLL_CALLBACK JSPropertyIdOp)(JSContext *cx, JSObject *obj, jsid id,
                                   jsval *vp);

/*
 * Get or set attributes of the property obj[id].  Return false on error or
 * exception, true with current attributes in *attrsp.  If prop is non-null,
 * it must come from the *propp out parameter of a prior JSDefinePropOp or
 * JSLookupPropOp call.
 */
typedef JSBool
(* JS_DLL_CALLBACK JSAttributesOp)(JSContext *cx, JSObject *obj, jsid id,
                                   JSProperty *prop, uintN *attrsp);

/*
 * JSObjectOps.checkAccess type: check whether obj[id] may be accessed per
 * mode, returning false on error/exception, true on success with obj[id]'s
 * last-got value in *vp, and its attributes in *attrsp.
 */
typedef JSBool
(* JS_DLL_CALLBACK JSCheckAccessIdOp)(JSContext *cx, JSObject *obj, jsid id,
                                      JSAccessMode mode, jsval *vp,
                                      uintN *attrsp);

/*
 * A generic type for functions mapping an object to another object, or null
 * if an error or exception was thrown on cx.  Used by JSObjectOps.thisObject
 * at present.
 */
typedef JSObject *
(* JS_DLL_CALLBACK JSObjectOp)(JSContext *cx, JSObject *obj);

/*
 * A generic type for functions taking a context, object, and property, with
 * no return value.  Used by JSObjectOps.dropProperty lwrrently (see above,
 * JSDefinePropOp and JSLookupPropOp, for the object-locking protocol in which
 * dropProperty participates).
 */
typedef void
(* JS_DLL_CALLBACK JSPropertyRefOp)(JSContext *cx, JSObject *obj,
                                    JSProperty *prop);

/*
 * Function type for JSObjectOps.setProto and JSObjectOps.setParent.  These
 * hooks must check for cycles without deadlocking, and otherwise take special
 * steps.  See jsobj.c, js_SetProtoOrParent, for an example.
 */
typedef JSBool
(* JS_DLL_CALLBACK JSSetObjectSlotOp)(JSContext *cx, JSObject *obj,
                                      uint32 slot, JSObject *pobj);

/*
 * Get and set a required slot, one that should already have been allocated.
 * These operations are infallible, so required slots must be pre-allocated,
 * or implementations must suppress out-of-memory errors.  The native ops
 * (js_ObjectOps, see jsobj.c) access slots reserved by including a call to
 * the JSCLASS_HAS_RESERVED_SLOTS(n) macro in the JSClass.flags initializer.
 *
 * NB: the slot parameter is a zero-based index into obj->slots[], unlike the
 * index parameter to the JS_GetReservedSlot and JS_SetReservedSlot API entry
 * points, which is a zero-based index into the JSCLASS_RESERVED_SLOTS(clasp)
 * reserved slots that come after the initial well-known slots: proto, parent,
 * class, and optionally, the private data slot.
 */
typedef jsval
(* JS_DLL_CALLBACK JSGetRequiredSlotOp)(JSContext *cx, JSObject *obj,
                                        uint32 slot);

typedef JSBool
(* JS_DLL_CALLBACK JSSetRequiredSlotOp)(JSContext *cx, JSObject *obj,
                                        uint32 slot, jsval v);

typedef JSObject *
(* JS_DLL_CALLBACK JSGetMethodOp)(JSContext *cx, JSObject *obj, jsid id,
                                  jsval *vp);

typedef JSBool
(* JS_DLL_CALLBACK JSSetMethodOp)(JSContext *cx, JSObject *obj, jsid id,
                                  jsval *vp);

typedef JSBool
(* JS_DLL_CALLBACK JSEnumerateValuesOp)(JSContext *cx, JSObject *obj,
                                        JSIterateOp enum_op,
                                        jsval *statep, jsid *idp, jsval *vp);

typedef JSBool
(* JS_DLL_CALLBACK JSEqualityOp)(JSContext *cx, JSObject *obj, jsval v,
                                 JSBool *bp);

typedef JSBool
(* JS_DLL_CALLBACK JSConcatenateOp)(JSContext *cx, JSObject *obj, jsval v,
                                    jsval *vp);

/* Typedef for native functions called by the JS VM. */

typedef JSBool
(* JS_DLL_CALLBACK JSNative)(JSContext *cx, JSObject *obj, uintN argc,
                             jsval *argv, jsval *rval);

/* Callbacks and their arguments. */

typedef enum JSContextOp {
    JSCONTEXT_NEW,
    JSCONTEXT_DESTROY
} JSContextOp;

/*
 * The possible values for contextOp when the runtime calls the callback are:
 *   JSCONTEXT_NEW      JS_NewContext succesfully created a new JSContext
 *                      instance. The callback can initialize the instance as
 *                      required. If the callback returns false, the instance
 *                      will be destroyed and JS_NewContext returns null. In
 *                      this case the callback is not called again.
 *   JSCONTEXT_DESTROY  One of JS_DestroyContext* methods is called. The
 *                      callback may perform its own cleanup and must always
 *                      return true.
 *   Any other value    For future compatibility the callback must do nothing
 *                      and return true in this case.
 */
typedef JSBool
(* JS_DLL_CALLBACK JSContextCallback)(JSContext *cx, uintN contextOp);

typedef enum JSGCStatus {
    JSGC_BEGIN,
    JSGC_END,
    JSGC_MARK_END,
    JSGC_FINALIZE_END
} JSGCStatus;

typedef JSBool
(* JS_DLL_CALLBACK JSGCCallback)(JSContext *cx, JSGCStatus status);

typedef JSBool
(* JS_DLL_CALLBACK JSBranchCallback)(JSContext *cx, JSScript *script);

typedef void
(* JS_DLL_CALLBACK JSErrorReporter)(JSContext *cx, const char *message,
                                    JSErrorReport *report);

/*
 * Possible exception types. These types are part of a JSErrorFormatString
 * structure. They define which error to throw in case of a runtime error.
 * JSEXN_NONE marks an unthrowable error.
 */
typedef enum JSExnType {
    JSEXN_NONE = -1,
      JSEXN_ERR,
        JSEXN_INTERNALERR,
        JSEXN_EVALERR,
        JSEXN_RANGEERR,
        JSEXN_REFERENCEERR,
        JSEXN_SYNTAXERR,
        JSEXN_TYPEERR,
        JSEXN_URIERR,
        JSEXN_LIMIT
} JSExnType;

typedef struct JSErrorFormatString {
    /* The error format string (UTF-8 if JS_C_STRINGS_ARE_UTF8 is defined). */
    const char *format;

    /* The number of arguments to expand in the formatted error message. */
    uint16 argCount;

    /* One of the JSExnType constants above. */
    int16 exnType;
} JSErrorFormatString;

typedef const JSErrorFormatString *
(* JS_DLL_CALLBACK JSErrorCallback)(void *userRef, const char *locale,
                                    const uintN errorNumber);

#ifdef va_start
#define JS_ARGUMENT_FORMATTER_DEFINED 1

typedef JSBool
(* JS_DLL_CALLBACK JSArgumentFormatter)(JSContext *cx, const char *format,
                                        JSBool fromJS, jsval **vpp,
                                        va_list *app);
#endif

typedef JSBool
(* JS_DLL_CALLBACK JSLocaleToUpperCase)(JSContext *cx, JSString *src,
                                        jsval *rval);

typedef JSBool
(* JS_DLL_CALLBACK JSLocaleToLowerCase)(JSContext *cx, JSString *src,
                                        jsval *rval);

typedef JSBool
(* JS_DLL_CALLBACK JSLocaleCompare)(JSContext *cx,
                                    JSString *src1, JSString *src2,
                                    jsval *rval);

typedef JSBool
(* JS_DLL_CALLBACK JSLocaleToUnicode)(JSContext *cx, char *src, jsval *rval);

/*
 * Security protocol types.
 */
typedef struct JSPrincipals JSPrincipals;

/*
 * XDR-encode or -decode a principals instance, based on whether xdr->mode is
 * JSXDR_ENCODE, in which case *principalsp should be encoded; or JSXDR_DECODE,
 * in which case implementations must return a held (via JSPRINCIPALS_HOLD),
 * non-null *principalsp out parameter.  Return true on success, false on any
 * error, which the implementation must have reported.
 */
typedef JSBool
(* JS_DLL_CALLBACK JSPrincipalsTranscoder)(JSXDRState *xdr,
                                           JSPrincipals **principalsp);

/*
 * Return a weak reference to the principals associated with obj, possibly via
 * the immutable parent chain leading from obj to a top-level container (e.g.,
 * a window object in the DOM level 0).  If there are no principals associated
 * with obj, return null.  Therefore null does not mean an error was reported;
 * in no event should an error be reported or an exception be thrown by this
 * callback's implementation.
 */
typedef JSPrincipals *
(* JS_DLL_CALLBACK JSObjectPrincipalsFinder)(JSContext *cx, JSObject *obj);

JS_END_EXTERN_C

#endif /* jspubtd_h___ */
