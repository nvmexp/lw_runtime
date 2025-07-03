// Tencent is pleased to support the open source community by making RapidJSON available->
// 
// Copyright (C) 2015 THL A29 Limited, a Tencent company, and Milo Yip-> All rights reserved->
//
// Licensed under the MIT License (the "License"); you may not use this file except
// in compliance with the License-> You may obtain a copy of the License at
//
// http://opensource->org/licenses/MIT
//
// Unless required by applicable law or agreed to in writing, software distributed 
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
// CONDITIONS OF ANY KIND, either express or implied-> See the License for the 
// specific language governing permissions and limitations under the License->

#ifndef RAPIDJSON_SCHEMA_H_
#define RAPIDJSON_SCHEMA_H_

#include "document.h"
#include "pointer.h"
#include <cmath> // abs, floor

#if !defined(RAPIDJSON_SCHEMA_USE_INTERNALREGEX)
#define RAPIDJSON_SCHEMA_USE_INTERNALREGEX 1
#else
#define RAPIDJSON_SCHEMA_USE_INTERNALREGEX 0
#endif

#if !RAPIDJSON_SCHEMA_USE_INTERNALREGEX && !defined(RAPIDJSON_SCHEMA_USE_STDREGEX) && (__cplusplus >=201103L || (defined(_MSC_VER) && _MSC_VER >= 1800))
#define RAPIDJSON_SCHEMA_USE_STDREGEX 1
#else
#define RAPIDJSON_SCHEMA_USE_STDREGEX 0
#endif

#if RAPIDJSON_SCHEMA_USE_INTERNALREGEX
#include "internal/regex.h"
#elif RAPIDJSON_SCHEMA_USE_STDREGEX
#include <regex>
#endif

#if RAPIDJSON_SCHEMA_USE_INTERNALREGEX || RAPIDJSON_SCHEMA_USE_STDREGEX
#define RAPIDJSON_SCHEMA_HAS_REGEX 1
#else
#define RAPIDJSON_SCHEMA_HAS_REGEX 0
#endif

#ifndef RAPIDJSON_SCHEMA_VERBOSE
#define RAPIDJSON_SCHEMA_VERBOSE 0
#endif

#if RAPIDJSON_SCHEMA_VERBOSE
#include "stringbuffer.h"
#endif

RAPIDJSON_DIAG_PUSH

#if defined(__GNUC__)
RAPIDJSON_DIAG_OFF(effc++)
#endif

#ifdef __clang__
RAPIDJSON_DIAG_OFF(weak-vtables)
RAPIDJSON_DIAG_OFF(exit-time-destructors)
RAPIDJSON_DIAG_OFF(c++98-compat-pedantic)
RAPIDJSON_DIAG_OFF(variadic-macros)
#endif

#ifdef _MSC_VER
RAPIDJSON_DIAG_OFF(4512) // assignment operator could not be generated
#endif

RAPIDJSON_NAMESPACE_BEGIN

///////////////////////////////////////////////////////////////////////////////
// Verbose Utilities

#if RAPIDJSON_SCHEMA_VERBOSE

namespace internal {

inline void PrintIlwalidKeyword(const char* keyword) {
    printf("Fail keyword: %s\n", keyword);
}

inline void PrintIlwalidKeyword(const wchar_t* keyword) {
    wprintf(L"Fail keyword: %ls\n", keyword);
}

inline void PrintIlwalidDolwment(const char* document) {
    printf("Fail document: %s\n\n", document);
}

inline void PrintIlwalidDolwment(const wchar_t* document) {
    wprintf(L"Fail document: %ls\n\n", document);
}

inline void PrintValidatorPointers(unsigned depth, const char* s, const char* d) {
    printf("S: %*s%s\nD: %*s%s\n\n", depth * 4, " ", s, depth * 4, " ", d);
}

inline void PrintValidatorPointers(unsigned depth, const wchar_t* s, const wchar_t* d) {
    wprintf(L"S: %*ls%ls\nD: %*ls%ls\n\n", depth * 4, L" ", s, depth * 4, L" ", d);
}

} // namespace internal

#endif // RAPIDJSON_SCHEMA_VERBOSE

///////////////////////////////////////////////////////////////////////////////
// RAPIDJSON_ILWALID_KEYWORD_RETURN

#if RAPIDJSON_SCHEMA_VERBOSE
#define RAPIDJSON_ILWALID_KEYWORD_VERBOSE(keyword) internal::PrintIlwalidKeyword(keyword)
#else
#define RAPIDJSON_ILWALID_KEYWORD_VERBOSE(keyword)
#endif

#define RAPIDJSON_ILWALID_KEYWORD_RETURN(keyword)\
RAPIDJSON_MULTILINEMACRO_BEGIN\
    context.ilwalidKeyword = keyword.GetString();\
    RAPIDJSON_ILWALID_KEYWORD_VERBOSE(keyword.GetString());\
    return false;\
RAPIDJSON_MULTILINEMACRO_END

///////////////////////////////////////////////////////////////////////////////
// Forward declarations

template <typename ValueType, typename Allocator>
class GenericSchemaDolwment;

namespace internal {

template <typename SchemaDolwmentType>
class Schema;

///////////////////////////////////////////////////////////////////////////////
// ISchemaValidator

class ISchemaValidator {
public:
    virtual ~ISchemaValidator() {}
    virtual bool IsValid() const = 0;
};

///////////////////////////////////////////////////////////////////////////////
// ISchemaStateFactory

template <typename SchemaType>
class ISchemaStateFactory {
public:
    virtual ~ISchemaStateFactory() {}
    virtual ISchemaValidator* CreateSchemaValidator(const SchemaType&) = 0;
    virtual void DestroySchemaValidator(ISchemaValidator* validator) = 0;
    virtual void* CreateHasher() = 0;
    virtual uint64_t GetHashCode(void* hasher) = 0;
    virtual void DestroryHasher(void* hasher) = 0;
    virtual void* MallocState(size_t size) = 0;
    virtual void FreeState(void* p) = 0;
};

///////////////////////////////////////////////////////////////////////////////
// Hasher

// For comparison of compound value
template<typename Encoding, typename Allocator>
class Hasher {
public:
    typedef typename Encoding::Ch Ch;

    Hasher(Allocator* allocator = 0, size_t stackCapacity = kDefaultSize) : stack_(allocator, stackCapacity) {}

    bool Null() { return WriteType(kNullType); }
    bool Bool(bool b) { return WriteType(b ? kTrueType : kFalseType); }
    bool Int(int i) { Number n; n.u.i = i; n.d = static_cast<double>(i); return WriteNumber(n); }
    bool Uint(unsigned u) { Number n; n.u.u = u; n.d = static_cast<double>(u); return WriteNumber(n); }
    bool Int64(int64_t i) { Number n; n.u.i = i; n.d = static_cast<double>(i); return WriteNumber(n); }
    bool Uint64(uint64_t u) { Number n; n.u.u = u; n.d = static_cast<double>(u); return WriteNumber(n); }
    bool Double(double d) { 
        Number n; 
        if (d < 0) n.u.i = static_cast<int64_t>(d);
        else       n.u.u = static_cast<uint64_t>(d); 
        n.d = d;
        return WriteNumber(n);
    }

    bool RawNumber(const Ch* str, SizeType len, bool) {
        WriteBuffer(kNumberType, str, len * sizeof(Ch));
        return true;
    }

    bool String(const Ch* str, SizeType len, bool) {
        WriteBuffer(kStringType, str, len * sizeof(Ch));
        return true;
    }

    bool StartObject() { return true; }
    bool Key(const Ch* str, SizeType len, bool copy) { return String(str, len, copy); }
    bool EndObject(SizeType memberCount) { 
        uint64_t h = Hash(0, kObjectType);
        uint64_t* kv = stack_.template Pop<uint64_t>(memberCount * 2);
        for (SizeType i = 0; i < memberCount; i++)
            h ^= Hash(kv[i * 2], kv[i * 2 + 1]);  // Use xor to achieve member order insensitive
        *stack_.template Push<uint64_t>() = h;
        return true;
    }
    
    bool StartArray() { return true; }
    bool EndArray(SizeType elementCount) { 
        uint64_t h = Hash(0, kArrayType);
        uint64_t* e = stack_.template Pop<uint64_t>(elementCount);
        for (SizeType i = 0; i < elementCount; i++)
            h = Hash(h, e[i]); // Use hash to achieve element order sensitive
        *stack_.template Push<uint64_t>() = h;
        return true;
    }

    bool IsValid() const { return stack_.GetSize() == sizeof(uint64_t); }

    uint64_t GetHashCode() const {
        RAPIDJSON_ASSERT(IsValid());
        return *stack_.template Top<uint64_t>();
    }

private:
    static const size_t kDefaultSize = 256;
    struct Number {
        union U {
            uint64_t u;
            int64_t i;
        }u;
        double d;
    };

    bool WriteType(Type type) { return WriteBuffer(type, 0, 0); }
    
    bool WriteNumber(const Number& n) { return WriteBuffer(kNumberType, &n, sizeof(n)); }
    
    bool WriteBuffer(Type type, const void* data, size_t len) {
        // FLW-1a from http://isthe.com/chongo/tech/comp/flw/
        uint64_t h = Hash(RAPIDJSON_UINT64_C2(0x84222325, 0xcbf29ce4), type);
        const unsigned char* d = static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < len; i++)
            h = Hash(h, d[i]);
        *stack_.template Push<uint64_t>() = h;
        return true;
    }

    static uint64_t Hash(uint64_t h, uint64_t d) {
        static const uint64_t kPrime = RAPIDJSON_UINT64_C2(0x00000100, 0x000001b3);
        h ^= d;
        h *= kPrime;
        return h;
    }

    Stack<Allocator> stack_;
};

///////////////////////////////////////////////////////////////////////////////
// SchemaValidationContext

template <typename SchemaDolwmentType>
struct SchemaValidationContext {
    typedef Schema<SchemaDolwmentType> SchemaType;
    typedef ISchemaStateFactory<SchemaType> SchemaValidatorFactoryType;
    typedef typename SchemaType::ValueType ValueType;
    typedef typename ValueType::Ch Ch;

    enum PatterlwalidatorType {
        kPatterlwalidatorOnly,
        kPatterlwalidatorWithProperty,
        kPatterlwalidatorWithAdditionalProperty
    };

    SchemaValidationContext(SchemaValidatorFactoryType& f, const SchemaType* s) :
        factory(f),
        schema(s),
        valueSchema(),
        ilwalidKeyword(),
        hasher(),
        arrayElementHashCodes(),
        validators(),
        validatorCount(),
        patternPropertiesValidators(),
        patternPropertiesValidatorCount(),
        patternPropertiesSchemas(),
        patternPropertiesSchemaCount(),
        valuePatterlwalidatorType(kPatterlwalidatorOnly),
        propertyExist(),
        inArray(false),
        valueUniqueness(false),
        arrayUniqueness(false)
    {
    }

    ~SchemaValidationContext() {
        if (hasher)
            factory.DestroryHasher(hasher);
        if (validators) {
            for (SizeType i = 0; i < validatorCount; i++)
                factory.DestroySchemaValidator(validators[i]);
            factory.FreeState(validators);
        }
        if (patternPropertiesValidators) {
            for (SizeType i = 0; i < patternPropertiesValidatorCount; i++)
                factory.DestroySchemaValidator(patternPropertiesValidators[i]);
            factory.FreeState(patternPropertiesValidators);
        }
        if (patternPropertiesSchemas)
            factory.FreeState(patternPropertiesSchemas);
        if (propertyExist)
            factory.FreeState(propertyExist);
    }

    SchemaValidatorFactoryType& factory;
    const SchemaType* schema;
    const SchemaType* valueSchema;
    const Ch* ilwalidKeyword;
    void* hasher; // Only validator access
    void* arrayElementHashCodes; // Only validator access this
    ISchemaValidator** validators;
    SizeType validatorCount;
    ISchemaValidator** patternPropertiesValidators;
    SizeType patternPropertiesValidatorCount;
    const SchemaType** patternPropertiesSchemas;
    SizeType patternPropertiesSchemaCount;
    PatterlwalidatorType valuePatterlwalidatorType;
    PatterlwalidatorType objectPatterlwalidatorType;
    SizeType arrayElementIndex;
    bool* propertyExist;
    bool inArray;
    bool valueUniqueness;
    bool arrayUniqueness;
};

///////////////////////////////////////////////////////////////////////////////
// Schema

template <typename SchemaDolwmentType>
class Schema {
public:
    typedef typename SchemaDolwmentType::ValueType ValueType;
    typedef typename SchemaDolwmentType::AllocatorType AllocatorType;
    typedef typename SchemaDolwmentType::PointerType PointerType;
    typedef typename ValueType::EncodingType EncodingType;
    typedef typename EncodingType::Ch Ch;
    typedef SchemaValidationContext<SchemaDolwmentType> Context;
    typedef Schema<SchemaDolwmentType> SchemaType;
    typedef GenericValue<EncodingType, AllocatorType> SValue;
    friend class GenericSchemaDolwment<ValueType, AllocatorType>;

    Schema(SchemaDolwmentType* schemaDolwment, const PointerType& p, const ValueType& value, const ValueType& document, AllocatorType* allocator) :
        allocator_(allocator),
        typeless_(schemaDolwment->GetTypeless()),
        enum_(),
        enumCount_(),
        not_(),
        type_((1 << kTotalSchemaType) - 1), // typeless
        validatorCount_(),
        properties_(),
        additionalPropertiesSchema_(),
        patternProperties_(),
        patternPropertyCount_(),
        propertyCount_(),
        minProperties_(),
        maxProperties_(SizeType(~0)),
        additionalProperties_(true),
        hasDependencies_(),
        hasRequired_(),
        hasSchemaDependencies_(),
        additionalItemsSchema_(),
        itemsList_(),
        itemsTuple_(),
        itemsTupleCount_(),
        minItems_(),
        maxItems_(SizeType(~0)),
        additionalItems_(true),
        uniqueItems_(false),
        pattern_(),
        minLength_(0),
        maxLength_(~SizeType(0)),
        exclusiveMinimum_(false),
        exclusiveMaximum_(false)
    {
        typedef typename SchemaDolwmentType::ValueType ValueType;
        typedef typename ValueType::ConstValueIterator ConstValueIterator;
        typedef typename ValueType::ConstMemberIterator ConstMemberIterator;

        if (!value.IsObject())
            return;

        if (const ValueType* v = GetMember(value, GetTypeString())) {
            type_ = 0;
            if (v->IsString())
                AddType(*v);
            else if (v->IsArray())
                for (ConstValueIterator itr = v->Begin(); itr != v->End(); ++itr)
                    AddType(*itr);
        }

        if (const ValueType* v = GetMember(value, GetEnumString()))
            if (v->IsArray() && v->Size() > 0) {
                enum_ = static_cast<uint64_t*>(allocator_->Malloc(sizeof(uint64_t) * v->Size()));
                for (ConstValueIterator itr = v->Begin(); itr != v->End(); ++itr) {
                    typedef Hasher<EncodingType, MemoryPoolAllocator<> > EnumHasherType;
                    char buffer[256 + 24];
                    MemoryPoolAllocator<> hasherAllocator(buffer, sizeof(buffer));
                    EnumHasherType h(&hasherAllocator, 256);
                    itr->Accept(h);
                    enum_[enumCount_++] = h.GetHashCode();
                }
            }

        if (schemaDolwment) {
            AssignIfExist(allOf_, *schemaDolwment, p, value, GetAllOfString(), document);
            AssignIfExist(anyOf_, *schemaDolwment, p, value, GetAnyOfString(), document);
            AssignIfExist(oneOf_, *schemaDolwment, p, value, GetOneOfString(), document);
        }

        if (const ValueType* v = GetMember(value, GetNotString())) {
            schemaDolwment->CreateSchema(&not_, p.Append(GetNotString(), allocator_), *v, document);
            notValidatorIndex_ = validatorCount_;
            validatorCount_++;
        }

        // Object

        const ValueType* properties = GetMember(value, GetPropertiesString());
        const ValueType* required = GetMember(value, GetRequiredString());
        const ValueType* dependencies = GetMember(value, GetDependenciesString());
        {
            // Gather properties from properties/required/dependencies
            SValue allProperties(kArrayType);

            if (properties && properties->IsObject())
                for (ConstMemberIterator itr = properties->MemberBegin(); itr != properties->MemberEnd(); ++itr)
                    AddUniqueElement(allProperties, itr->name);
            
            if (required && required->IsArray())
                for (ConstValueIterator itr = required->Begin(); itr != required->End(); ++itr)
                    if (itr->IsString())
                        AddUniqueElement(allProperties, *itr);

            if (dependencies && dependencies->IsObject())
                for (ConstMemberIterator itr = dependencies->MemberBegin(); itr != dependencies->MemberEnd(); ++itr) {
                    AddUniqueElement(allProperties, itr->name);
                    if (itr->value.IsArray())
                        for (ConstValueIterator i = itr->value.Begin(); i != itr->value.End(); ++i)
                            if (i->IsString())
                                AddUniqueElement(allProperties, *i);
                }

            if (allProperties.Size() > 0) {
                propertyCount_ = allProperties.Size();
                properties_ = static_cast<Property*>(allocator_->Malloc(sizeof(Property) * propertyCount_));
                for (SizeType i = 0; i < propertyCount_; i++) {
                    new (&properties_[i]) Property();
                    properties_[i].name = allProperties[i];
                    properties_[i].schema = typeless_;
                }
            }
        }

        if (properties && properties->IsObject()) {
            PointerType q = p.Append(GetPropertiesString(), allocator_);
            for (ConstMemberIterator itr = properties->MemberBegin(); itr != properties->MemberEnd(); ++itr) {
                SizeType index;
                if (FindPropertyIndex(itr->name, &index))
                    schemaDolwment->CreateSchema(&properties_[index].schema, q.Append(itr->name, allocator_), itr->value, document);
            }
        }

        if (const ValueType* v = GetMember(value, GetPatternPropertiesString())) {
            PointerType q = p.Append(GetPatternPropertiesString(), allocator_);
            patternProperties_ = static_cast<PatternProperty*>(allocator_->Malloc(sizeof(PatternProperty) * v->MemberCount()));
            patternPropertyCount_ = 0;

            for (ConstMemberIterator itr = v->MemberBegin(); itr != v->MemberEnd(); ++itr) {
                new (&patternProperties_[patternPropertyCount_]) PatternProperty();
                patternProperties_[patternPropertyCount_].pattern = CreatePattern(itr->name);
                schemaDolwment->CreateSchema(&patternProperties_[patternPropertyCount_].schema, q.Append(itr->name, allocator_), itr->value, document);
                patternPropertyCount_++;
            }
        }

        if (required && required->IsArray())
            for (ConstValueIterator itr = required->Begin(); itr != required->End(); ++itr)
                if (itr->IsString()) {
                    SizeType index;
                    if (FindPropertyIndex(*itr, &index)) {
                        properties_[index].required = true;
                        hasRequired_ = true;
                    }
                }

        if (dependencies && dependencies->IsObject()) {
            PointerType q = p.Append(GetDependenciesString(), allocator_);
            hasDependencies_ = true;
            for (ConstMemberIterator itr = dependencies->MemberBegin(); itr != dependencies->MemberEnd(); ++itr) {
                SizeType sourceIndex;
                if (FindPropertyIndex(itr->name, &sourceIndex)) {
                    if (itr->value.IsArray()) {
                        properties_[sourceIndex].dependencies = static_cast<bool*>(allocator_->Malloc(sizeof(bool) * propertyCount_));
                        std::memset(properties_[sourceIndex].dependencies, 0, sizeof(bool)* propertyCount_);
                        for (ConstValueIterator targetItr = itr->value.Begin(); targetItr != itr->value.End(); ++targetItr) {
                            SizeType targetIndex;
                            if (FindPropertyIndex(*targetItr, &targetIndex))
                                properties_[sourceIndex].dependencies[targetIndex] = true;
                        }
                    }
                    else if (itr->value.IsObject()) {
                        hasSchemaDependencies_ = true;
                        schemaDolwment->CreateSchema(&properties_[sourceIndex].dependenciesSchema, q.Append(itr->name, allocator_), itr->value, document);
                        properties_[sourceIndex].dependenciesValidatorIndex = validatorCount_;
                        validatorCount_++;
                    }
                }
            }
        }

        if (const ValueType* v = GetMember(value, GetAdditionalPropertiesString())) {
            if (v->IsBool())
                additionalProperties_ = v->GetBool();
            else if (v->IsObject())
                schemaDolwment->CreateSchema(&additionalPropertiesSchema_, p.Append(GetAdditionalPropertiesString(), allocator_), *v, document);
        }

        AssignIfExist(minProperties_, value, GetMinPropertiesString());
        AssignIfExist(maxProperties_, value, GetMaxPropertiesString());

        // Array
        if (const ValueType* v = GetMember(value, GetItemsString())) {
            PointerType q = p.Append(GetItemsString(), allocator_);
            if (v->IsObject()) // List validation
                schemaDolwment->CreateSchema(&itemsList_, q, *v, document);
            else if (v->IsArray()) { // Tuple validation
                itemsTuple_ = static_cast<const Schema**>(allocator_->Malloc(sizeof(const Schema*) * v->Size()));
                SizeType index = 0;
                for (ConstValueIterator itr = v->Begin(); itr != v->End(); ++itr, index++)
                    schemaDolwment->CreateSchema(&itemsTuple_[itemsTupleCount_++], q.Append(index, allocator_), *itr, document);
            }
        }

        AssignIfExist(minItems_, value, GetMinItemsString());
        AssignIfExist(maxItems_, value, GetMaxItemsString());

        if (const ValueType* v = GetMember(value, GetAdditionalItemsString())) {
            if (v->IsBool())
                additionalItems_ = v->GetBool();
            else if (v->IsObject())
                schemaDolwment->CreateSchema(&additionalItemsSchema_, p.Append(GetAdditionalItemsString(), allocator_), *v, document);
        }

        AssignIfExist(uniqueItems_, value, GetUniqueItemsString());

        // String
        AssignIfExist(minLength_, value, GetMinLengthString());
        AssignIfExist(maxLength_, value, GetMaxLengthString());

        if (const ValueType* v = GetMember(value, GetPatternString()))
            pattern_ = CreatePattern(*v);

        // Number
        if (const ValueType* v = GetMember(value, GetMinimumString()))
            if (v->IsNumber())
                minimum_.CopyFrom(*v, *allocator_);

        if (const ValueType* v = GetMember(value, GetMaximumString()))
            if (v->IsNumber())
                maximum_.CopyFrom(*v, *allocator_);

        AssignIfExist(exclusiveMinimum_, value, GetExclusiveMinimumString());
        AssignIfExist(exclusiveMaximum_, value, GetExclusiveMaximumString());

        if (const ValueType* v = GetMember(value, GetMultipleOfString()))
            if (v->IsNumber() && v->GetDouble() > 0.0)
                multipleOf_.CopyFrom(*v, *allocator_);
    }

    ~Schema() {
        AllocatorType::Free(enum_);
        if (properties_) {
            for (SizeType i = 0; i < propertyCount_; i++)
                properties_[i].~Property();
            AllocatorType::Free(properties_);
        }
        if (patternProperties_) {
            for (SizeType i = 0; i < patternPropertyCount_; i++)
                patternProperties_[i].~PatternProperty();
            AllocatorType::Free(patternProperties_);
        }
        AllocatorType::Free(itemsTuple_);
#if RAPIDJSON_SCHEMA_HAS_REGEX
        if (pattern_) {
            pattern_->~RegexType();
            AllocatorType::Free(pattern_);
        }
#endif
    }

    bool Begilwalue(Context& context) const {
        if (context.inArray) {
            if (uniqueItems_)
                context.valueUniqueness = true;

            if (itemsList_)
                context.valueSchema = itemsList_;
            else if (itemsTuple_) {
                if (context.arrayElementIndex < itemsTupleCount_)
                    context.valueSchema = itemsTuple_[context.arrayElementIndex];
                else if (additionalItemsSchema_)
                    context.valueSchema = additionalItemsSchema_;
                else if (additionalItems_)
                    context.valueSchema = typeless_;
                else
                    RAPIDJSON_ILWALID_KEYWORD_RETURN(GetItemsString());
            }
            else
                context.valueSchema = typeless_;

            context.arrayElementIndex++;
        }
        return true;
    }

    RAPIDJSON_FORCEINLINE bool EndValue(Context& context) const {
        if (context.patternPropertiesValidatorCount > 0) {
            bool otherValid = false;
            SizeType count = context.patternPropertiesValidatorCount;
            if (context.objectPatterlwalidatorType != Context::kPatterlwalidatorOnly)
                otherValid = context.patternPropertiesValidators[--count]->IsValid();

            bool patterlwalid = true;
            for (SizeType i = 0; i < count; i++)
                if (!context.patternPropertiesValidators[i]->IsValid()) {
                    patterlwalid = false;
                    break;
                }

            if (context.objectPatterlwalidatorType == Context::kPatterlwalidatorOnly) {
                if (!patterlwalid)
                    RAPIDJSON_ILWALID_KEYWORD_RETURN(GetPatternPropertiesString());
            }
            else if (context.objectPatterlwalidatorType == Context::kPatterlwalidatorWithProperty) {
                if (!patterlwalid || !otherValid)
                    RAPIDJSON_ILWALID_KEYWORD_RETURN(GetPatternPropertiesString());
            }
            else if (!patterlwalid && !otherValid) // kPatterlwalidatorWithAdditionalProperty)
                RAPIDJSON_ILWALID_KEYWORD_RETURN(GetPatternPropertiesString());
        }

        if (enum_) {
            const uint64_t h = context.factory.GetHashCode(context.hasher);
            for (SizeType i = 0; i < enumCount_; i++)
                if (enum_[i] == h)
                    goto foundEnum;
            RAPIDJSON_ILWALID_KEYWORD_RETURN(GetEnumString());
            foundEnum:;
        }

        if (allOf_.schemas)
            for (SizeType i = allOf_.begin; i < allOf_.begin + allOf_.count; i++)
                if (!context.validators[i]->IsValid())
                    RAPIDJSON_ILWALID_KEYWORD_RETURN(GetAllOfString());
        
        if (anyOf_.schemas) {
            for (SizeType i = anyOf_.begin; i < anyOf_.begin + anyOf_.count; i++)
                if (context.validators[i]->IsValid())
                    goto foundAny;
            RAPIDJSON_ILWALID_KEYWORD_RETURN(GetAnyOfString());
            foundAny:;
        }

        if (oneOf_.schemas) {
            bool oneValid = false;
            for (SizeType i = oneOf_.begin; i < oneOf_.begin + oneOf_.count; i++)
                if (context.validators[i]->IsValid()) {
                    if (oneValid)
                        RAPIDJSON_ILWALID_KEYWORD_RETURN(GetOneOfString());
                    else
                        oneValid = true;
                }
            if (!oneValid)
                RAPIDJSON_ILWALID_KEYWORD_RETURN(GetOneOfString());
        }

        if (not_ && context.validators[notValidatorIndex_]->IsValid())
            RAPIDJSON_ILWALID_KEYWORD_RETURN(GetNotString());

        return true;
    }

    bool Null(Context& context) const { 
        if (!(type_ & (1 << kNullSchemaType)))
            RAPIDJSON_ILWALID_KEYWORD_RETURN(GetTypeString());
        return CreateParallelValidator(context);
    }
    
    bool Bool(Context& context, bool) const { 
        if (!(type_ & (1 << kBooleanSchemaType)))
            RAPIDJSON_ILWALID_KEYWORD_RETURN(GetTypeString());
        return CreateParallelValidator(context);
    }

    bool Int(Context& context, int i) const {
        if (!CheckInt(context, i))
            return false;
        return CreateParallelValidator(context);
    }

    bool Uint(Context& context, unsigned u) const {
        if (!CheckUint(context, u))
            return false;
        return CreateParallelValidator(context);
    }

    bool Int64(Context& context, int64_t i) const {
        if (!CheckInt(context, i))
            return false;
        return CreateParallelValidator(context);
    }

    bool Uint64(Context& context, uint64_t u) const {
        if (!CheckUint(context, u))
            return false;
        return CreateParallelValidator(context);
    }

    bool Double(Context& context, double d) const {
        if (!(type_ & (1 << kNumberSchemaType)))
            RAPIDJSON_ILWALID_KEYWORD_RETURN(GetTypeString());

        if (!minimum_.IsNull() && !CheckDoubleMinimum(context, d))
            return false;

        if (!maximum_.IsNull() && !CheckDoubleMaximum(context, d))
            return false;
        
        if (!multipleOf_.IsNull() && !CheckDoubleMultipleOf(context, d))
            return false;
        
        return CreateParallelValidator(context);
    }
    
    bool String(Context& context, const Ch* str, SizeType length, bool) const {
        if (!(type_ & (1 << kStringSchemaType)))
            RAPIDJSON_ILWALID_KEYWORD_RETURN(GetTypeString());

        if (minLength_ != 0 || maxLength_ != SizeType(~0)) {
            SizeType count;
            if (internal::CountStringCodePoint<EncodingType>(str, length, &count)) {
                if (count < minLength_)
                    RAPIDJSON_ILWALID_KEYWORD_RETURN(GetMinLengthString());
                if (count > maxLength_)
                    RAPIDJSON_ILWALID_KEYWORD_RETURN(GetMaxLengthString());
            }
        }

        if (pattern_ && !IsPatternMatch(pattern_, str, length))
            RAPIDJSON_ILWALID_KEYWORD_RETURN(GetPatternString());

        return CreateParallelValidator(context);
    }

    bool StartObject(Context& context) const { 
        if (!(type_ & (1 << kObjectSchemaType)))
            RAPIDJSON_ILWALID_KEYWORD_RETURN(GetTypeString());

        if (hasDependencies_ || hasRequired_) {
            context.propertyExist = static_cast<bool*>(context.factory.MallocState(sizeof(bool) * propertyCount_));
            std::memset(context.propertyExist, 0, sizeof(bool) * propertyCount_);
        }

        if (patternProperties_) { // pre-allocate schema array
            SizeType count = patternPropertyCount_ + 1; // extra for valuePatterlwalidatorType
            context.patternPropertiesSchemas = static_cast<const SchemaType**>(context.factory.MallocState(sizeof(const SchemaType*) * count));
            context.patternPropertiesSchemaCount = 0;
            std::memset(context.patternPropertiesSchemas, 0, sizeof(SchemaType*) * count);
        }

        return CreateParallelValidator(context);
    }
    
    bool Key(Context& context, const Ch* str, SizeType len, bool) const {
        if (patternProperties_) {
            context.patternPropertiesSchemaCount = 0;
            for (SizeType i = 0; i < patternPropertyCount_; i++)
                if (patternProperties_[i].pattern && IsPatternMatch(patternProperties_[i].pattern, str, len)) {
                    context.patternPropertiesSchemas[context.patternPropertiesSchemaCount++] = patternProperties_[i].schema;
                    context.valueSchema = typeless_;
                }
        }

        SizeType index;
        if (FindPropertyIndex(ValueType(str, len).Move(), &index)) {
            if (context.patternPropertiesSchemaCount > 0) {
                context.patternPropertiesSchemas[context.patternPropertiesSchemaCount++] = properties_[index].schema;
                context.valueSchema = typeless_;
                context.valuePatterlwalidatorType = Context::kPatterlwalidatorWithProperty;
            }
            else
                context.valueSchema = properties_[index].schema;

            if (context.propertyExist)
                context.propertyExist[index] = true;

            return true;
        }

        if (additionalPropertiesSchema_) {
            if (additionalPropertiesSchema_ && context.patternPropertiesSchemaCount > 0) {
                context.patternPropertiesSchemas[context.patternPropertiesSchemaCount++] = additionalPropertiesSchema_;
                context.valueSchema = typeless_;
                context.valuePatterlwalidatorType = Context::kPatterlwalidatorWithAdditionalProperty;
            }
            else
                context.valueSchema = additionalPropertiesSchema_;
            return true;
        }
        else if (additionalProperties_) {
            context.valueSchema = typeless_;
            return true;
        }

        if (context.patternPropertiesSchemaCount == 0) // patternProperties are not additional properties
            RAPIDJSON_ILWALID_KEYWORD_RETURN(GetAdditionalPropertiesString());

        return true;
    }

    bool EndObject(Context& context, SizeType memberCount) const {
        if (hasRequired_)
            for (SizeType index = 0; index < propertyCount_; index++)
                if (properties_[index].required)
                    if (!context.propertyExist[index])
                        RAPIDJSON_ILWALID_KEYWORD_RETURN(GetRequiredString());

        if (memberCount < minProperties_)
            RAPIDJSON_ILWALID_KEYWORD_RETURN(GetMinPropertiesString());

        if (memberCount > maxProperties_)
            RAPIDJSON_ILWALID_KEYWORD_RETURN(GetMaxPropertiesString());

        if (hasDependencies_) {
            for (SizeType sourceIndex = 0; sourceIndex < propertyCount_; sourceIndex++)
                if (context.propertyExist[sourceIndex]) {
                    if (properties_[sourceIndex].dependencies) {
                        for (SizeType targetIndex = 0; targetIndex < propertyCount_; targetIndex++)
                            if (properties_[sourceIndex].dependencies[targetIndex] && !context.propertyExist[targetIndex])
                                RAPIDJSON_ILWALID_KEYWORD_RETURN(GetDependenciesString());
                    }
                    else if (properties_[sourceIndex].dependenciesSchema)
                        if (!context.validators[properties_[sourceIndex].dependenciesValidatorIndex]->IsValid())
                            RAPIDJSON_ILWALID_KEYWORD_RETURN(GetDependenciesString());
                }
        }

        return true;
    }

    bool StartArray(Context& context) const { 
        if (!(type_ & (1 << kArraySchemaType)))
            RAPIDJSON_ILWALID_KEYWORD_RETURN(GetTypeString());

        context.arrayElementIndex = 0;
        context.inArray = true;

        return CreateParallelValidator(context);
    }

    bool EndArray(Context& context, SizeType elementCount) const { 
        context.inArray = false;
        
        if (elementCount < minItems_)
            RAPIDJSON_ILWALID_KEYWORD_RETURN(GetMinItemsString());
        
        if (elementCount > maxItems_)
            RAPIDJSON_ILWALID_KEYWORD_RETURN(GetMaxItemsString());

        return true;
    }

    // Generate functions for string literal according to Ch
#define RAPIDJSON_STRING_(name, ...) \
    static const ValueType& Get##name##String() {\
        static const Ch s[] = { __VA_ARGS__, '\0' };\
        static const ValueType v(s, static_cast<SizeType>(sizeof(s) / sizeof(Ch) - 1));\
        return v;\
    }

    RAPIDJSON_STRING_(Null, 'n', 'u', 'l', 'l')
    RAPIDJSON_STRING_(Boolean, 'b', 'o', 'o', 'l', 'e', 'a', 'n')
    RAPIDJSON_STRING_(Object, 'o', 'b', 'j', 'e', 'c', 't')
    RAPIDJSON_STRING_(Array, 'a', 'r', 'r', 'a', 'y')
    RAPIDJSON_STRING_(String, 's', 't', 'r', 'i', 'n', 'g')
    RAPIDJSON_STRING_(Number, 'n', 'u', 'm', 'b', 'e', 'r')
    RAPIDJSON_STRING_(Integer, 'i', 'n', 't', 'e', 'g', 'e', 'r')
    RAPIDJSON_STRING_(Type, 't', 'y', 'p', 'e')
    RAPIDJSON_STRING_(Enum, 'e', 'n', 'u', 'm')
    RAPIDJSON_STRING_(AllOf, 'a', 'l', 'l', 'O', 'f')
    RAPIDJSON_STRING_(AnyOf, 'a', 'n', 'y', 'O', 'f')
    RAPIDJSON_STRING_(OneOf, 'o', 'n', 'e', 'O', 'f')
    RAPIDJSON_STRING_(Not, 'n', 'o', 't')
    RAPIDJSON_STRING_(Properties, 'p', 'r', 'o', 'p', 'e', 'r', 't', 'i', 'e', 's')
    RAPIDJSON_STRING_(Required, 'r', 'e', 'q', 'u', 'i', 'r', 'e', 'd')
    RAPIDJSON_STRING_(Dependencies, 'd', 'e', 'p', 'e', 'n', 'd', 'e', 'n', 'c', 'i', 'e', 's')
    RAPIDJSON_STRING_(PatternProperties, 'p', 'a', 't', 't', 'e', 'r', 'n', 'P', 'r', 'o', 'p', 'e', 'r', 't', 'i', 'e', 's')
    RAPIDJSON_STRING_(AdditionalProperties, 'a', 'd', 'd', 'i', 't', 'i', 'o', 'n', 'a', 'l', 'P', 'r', 'o', 'p', 'e', 'r', 't', 'i', 'e', 's')
    RAPIDJSON_STRING_(MinProperties, 'm', 'i', 'n', 'P', 'r', 'o', 'p', 'e', 'r', 't', 'i', 'e', 's')
    RAPIDJSON_STRING_(MaxProperties, 'm', 'a', 'x', 'P', 'r', 'o', 'p', 'e', 'r', 't', 'i', 'e', 's')
    RAPIDJSON_STRING_(Items, 'i', 't', 'e', 'm', 's')
    RAPIDJSON_STRING_(MinItems, 'm', 'i', 'n', 'I', 't', 'e', 'm', 's')
    RAPIDJSON_STRING_(MaxItems, 'm', 'a', 'x', 'I', 't', 'e', 'm', 's')
    RAPIDJSON_STRING_(AdditionalItems, 'a', 'd', 'd', 'i', 't', 'i', 'o', 'n', 'a', 'l', 'I', 't', 'e', 'm', 's')
    RAPIDJSON_STRING_(UniqueItems, 'u', 'n', 'i', 'q', 'u', 'e', 'I', 't', 'e', 'm', 's')
    RAPIDJSON_STRING_(MinLength, 'm', 'i', 'n', 'L', 'e', 'n', 'g', 't', 'h')
    RAPIDJSON_STRING_(MaxLength, 'm', 'a', 'x', 'L', 'e', 'n', 'g', 't', 'h')
    RAPIDJSON_STRING_(Pattern, 'p', 'a', 't', 't', 'e', 'r', 'n')
    RAPIDJSON_STRING_(Minimum, 'm', 'i', 'n', 'i', 'm', 'u', 'm')
    RAPIDJSON_STRING_(Maximum, 'm', 'a', 'x', 'i', 'm', 'u', 'm')
    RAPIDJSON_STRING_(ExclusiveMinimum, 'e', 'x', 'c', 'l', 'u', 's', 'i', 'v', 'e', 'M', 'i', 'n', 'i', 'm', 'u', 'm')
    RAPIDJSON_STRING_(ExclusiveMaximum, 'e', 'x', 'c', 'l', 'u', 's', 'i', 'v', 'e', 'M', 'a', 'x', 'i', 'm', 'u', 'm')
    RAPIDJSON_STRING_(MultipleOf, 'm', 'u', 'l', 't', 'i', 'p', 'l', 'e', 'O', 'f')

#undef RAPIDJSON_STRING_

private:
    enum SchemaValueType {
        kNullSchemaType,
        kBooleanSchemaType,
        kObjectSchemaType,
        kArraySchemaType,
        kStringSchemaType,
        kNumberSchemaType,
        kIntegerSchemaType,
        kTotalSchemaType
    };

#if RAPIDJSON_SCHEMA_USE_INTERNALREGEX
        typedef internal::GenericRegex<EncodingType, AllocatorType> RegexType;
#elif RAPIDJSON_SCHEMA_USE_STDREGEX
        typedef std::basic_regex<Ch> RegexType;
#else
        typedef char RegexType;
#endif

    struct SchemaArray {
        SchemaArray() : schemas(), count() {}
        ~SchemaArray() { AllocatorType::Free(schemas); }
        const SchemaType** schemas;
        SizeType begin; // begin index of context.validators
        SizeType count;
    };

    template <typename V1, typename V2>
    void AddUniqueElement(V1& a, const V2& v) {
        for (typename V1::ConstValueIterator itr = a.Begin(); itr != a.End(); ++itr)
            if (*itr == v)
                return;
        V1 c(v, *allocator_);
        a.PushBack(c, *allocator_);
    }

    static const ValueType* GetMember(const ValueType& value, const ValueType& name) {
        typename ValueType::ConstMemberIterator itr = value.FindMember(name);
        return itr != value.MemberEnd() ? &(itr->value) : 0;
    }

    static void AssignIfExist(bool& out, const ValueType& value, const ValueType& name) {
        if (const ValueType* v = GetMember(value, name))
            if (v->IsBool())
                out = v->GetBool();
    }

    static void AssignIfExist(SizeType& out, const ValueType& value, const ValueType& name) {
        if (const ValueType* v = GetMember(value, name))
            if (v->IsUint64() && v->GetUint64() <= SizeType(~0))
                out = static_cast<SizeType>(v->GetUint64());
    }

    void AssignIfExist(SchemaArray& out, SchemaDolwmentType& schemaDolwment, const PointerType& p, const ValueType& value, const ValueType& name, const ValueType& document) {
        if (const ValueType* v = GetMember(value, name)) {
            if (v->IsArray() && v->Size() > 0) {
                PointerType q = p.Append(name, allocator_);
                out.count = v->Size();
                out.schemas = static_cast<const Schema**>(allocator_->Malloc(out.count * sizeof(const Schema*)));
                memset(out.schemas, 0, sizeof(Schema*)* out.count);
                for (SizeType i = 0; i < out.count; i++)
                    schemaDolwment.CreateSchema(&out.schemas[i], q.Append(i, allocator_), (*v)[i], document);
                out.begin = validatorCount_;
                validatorCount_ += out.count;
            }
        }
    }

#if RAPIDJSON_SCHEMA_USE_INTERNALREGEX
    template <typename ValueType>
    RegexType* CreatePattern(const ValueType& value) {
        if (value.IsString()) {
            RegexType* r = new (allocator_->Malloc(sizeof(RegexType))) RegexType(value.GetString(), allocator_);
            if (!r->IsValid()) {
                r->~RegexType();
                AllocatorType::Free(r);
                r = 0;
            }
            return r;
        }
        return 0;
    }

    static bool IsPatternMatch(const RegexType* pattern, const Ch *str, SizeType) {
        GenericRegexSearch<RegexType> rs(*pattern);
        return rs.Search(str);
    }
#elif RAPIDJSON_SCHEMA_USE_STDREGEX
    template <typename ValueType>
    RegexType* CreatePattern(const ValueType& value) {
        if (value.IsString())
            try {
                return new (allocator_->Malloc(sizeof(RegexType))) RegexType(value.GetString(), std::size_t(value.GetStringLength()), std::regex_constants::ECMAScript);
            }
            catch (const std::regex_error&) {
            }
        return 0;
    }

    static bool IsPatternMatch(const RegexType* pattern, const Ch *str, SizeType length) {
        std::match_results<const Ch*> r;
        return std::regex_search(str, str + length, r, *pattern);
    }
#else
    template <typename ValueType>
    RegexType* CreatePattern(const ValueType&) { return 0; }

    static bool IsPatternMatch(const RegexType*, const Ch *, SizeType) { return true; }
#endif // RAPIDJSON_SCHEMA_USE_STDREGEX

    void AddType(const ValueType& type) {
        if      (type == GetNullString()   ) type_ |= 1 << kNullSchemaType;
        else if (type == GetBooleanString()) type_ |= 1 << kBooleanSchemaType;
        else if (type == GetObjectString() ) type_ |= 1 << kObjectSchemaType;
        else if (type == GetArrayString()  ) type_ |= 1 << kArraySchemaType;
        else if (type == GetStringString() ) type_ |= 1 << kStringSchemaType;
        else if (type == GetIntegerString()) type_ |= 1 << kIntegerSchemaType;
        else if (type == GetNumberString() ) type_ |= (1 << kNumberSchemaType) | (1 << kIntegerSchemaType);
    }

    bool CreateParallelValidator(Context& context) const {
        if (enum_ || context.arrayUniqueness)
            context.hasher = context.factory.CreateHasher();

        if (validatorCount_) {
            RAPIDJSON_ASSERT(context.validators == 0);
            context.validators = static_cast<ISchemaValidator**>(context.factory.MallocState(sizeof(ISchemaValidator*) * validatorCount_));
            context.validatorCount = validatorCount_;

            if (allOf_.schemas)
                CreateSchemaValidators(context, allOf_);

            if (anyOf_.schemas)
                CreateSchemaValidators(context, anyOf_);
            
            if (oneOf_.schemas)
                CreateSchemaValidators(context, oneOf_);
            
            if (not_)
                context.validators[notValidatorIndex_] = context.factory.CreateSchemaValidator(*not_);
            
            if (hasSchemaDependencies_) {
                for (SizeType i = 0; i < propertyCount_; i++)
                    if (properties_[i].dependenciesSchema)
                        context.validators[properties_[i].dependenciesValidatorIndex] = context.factory.CreateSchemaValidator(*properties_[i].dependenciesSchema);
            }
        }

        return true;
    }

    void CreateSchemaValidators(Context& context, const SchemaArray& schemas) const {
        for (SizeType i = 0; i < schemas.count; i++)
            context.validators[schemas.begin + i] = context.factory.CreateSchemaValidator(*schemas.schemas[i]);
    }

    // O(n)
    bool FindPropertyIndex(const ValueType& name, SizeType* outIndex) const {
        SizeType len = name.GetStringLength();
        const Ch* str = name.GetString();
        for (SizeType index = 0; index < propertyCount_; index++)
            if (properties_[index].name.GetStringLength() == len && 
                (std::memcmp(properties_[index].name.GetString(), str, sizeof(Ch) * len) == 0))
            {
                *outIndex = index;
                return true;
            }
        return false;
    }

    bool CheckInt(Context& context, int64_t i) const {
        if (!(type_ & ((1 << kIntegerSchemaType) | (1 << kNumberSchemaType))))
            RAPIDJSON_ILWALID_KEYWORD_RETURN(GetTypeString());

        if (!minimum_.IsNull()) {
            if (minimum_.IsInt64()) {
                if (exclusiveMinimum_ ? i <= minimum_.GetInt64() : i < minimum_.GetInt64())
                    RAPIDJSON_ILWALID_KEYWORD_RETURN(GetMinimumString());
            }
            else if (minimum_.IsUint64()) {
                RAPIDJSON_ILWALID_KEYWORD_RETURN(GetMinimumString()); // i <= max(int64_t) < minimum.GetUint64()
            }
            else if (!CheckDoubleMinimum(context, static_cast<double>(i)))
                return false;
        }

        if (!maximum_.IsNull()) {
            if (maximum_.IsInt64()) {
                if (exclusiveMaximum_ ? i >= maximum_.GetInt64() : i > maximum_.GetInt64())
                    RAPIDJSON_ILWALID_KEYWORD_RETURN(GetMaximumString());
            }
            else if (maximum_.IsUint64()) { }
                /* do nothing */ // i <= max(int64_t) < maximum_.GetUint64()
            else if (!CheckDoubleMaximum(context, static_cast<double>(i)))
                return false;
        }

        if (!multipleOf_.IsNull()) {
            if (multipleOf_.IsUint64()) {
                if (static_cast<uint64_t>(i >= 0 ? i : -i) % multipleOf_.GetUint64() != 0)
                    RAPIDJSON_ILWALID_KEYWORD_RETURN(GetMultipleOfString());
            }
            else if (!CheckDoubleMultipleOf(context, static_cast<double>(i)))
                return false;
        }

        return true;
    }

    bool CheckUint(Context& context, uint64_t i) const {
        if (!(type_ & ((1 << kIntegerSchemaType) | (1 << kNumberSchemaType))))
            RAPIDJSON_ILWALID_KEYWORD_RETURN(GetTypeString());

        if (!minimum_.IsNull()) {
            if (minimum_.IsUint64()) {
                if (exclusiveMinimum_ ? i <= minimum_.GetUint64() : i < minimum_.GetUint64())
                    RAPIDJSON_ILWALID_KEYWORD_RETURN(GetMinimumString());
            }
            else if (minimum_.IsInt64())
                /* do nothing */; // i >= 0 > minimum.Getint64()
            else if (!CheckDoubleMinimum(context, static_cast<double>(i)))
                return false;
        }

        if (!maximum_.IsNull()) {
            if (maximum_.IsUint64()) {
                if (exclusiveMaximum_ ? i >= maximum_.GetUint64() : i > maximum_.GetUint64())
                    RAPIDJSON_ILWALID_KEYWORD_RETURN(GetMaximumString());
            }
            else if (maximum_.IsInt64())
                RAPIDJSON_ILWALID_KEYWORD_RETURN(GetMaximumString()); // i >= 0 > maximum_
            else if (!CheckDoubleMaximum(context, static_cast<double>(i)))
                return false;
        }

        if (!multipleOf_.IsNull()) {
            if (multipleOf_.IsUint64()) {
                if (i % multipleOf_.GetUint64() != 0)
                    RAPIDJSON_ILWALID_KEYWORD_RETURN(GetMultipleOfString());
            }
            else if (!CheckDoubleMultipleOf(context, static_cast<double>(i)))
                return false;
        }

        return true;
    }

    bool CheckDoubleMinimum(Context& context, double d) const {
        if (exclusiveMinimum_ ? d <= minimum_.GetDouble() : d < minimum_.GetDouble())
            RAPIDJSON_ILWALID_KEYWORD_RETURN(GetMinimumString());
        return true;
    }

    bool CheckDoubleMaximum(Context& context, double d) const {
        if (exclusiveMaximum_ ? d >= maximum_.GetDouble() : d > maximum_.GetDouble())
            RAPIDJSON_ILWALID_KEYWORD_RETURN(GetMaximumString());
        return true;
    }

    bool CheckDoubleMultipleOf(Context& context, double d) const {
        double a = std::abs(d), b = std::abs(multipleOf_.GetDouble());
        double q = std::floor(a / b);
        double r = a - q * b;
        if (r > 0.0)
            RAPIDJSON_ILWALID_KEYWORD_RETURN(GetMultipleOfString());
        return true;
    }

    struct Property {
        Property() : schema(), dependenciesSchema(), dependenciesValidatorIndex(), dependencies(), required(false) {}
        ~Property() { AllocatorType::Free(dependencies); }
        SValue name;
        const SchemaType* schema;
        const SchemaType* dependenciesSchema;
        SizeType dependenciesValidatorIndex;
        bool* dependencies;
        bool required;
    };

    struct PatternProperty {
        PatternProperty() : schema(), pattern() {}
        ~PatternProperty() { 
            if (pattern) {
                pattern->~RegexType();
                AllocatorType::Free(pattern);
            }
        }
        const SchemaType* schema;
        RegexType* pattern;
    };

    AllocatorType* allocator_;
    const SchemaType* typeless_;
    uint64_t* enum_;
    SizeType enumCount_;
    SchemaArray allOf_;
    SchemaArray anyOf_;
    SchemaArray oneOf_;
    const SchemaType* not_;
    unsigned type_; // bitmask of kSchemaType
    SizeType validatorCount_;
    SizeType notValidatorIndex_;

    Property* properties_;
    const SchemaType* additionalPropertiesSchema_;
    PatternProperty* patternProperties_;
    SizeType patternPropertyCount_;
    SizeType propertyCount_;
    SizeType minProperties_;
    SizeType maxProperties_;
    bool additionalProperties_;
    bool hasDependencies_;
    bool hasRequired_;
    bool hasSchemaDependencies_;

    const SchemaType* additionalItemsSchema_;
    const SchemaType* itemsList_;
    const SchemaType** itemsTuple_;
    SizeType itemsTupleCount_;
    SizeType minItems_;
    SizeType maxItems_;
    bool additionalItems_;
    bool uniqueItems_;

    RegexType* pattern_;
    SizeType minLength_;
    SizeType maxLength_;

    SValue minimum_;
    SValue maximum_;
    SValue multipleOf_;
    bool exclusiveMinimum_;
    bool exclusiveMaximum_;
};

template<typename Stack, typename Ch>
struct TokenHelper {
    RAPIDJSON_FORCEINLINE static void AppendIndexToken(Stack& dolwmentStack, SizeType index) {
        *dolwmentStack.template Push<Ch>() = '/';
        char buffer[21];
        size_t length = static_cast<size_t>((sizeof(SizeType) == 4 ? u32toa(index, buffer) : u64toa(index, buffer)) - buffer);
        for (size_t i = 0; i < length; i++)
            *dolwmentStack.template Push<Ch>() = static_cast<Ch>(buffer[i]);
    }
};

// Partial specialized version for char to prevent buffer copying.
template <typename Stack>
struct TokenHelper<Stack, char> {
    RAPIDJSON_FORCEINLINE static void AppendIndexToken(Stack& dolwmentStack, SizeType index) {
        if (sizeof(SizeType) == 4) {
            char *buffer = dolwmentStack.template Push<char>(1 + 10); // '/' + uint
            *buffer++ = '/';
            const char* end = internal::u32toa(index, buffer);
             dolwmentStack.template Pop<char>(static_cast<size_t>(10 - (end - buffer)));
        }
        else {
            char *buffer = dolwmentStack.template Push<char>(1 + 20); // '/' + uint64
            *buffer++ = '/';
            const char* end = internal::u64toa(index, buffer);
            dolwmentStack.template Pop<char>(static_cast<size_t>(20 - (end - buffer)));
        }
    }
};

} // namespace internal

///////////////////////////////////////////////////////////////////////////////
// IGenericRemoteSchemaDolwmentProvider

template <typename SchemaDolwmentType>
class IGenericRemoteSchemaDolwmentProvider {
public:
    typedef typename SchemaDolwmentType::Ch Ch;

    virtual ~IGenericRemoteSchemaDolwmentProvider() {}
    virtual const SchemaDolwmentType* GetRemoteDolwment(const Ch* uri, SizeType length) = 0;
};

///////////////////////////////////////////////////////////////////////////////
// GenericSchemaDolwment

//! JSON schema document.
/*!
    A JSON schema document is a compiled version of a JSON schema.
    It is basically a tree of internal::Schema.

    \note This is an immutable class (i.e. its instance cannot be modified after construction).
    \tparam ValueT Type of JSON value (e.g. \c Value ), which also determine the encoding.
    \tparam Allocator Allocator type for allocating memory of this document.
*/
template <typename ValueT, typename Allocator = CrtAllocator>
class GenericSchemaDolwment {
public:
    typedef ValueT ValueType;
    typedef IGenericRemoteSchemaDolwmentProvider<GenericSchemaDolwment> IRemoteSchemaDolwmentProviderType;
    typedef Allocator AllocatorType;
    typedef typename ValueType::EncodingType EncodingType;
    typedef typename EncodingType::Ch Ch;
    typedef internal::Schema<GenericSchemaDolwment> SchemaType;
    typedef GenericPointer<ValueType, Allocator> PointerType;
    friend class internal::Schema<GenericSchemaDolwment>;
    template <typename, typename, typename>
    friend class GenericSchemaValidator;

    //! Constructor.
    /*!
        Compile a JSON document into schema document.

        \param document A JSON document as source.
        \param remoteProvider An optional remote schema document provider for resolving remote reference. Can be null.
        \param allocator An optional allocator instance for allocating memory. Can be null.
    */
    explicit GenericSchemaDolwment(const ValueType& document, IRemoteSchemaDolwmentProviderType* remoteProvider = 0, Allocator* allocator = 0) :
        remoteProvider_(remoteProvider),
        allocator_(allocator),
        ownAllocator_(),
        root_(),
        typeless_(),
        schemaMap_(allocator, kInitialSchemaMapSize),
        schemaRef_(allocator, kInitialSchemaRefSize)
    {
        if (!allocator_)
            ownAllocator_ = allocator_ = RAPIDJSON_NEW(Allocator)();

        typeless_ = static_cast<SchemaType*>(allocator_->Malloc(sizeof(SchemaType)));
        new (typeless_) SchemaType(this, PointerType(), ValueType(kObjectType).Move(), ValueType(kObjectType).Move(), 0);

        // Generate root schema, it will call CreateSchema() to create sub-schemas,
        // And call AddRefSchema() if there are $ref.
        CreateSchemaRelwrsive(&root_, PointerType(), document, document);

        // Resolve $ref
        while (!schemaRef_.Empty()) {
            SchemaRefEntry* refEntry = schemaRef_.template Pop<SchemaRefEntry>(1);
            if (const SchemaType* s = GetSchema(refEntry->target)) {
                if (refEntry->schema)
                    *refEntry->schema = s;

                // Create entry in map if not exist
                if (!GetSchema(refEntry->source)) {
                    new (schemaMap_.template Push<SchemaEntry>()) SchemaEntry(refEntry->source, const_cast<SchemaType*>(s), false, allocator_);
                }
            }
            else if (refEntry->schema)
                *refEntry->schema = typeless_;

            refEntry->~SchemaRefEntry();
        }

        RAPIDJSON_ASSERT(root_ != 0);

        schemaRef_.ShrinkToFit(); // Deallocate all memory for ref
    }

#if RAPIDJSON_HAS_CXX11_RVALUE_REFS
    //! Move constructor in C++11
    GenericSchemaDolwment(GenericSchemaDolwment&& rhs) RAPIDJSON_NOEXCEPT :
        remoteProvider_(rhs.remoteProvider_),
        allocator_(rhs.allocator_),
        ownAllocator_(rhs.ownAllocator_),
        root_(rhs.root_),
        typeless_(rhs.typeless_),
        schemaMap_(std::move(rhs.schemaMap_)),
        schemaRef_(std::move(rhs.schemaRef_))
    {
        rhs.remoteProvider_ = 0;
        rhs.allocator_ = 0;
        rhs.ownAllocator_ = 0;
        rhs.typeless_ = 0;
    }
#endif

    //! Destructor
    ~GenericSchemaDolwment() {
        while (!schemaMap_.Empty())
            schemaMap_.template Pop<SchemaEntry>(1)->~SchemaEntry();

        if (typeless_) {
            typeless_->~SchemaType();
            Allocator::Free(typeless_);
        }

        RAPIDJSON_DELETE(ownAllocator_);
    }

    //! Get the root schema.
    const SchemaType& GetRoot() const { return *root_; }

private:
    //! Prohibit copying
    GenericSchemaDolwment(const GenericSchemaDolwment&);
    //! Prohibit assignment
    GenericSchemaDolwment& operator=(const GenericSchemaDolwment&);

    struct SchemaRefEntry {
        SchemaRefEntry(const PointerType& s, const PointerType& t, const SchemaType** outSchema, Allocator *allocator) : source(s, allocator), target(t, allocator), schema(outSchema) {}
        PointerType source;
        PointerType target;
        const SchemaType** schema;
    };

    struct SchemaEntry {
        SchemaEntry(const PointerType& p, SchemaType* s, bool o, Allocator* allocator) : pointer(p, allocator), schema(s), owned(o) {}
        ~SchemaEntry() {
            if (owned) {
                schema->~SchemaType();
                Allocator::Free(schema);
            }
        }
        PointerType pointer;
        SchemaType* schema;
        bool owned;
    };

    void CreateSchemaRelwrsive(const SchemaType** schema, const PointerType& pointer, const ValueType& v, const ValueType& document) {
        if (schema)
            *schema = typeless_;

        if (v.GetType() == kObjectType) {
            const SchemaType* s = GetSchema(pointer);
            if (!s)
                CreateSchema(schema, pointer, v, document);

            for (typename ValueType::ConstMemberIterator itr = v.MemberBegin(); itr != v.MemberEnd(); ++itr)
                CreateSchemaRelwrsive(0, pointer.Append(itr->name, allocator_), itr->value, document);
        }
        else if (v.GetType() == kArrayType)
            for (SizeType i = 0; i < v.Size(); i++)
                CreateSchemaRelwrsive(0, pointer.Append(i, allocator_), v[i], document);
    }

    void CreateSchema(const SchemaType** schema, const PointerType& pointer, const ValueType& v, const ValueType& document) {
        RAPIDJSON_ASSERT(pointer.IsValid());
        if (v.IsObject()) {
            if (!HandleRefSchema(pointer, schema, v, document)) {
                SchemaType* s = new (allocator_->Malloc(sizeof(SchemaType))) SchemaType(this, pointer, v, document, allocator_);
                new (schemaMap_.template Push<SchemaEntry>()) SchemaEntry(pointer, s, true, allocator_);
                if (schema)
                    *schema = s;
            }
        }
    }

    bool HandleRefSchema(const PointerType& source, const SchemaType** schema, const ValueType& v, const ValueType& document) {
        static const Ch kRefString[] = { '$', 'r', 'e', 'f', '\0' };
        static const ValueType kRefValue(kRefString, 4);

        typename ValueType::ConstMemberIterator itr = v.FindMember(kRefValue);
        if (itr == v.MemberEnd())
            return false;

        if (itr->value.IsString()) {
            SizeType len = itr->value.GetStringLength();
            if (len > 0) {
                const Ch* s = itr->value.GetString();
                SizeType i = 0;
                while (i < len && s[i] != '#') // Find the first #
                    i++;

                if (i > 0) { // Remote reference, resolve immediately
                    if (remoteProvider_) {
                        if (const GenericSchemaDolwment* remoteDolwment = remoteProvider_->GetRemoteDolwment(s, i)) {
                            PointerType pointer(&s[i], len - i, allocator_);
                            if (pointer.IsValid()) {
                                if (const SchemaType* sc = remoteDolwment->GetSchema(pointer)) {
                                    if (schema)
                                        *schema = sc;
                                    return true;
                                }
                            }
                        }
                    }
                }
                else if (s[i] == '#') { // Local reference, defer resolution
                    PointerType pointer(&s[i], len - i, allocator_);
                    if (pointer.IsValid()) {
                        if (const ValueType* lw = pointer.Get(document))
                            if (HandleRefSchema(source, schema, *lw, document))
                                return true;

                        new (schemaRef_.template Push<SchemaRefEntry>()) SchemaRefEntry(source, pointer, schema, allocator_);
                        return true;
                    }
                }
            }
        }
        return false;
    }

    const SchemaType* GetSchema(const PointerType& pointer) const {
        for (const SchemaEntry* target = schemaMap_.template Bottom<SchemaEntry>(); target != schemaMap_.template End<SchemaEntry>(); ++target)
            if (pointer == target->pointer)
                return target->schema;
        return 0;
    }

    PointerType GetPointer(const SchemaType* schema) const {
        for (const SchemaEntry* target = schemaMap_.template Bottom<SchemaEntry>(); target != schemaMap_.template End<SchemaEntry>(); ++target)
            if (schema == target->schema)
                return target->pointer;
        return PointerType();
    }

    const SchemaType* GetTypeless() const { return typeless_; }

    static const size_t kInitialSchemaMapSize = 64;
    static const size_t kInitialSchemaRefSize = 64;

    IRemoteSchemaDolwmentProviderType* remoteProvider_;
    Allocator *allocator_;
    Allocator *ownAllocator_;
    const SchemaType* root_;                //!< Root schema.
    SchemaType* typeless_;
    internal::Stack<Allocator> schemaMap_;  // Stores created Pointer -> Schemas
    internal::Stack<Allocator> schemaRef_;  // Stores Pointer from $ref and schema which holds the $ref
};

//! GenericSchemaDolwment using Value type.
typedef GenericSchemaDolwment<Value> SchemaDolwment;
//! IGenericRemoteSchemaDolwmentProvider using SchemaDolwment.
typedef IGenericRemoteSchemaDolwmentProvider<SchemaDolwment> IRemoteSchemaDolwmentProvider;

///////////////////////////////////////////////////////////////////////////////
// GenericSchemaValidator

//! JSON Schema Validator.
/*!
    A SAX style JSON schema validator.
    It uses a \c GenericSchemaDolwment to validate SAX events.
    It delegates the incoming SAX events to an output handler.
    The default output handler does nothing.
    It can be reused multiple times by calling \c Reset().

    \tparam SchemaDolwmentType Type of schema document.
    \tparam OutputHandler Type of output handler. Default handler does nothing.
    \tparam StateAllocator Allocator for storing the internal validation states.
*/
template <
    typename SchemaDolwmentType,
    typename OutputHandler = BaseReaderHandler<typename SchemaDolwmentType::SchemaType::EncodingType>,
    typename StateAllocator = CrtAllocator>
class GenericSchemaValidator :
    public internal::ISchemaStateFactory<typename SchemaDolwmentType::SchemaType>, 
    public internal::ISchemaValidator
{
public:
    typedef typename SchemaDolwmentType::SchemaType SchemaType;
    typedef typename SchemaDolwmentType::PointerType PointerType;
    typedef typename SchemaType::EncodingType EncodingType;
    typedef typename EncodingType::Ch Ch;

    //! Constructor without output handler.
    /*!
        \param schemaDolwment The schema document to conform to.
        \param allocator Optional allocator for storing internal validation states.
        \param schemaStackCapacity Optional initial capacity of schema path stack.
        \param dolwmentStackCapacity Optional initial capacity of document path stack.
    */
    GenericSchemaValidator(
        const SchemaDolwmentType& schemaDolwment,
        StateAllocator* allocator = 0, 
        size_t schemaStackCapacity = kDefaultSchemaStackCapacity,
        size_t dolwmentStackCapacity = kDefaultDolwmentStackCapacity)
        :
        schemaDolwment_(&schemaDolwment),
        root_(schemaDolwment.GetRoot()),
        stateAllocator_(allocator),
        ownStateAllocator_(0),
        schemaStack_(allocator, schemaStackCapacity),
        dolwmentStack_(allocator, dolwmentStackCapacity),
        outputHandler_(0),
        valid_(true)
#if RAPIDJSON_SCHEMA_VERBOSE
        , depth_(0)
#endif
    {
    }

    //! Constructor with output handler.
    /*!
        \param schemaDolwment The schema document to conform to.
        \param allocator Optional allocator for storing internal validation states.
        \param schemaStackCapacity Optional initial capacity of schema path stack.
        \param dolwmentStackCapacity Optional initial capacity of document path stack.
    */
    GenericSchemaValidator(
        const SchemaDolwmentType& schemaDolwment,
        OutputHandler& outputHandler,
        StateAllocator* allocator = 0, 
        size_t schemaStackCapacity = kDefaultSchemaStackCapacity,
        size_t dolwmentStackCapacity = kDefaultDolwmentStackCapacity)
        :
        schemaDolwment_(&schemaDolwment),
        root_(schemaDolwment.GetRoot()),
        stateAllocator_(allocator),
        ownStateAllocator_(0),
        schemaStack_(allocator, schemaStackCapacity),
        dolwmentStack_(allocator, dolwmentStackCapacity),
        outputHandler_(&outputHandler),
        valid_(true)
#if RAPIDJSON_SCHEMA_VERBOSE
        , depth_(0)
#endif
    {
    }

    //! Destructor.
    ~GenericSchemaValidator() {
        Reset();
        RAPIDJSON_DELETE(ownStateAllocator_);
    }

    //! Reset the internal states.
    void Reset() {
        while (!schemaStack_.Empty())
            PopSchema();
        dolwmentStack_.Clear();
        valid_ = true;
    }

    //! Checks whether the current state is valid.
    // Implementation of ISchemaValidator
    virtual bool IsValid() const { return valid_; }

    //! Gets the JSON pointer pointed to the invalid schema.
    PointerType GetIlwalidSchemaPointer() const {
        return schemaStack_.Empty() ? PointerType() : schemaDolwment_->GetPointer(&LwrrentSchema());
    }

    //! Gets the keyword of invalid schema.
    const Ch* GetIlwalidSchemaKeyword() const {
        return schemaStack_.Empty() ? 0 : LwrrentContext().ilwalidKeyword;
    }

    //! Gets the JSON pointer pointed to the invalid value.
    PointerType GetIlwalidDolwmentPointer() const {
        return dolwmentStack_.Empty() ? PointerType() : PointerType(dolwmentStack_.template Bottom<Ch>(), dolwmentStack_.GetSize() / sizeof(Ch));
    }

#if RAPIDJSON_SCHEMA_VERBOSE
#define RAPIDJSON_SCHEMA_HANDLE_BEGIN_VERBOSE_() \
RAPIDJSON_MULTILINEMACRO_BEGIN\
    *dolwmentStack_.template Push<Ch>() = '\0';\
    dolwmentStack_.template Pop<Ch>(1);\
    internal::PrintIlwalidDolwment(dolwmentStack_.template Bottom<Ch>());\
RAPIDJSON_MULTILINEMACRO_END
#else
#define RAPIDJSON_SCHEMA_HANDLE_BEGIN_VERBOSE_()
#endif

#define RAPIDJSON_SCHEMA_HANDLE_BEGIN_(method, arg1)\
    if (!valid_) return false; \
    if (!Begilwalue() || !LwrrentSchema().method arg1) {\
        RAPIDJSON_SCHEMA_HANDLE_BEGIN_VERBOSE_();\
        return valid_ = false;\
    }

#define RAPIDJSON_SCHEMA_HANDLE_PARALLEL_(method, arg2)\
    for (Context* context = schemaStack_.template Bottom<Context>(); context != schemaStack_.template End<Context>(); context++) {\
        if (context->hasher)\
            static_cast<HasherType*>(context->hasher)->method arg2;\
        if (context->validators)\
            for (SizeType i_ = 0; i_ < context->validatorCount; i_++)\
                static_cast<GenericSchemaValidator*>(context->validators[i_])->method arg2;\
        if (context->patternPropertiesValidators)\
            for (SizeType i_ = 0; i_ < context->patternPropertiesValidatorCount; i_++)\
                static_cast<GenericSchemaValidator*>(context->patternPropertiesValidators[i_])->method arg2;\
    }

#define RAPIDJSON_SCHEMA_HANDLE_END_(method, arg2)\
    return valid_ = EndValue() && (!outputHandler_ || outputHandler_->method arg2)

#define RAPIDJSON_SCHEMA_HANDLE_VALUE_(method, arg1, arg2) \
    RAPIDJSON_SCHEMA_HANDLE_BEGIN_   (method, arg1);\
    RAPIDJSON_SCHEMA_HANDLE_PARALLEL_(method, arg2);\
    RAPIDJSON_SCHEMA_HANDLE_END_     (method, arg2)

    bool Null()             { RAPIDJSON_SCHEMA_HANDLE_VALUE_(Null,   (LwrrentContext()   ), ( )); }
    bool Bool(bool b)       { RAPIDJSON_SCHEMA_HANDLE_VALUE_(Bool,   (LwrrentContext(), b), (b)); }
    bool Int(int i)         { RAPIDJSON_SCHEMA_HANDLE_VALUE_(Int,    (LwrrentContext(), i), (i)); }
    bool Uint(unsigned u)   { RAPIDJSON_SCHEMA_HANDLE_VALUE_(Uint,   (LwrrentContext(), u), (u)); }
    bool Int64(int64_t i)   { RAPIDJSON_SCHEMA_HANDLE_VALUE_(Int64,  (LwrrentContext(), i), (i)); }
    bool Uint64(uint64_t u) { RAPIDJSON_SCHEMA_HANDLE_VALUE_(Uint64, (LwrrentContext(), u), (u)); }
    bool Double(double d)   { RAPIDJSON_SCHEMA_HANDLE_VALUE_(Double, (LwrrentContext(), d), (d)); }
    bool RawNumber(const Ch* str, SizeType length, bool copy)
                                    { RAPIDJSON_SCHEMA_HANDLE_VALUE_(String, (LwrrentContext(), str, length, copy), (str, length, copy)); }
    bool String(const Ch* str, SizeType length, bool copy)
                                    { RAPIDJSON_SCHEMA_HANDLE_VALUE_(String, (LwrrentContext(), str, length, copy), (str, length, copy)); }

    bool StartObject() {
        RAPIDJSON_SCHEMA_HANDLE_BEGIN_(StartObject, (LwrrentContext()));
        RAPIDJSON_SCHEMA_HANDLE_PARALLEL_(StartObject, ());
        return valid_ = !outputHandler_ || outputHandler_->StartObject();
    }
    
    bool Key(const Ch* str, SizeType len, bool copy) {
        if (!valid_) return false;
        AppendToken(str, len);
        if (!LwrrentSchema().Key(LwrrentContext(), str, len, copy)) return valid_ = false;
        RAPIDJSON_SCHEMA_HANDLE_PARALLEL_(Key, (str, len, copy));
        return valid_ = !outputHandler_ || outputHandler_->Key(str, len, copy);
    }
    
    bool EndObject(SizeType memberCount) { 
        if (!valid_) return false;
        RAPIDJSON_SCHEMA_HANDLE_PARALLEL_(EndObject, (memberCount));
        if (!LwrrentSchema().EndObject(LwrrentContext(), memberCount)) return valid_ = false;
        RAPIDJSON_SCHEMA_HANDLE_END_(EndObject, (memberCount));
    }

    bool StartArray() {
        RAPIDJSON_SCHEMA_HANDLE_BEGIN_(StartArray, (LwrrentContext()));
        RAPIDJSON_SCHEMA_HANDLE_PARALLEL_(StartArray, ());
        return valid_ = !outputHandler_ || outputHandler_->StartArray();
    }
    
    bool EndArray(SizeType elementCount) {
        if (!valid_) return false;
        RAPIDJSON_SCHEMA_HANDLE_PARALLEL_(EndArray, (elementCount));
        if (!LwrrentSchema().EndArray(LwrrentContext(), elementCount)) return valid_ = false;
        RAPIDJSON_SCHEMA_HANDLE_END_(EndArray, (elementCount));
    }

#undef RAPIDJSON_SCHEMA_HANDLE_BEGIN_VERBOSE_
#undef RAPIDJSON_SCHEMA_HANDLE_BEGIN_
#undef RAPIDJSON_SCHEMA_HANDLE_PARALLEL_
#undef RAPIDJSON_SCHEMA_HANDLE_VALUE_

    // Implementation of ISchemaStateFactory<SchemaType>
    virtual ISchemaValidator* CreateSchemaValidator(const SchemaType& root) {
        return new (GetStateAllocator().Malloc(sizeof(GenericSchemaValidator))) GenericSchemaValidator(*schemaDolwment_, root,
#if RAPIDJSON_SCHEMA_VERBOSE
        depth_ + 1,
#endif
        &GetStateAllocator());
    }

    virtual void DestroySchemaValidator(ISchemaValidator* validator) {
        GenericSchemaValidator* v = static_cast<GenericSchemaValidator*>(validator);
        v->~GenericSchemaValidator();
        StateAllocator::Free(v);
    }

    virtual void* CreateHasher() {
        return new (GetStateAllocator().Malloc(sizeof(HasherType))) HasherType(&GetStateAllocator());
    }

    virtual uint64_t GetHashCode(void* hasher) {
        return static_cast<HasherType*>(hasher)->GetHashCode();
    }

    virtual void DestroryHasher(void* hasher) {
        HasherType* h = static_cast<HasherType*>(hasher);
        h->~HasherType();
        StateAllocator::Free(h);
    }

    virtual void* MallocState(size_t size) {
        return GetStateAllocator().Malloc(size);
    }

    virtual void FreeState(void* p) {
        StateAllocator::Free(p);
    }

private:
    typedef typename SchemaType::Context Context;
    typedef GenericValue<UTF8<>, StateAllocator> HashCodeArray;
    typedef internal::Hasher<EncodingType, StateAllocator> HasherType;

    GenericSchemaValidator( 
        const SchemaDolwmentType& schemaDolwment,
        const SchemaType& root,
#if RAPIDJSON_SCHEMA_VERBOSE
        unsigned depth,
#endif
        StateAllocator* allocator = 0,
        size_t schemaStackCapacity = kDefaultSchemaStackCapacity,
        size_t dolwmentStackCapacity = kDefaultDolwmentStackCapacity)
        :
        schemaDolwment_(&schemaDolwment),
        root_(root),
        stateAllocator_(allocator),
        ownStateAllocator_(0),
        schemaStack_(allocator, schemaStackCapacity),
        dolwmentStack_(allocator, dolwmentStackCapacity),
        outputHandler_(0),
        valid_(true)
#if RAPIDJSON_SCHEMA_VERBOSE
        , depth_(depth)
#endif
    {
    }

    StateAllocator& GetStateAllocator() {
        if (!stateAllocator_)
            stateAllocator_ = ownStateAllocator_ = RAPIDJSON_NEW(StateAllocator)();
        return *stateAllocator_;
    }

    bool Begilwalue() {
        if (schemaStack_.Empty())
            PushSchema(root_);
        else {
            if (LwrrentContext().inArray)
                internal::TokenHelper<internal::Stack<StateAllocator>, Ch>::AppendIndexToken(dolwmentStack_, LwrrentContext().arrayElementIndex);

            if (!LwrrentSchema().Begilwalue(LwrrentContext()))
                return false;

            SizeType count = LwrrentContext().patternPropertiesSchemaCount;
            const SchemaType** sa = LwrrentContext().patternPropertiesSchemas;
            typename Context::PatterlwalidatorType patterlwalidatorType = LwrrentContext().valuePatterlwalidatorType;
            bool valueUniqueness = LwrrentContext().valueUniqueness;
            RAPIDJSON_ASSERT(LwrrentContext().valueSchema);
            PushSchema(*LwrrentContext().valueSchema);

            if (count > 0) {
                LwrrentContext().objectPatterlwalidatorType = patterlwalidatorType;
                ISchemaValidator**& va = LwrrentContext().patternPropertiesValidators;
                SizeType& validatorCount = LwrrentContext().patternPropertiesValidatorCount;
                va = static_cast<ISchemaValidator**>(MallocState(sizeof(ISchemaValidator*) * count));
                for (SizeType i = 0; i < count; i++)
                    va[validatorCount++] = CreateSchemaValidator(*sa[i]);
            }

            LwrrentContext().arrayUniqueness = valueUniqueness;
        }
        return true;
    }

    bool EndValue() {
        if (!LwrrentSchema().EndValue(LwrrentContext()))
            return false;

#if RAPIDJSON_SCHEMA_VERBOSE
        GenericStringBuffer<EncodingType> sb;
        schemaDolwment_->GetPointer(&LwrrentSchema()).Stringify(sb);

        *dolwmentStack_.template Push<Ch>() = '\0';
        dolwmentStack_.template Pop<Ch>(1);
        internal::PrintValidatorPointers(depth_, sb.GetString(), dolwmentStack_.template Bottom<Ch>());
#endif

        uint64_t h = LwrrentContext().arrayUniqueness ? static_cast<HasherType*>(LwrrentContext().hasher)->GetHashCode() : 0;
        
        PopSchema();

        if (!schemaStack_.Empty()) {
            Context& context = LwrrentContext();
            if (context.valueUniqueness) {
                HashCodeArray* a = static_cast<HashCodeArray*>(context.arrayElementHashCodes);
                if (!a)
                    LwrrentContext().arrayElementHashCodes = a = new (GetStateAllocator().Malloc(sizeof(HashCodeArray))) HashCodeArray(kArrayType);
                for (typename HashCodeArray::ConstValueIterator itr = a->Begin(); itr != a->End(); ++itr)
                    if (itr->GetUint64() == h)
                        RAPIDJSON_ILWALID_KEYWORD_RETURN(SchemaType::GetUniqueItemsString());
                a->PushBack(h, GetStateAllocator());
            }
        }

        // Remove the last token of document pointer
        while (!dolwmentStack_.Empty() && *dolwmentStack_.template Pop<Ch>(1) != '/')
            ;

        return true;
    }

    void AppendToken(const Ch* str, SizeType len) {
        dolwmentStack_.template Reserve<Ch>(1 + len * 2); // worst case all characters are escaped as two characters
        *dolwmentStack_.template PushUnsafe<Ch>() = '/';
        for (SizeType i = 0; i < len; i++) {
            if (str[i] == '~') {
                *dolwmentStack_.template PushUnsafe<Ch>() = '~';
                *dolwmentStack_.template PushUnsafe<Ch>() = '0';
            }
            else if (str[i] == '/') {
                *dolwmentStack_.template PushUnsafe<Ch>() = '~';
                *dolwmentStack_.template PushUnsafe<Ch>() = '1';
            }
            else
                *dolwmentStack_.template PushUnsafe<Ch>() = str[i];
        }
    }

    RAPIDJSON_FORCEINLINE void PushSchema(const SchemaType& schema) { new (schemaStack_.template Push<Context>()) Context(*this, &schema); }
    
    RAPIDJSON_FORCEINLINE void PopSchema() {
        Context* c = schemaStack_.template Pop<Context>(1);
        if (HashCodeArray* a = static_cast<HashCodeArray*>(c->arrayElementHashCodes)) {
            a->~HashCodeArray();
            StateAllocator::Free(a);
        }
        c->~Context();
    }

    const SchemaType& LwrrentSchema() const { return *schemaStack_.template Top<Context>()->schema; }
    Context& LwrrentContext() { return *schemaStack_.template Top<Context>(); }
    const Context& LwrrentContext() const { return *schemaStack_.template Top<Context>(); }

    static const size_t kDefaultSchemaStackCapacity = 1024;
    static const size_t kDefaultDolwmentStackCapacity = 256;
    const SchemaDolwmentType* schemaDolwment_;
    const SchemaType& root_;
    StateAllocator* stateAllocator_;
    StateAllocator* ownStateAllocator_;
    internal::Stack<StateAllocator> schemaStack_;    //!< stack to store the current path of schema (BaseSchemaType *)
    internal::Stack<StateAllocator> dolwmentStack_;  //!< stack to store the current path of validating document (Ch)
    OutputHandler* outputHandler_;
    bool valid_;
#if RAPIDJSON_SCHEMA_VERBOSE
    unsigned depth_;
#endif
};

typedef GenericSchemaValidator<SchemaDolwment> SchemaValidator;

///////////////////////////////////////////////////////////////////////////////
// SchemaValidatingReader

//! A helper class for parsing with validation.
/*!
    This helper class is a functor, designed as a parameter of \ref GenericDolwment::Populate().

    \tparam parseFlags Combination of \ref ParseFlag.
    \tparam InputStream Type of input stream, implementing Stream concept.
    \tparam SourceEncoding Encoding of the input stream.
    \tparam SchemaDolwmentType Type of schema document.
    \tparam StackAllocator Allocator type for stack.
*/
template <
    unsigned parseFlags,
    typename InputStream,
    typename SourceEncoding,
    typename SchemaDolwmentType = SchemaDolwment,
    typename StackAllocator = CrtAllocator>
class SchemaValidatingReader {
public:
    typedef typename SchemaDolwmentType::PointerType PointerType;
    typedef typename InputStream::Ch Ch;

    //! Constructor
    /*!
        \param is Input stream.
        \param sd Schema document.
    */
    SchemaValidatingReader(InputStream& is, const SchemaDolwmentType& sd) : is_(is), sd_(sd), ilwalidSchemaKeyword_(), isValid_(true) {}

    template <typename Handler>
    bool operator()(Handler& handler) {
        GenericReader<SourceEncoding, typename SchemaDolwmentType::EncodingType, StackAllocator> reader;
        GenericSchemaValidator<SchemaDolwmentType, Handler> validator(sd_, handler);
        parseResult_ = reader.template Parse<parseFlags>(is_, validator);

        isValid_ = validator.IsValid();
        if (isValid_) {
            ilwalidSchemaPointer_ = PointerType();
            ilwalidSchemaKeyword_ = 0;
            ilwalidDolwmentPointer_ = PointerType();
        }
        else {
            ilwalidSchemaPointer_ = validator.GetIlwalidSchemaPointer();
            ilwalidSchemaKeyword_ = validator.GetIlwalidSchemaKeyword();
            ilwalidDolwmentPointer_ = validator.GetIlwalidDolwmentPointer();
        }

        return parseResult_;
    }

    const ParseResult& GetParseResult() const { return parseResult_; }
    bool IsValid() const { return isValid_; }
    const PointerType& GetIlwalidSchemaPointer() const { return ilwalidSchemaPointer_; }
    const Ch* GetIlwalidSchemaKeyword() const { return ilwalidSchemaKeyword_; }
    const PointerType& GetIlwalidDolwmentPointer() const { return ilwalidDolwmentPointer_; }

private:
    InputStream& is_;
    const SchemaDolwmentType& sd_;

    ParseResult parseResult_;
    PointerType ilwalidSchemaPointer_;
    const Ch* ilwalidSchemaKeyword_;
    PointerType ilwalidDolwmentPointer_;
    bool isValid_;
};

RAPIDJSON_NAMESPACE_END
RAPIDJSON_DIAG_POP

#endif // RAPIDJSON_SCHEMA_H_
