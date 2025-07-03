#include "ir/TypeEnums.h"
#include "ir/TypelessVariable.h"
#include "ir/UserConstant.h"
#include "Log.h"

namespace shadermod
{
namespace ir
{
    void TypelessVariable::setDimensionality(unsigned int dims)
    {
        assert(dims <= MAX_GROUPED_VARIABLE_DIMENSION);
        m_dimensions = dims;
    }

    TypelessVariable TypelessVariable::makeZero(UserConstDataType t)
    {
        TypelessVariable ret;
        ret.m_type = t;

        switch (t)
        {
        case  UserConstDataType::kBool:
            ret.set(UserConstType<UserConstDataType::kBool>::getZero());
            break;
        case  UserConstDataType::kFloat:
            ret.set(UserConstType<UserConstDataType::kFloat>::getZero());
            break;
        case  UserConstDataType::kInt:
            ret.set(UserConstType<UserConstDataType::kInt>::getZero());
            break;
        case  UserConstDataType::kUInt:
            ret.set(UserConstType<UserConstDataType::kUInt>::getZero());
            break;
        }

        return ret;
    }

    TypelessVariable TypelessVariable::makeMin(UserConstDataType t)
    {
        TypelessVariable ret;
        ret.m_type = t;

        switch (t)
        {
        case  UserConstDataType::kBool:
            ret.set(UserConstType<UserConstDataType::kBool>::getMilwal());
            break;
        case  UserConstDataType::kFloat:
            ret.set(UserConstType<UserConstDataType::kFloat>::getMilwal());
            break;
        case  UserConstDataType::kInt:
            ret.set(UserConstType<UserConstDataType::kInt>::getMilwal());
            break;
        case  UserConstDataType::kUInt:
            ret.set(UserConstType<UserConstDataType::kUInt>::getMilwal());
            break;
        }

        return ret;
    }

    TypelessVariable TypelessVariable::makeMax(UserConstDataType t)
    {
        TypelessVariable ret;
        ret.m_type = t;

        switch (t)
        {
        case  UserConstDataType::kBool:
            ret.set(UserConstType<UserConstDataType::kBool>::getMaxVal());
            break;
        case  UserConstDataType::kFloat:
            ret.set(UserConstType<UserConstDataType::kFloat>::getMaxVal());
            break;
        case  UserConstDataType::kInt:
            ret.set(UserConstType<UserConstDataType::kInt>::getMaxVal());
            break;
        case  UserConstDataType::kUInt:
            ret.set(UserConstType<UserConstDataType::kUInt>::getMaxVal());
            break;
        }

        return ret;
    }

    TypelessVariable TypelessVariable::getClamped(const TypelessVariable& minimum, const TypelessVariable& maximum) const
    {
        TypelessVariable ret;

        if (minimum.m_type != m_type || maximum.m_type != m_type)
            return ret;

        // TODO avoroshilov: add dimensionality here
        ret.m_type = m_type;
        ret.m_dimensions = m_dimensions;

        assert(minimum.m_dimensions == m_dimensions);
        assert(maximum.m_dimensions == m_dimensions);
        for (unsigned int idx = 0; idx < m_dimensions; ++idx)
        {
            switch (m_type)
            {
            case UserConstDataType::kBool:
                ret.set(idx, m_boolValue[idx] > maximum.m_boolValue[idx] ? maximum.m_boolValue[idx] : (m_boolValue[idx] < minimum.m_boolValue[idx] ? minimum.m_boolValue[idx] : m_boolValue[idx]));
                break;
            case UserConstDataType::kFloat:
                ret.set(idx, m_floatValue[idx] > maximum.m_floatValue[idx] ? maximum.m_floatValue[idx] : (m_floatValue[idx] < minimum.m_floatValue[idx] ? minimum.m_floatValue[idx] : m_floatValue[idx]));
                break;
            case UserConstDataType::kInt:
                ret.set(idx, m_intValue[idx] > maximum.m_intValue[idx] ? maximum.m_intValue[idx] : (m_intValue[idx] < minimum.m_intValue[idx] ? minimum.m_intValue[idx] : m_intValue[idx]));
                break;
            case UserConstDataType::kUInt:
                ret.set(idx, m_uintValue[idx] > maximum.m_uintValue[idx] ? maximum.m_uintValue[idx] : (m_uintValue[idx] < minimum.m_uintValue[idx] ? minimum.m_uintValue[idx] : m_uintValue[idx]));
                break;
            }
        }

        return ret;
    }

    bool TypelessVariable::operator < (const TypelessVariable& ref) const
    {
        if (ref.m_type != m_type)
            return false;

        switch (m_type)
        {
        case  UserConstDataType::kBool:
            return m_boolValue < ref.m_boolValue;
            break;
        case  UserConstDataType::kFloat:
            return m_floatValue < ref.m_floatValue;
            break;
        case  UserConstDataType::kInt:
            return m_intValue < ref.m_intValue;
            break;
        case  UserConstDataType::kUInt:
            return m_uintValue < ref.m_uintValue;
            break;
        default:
            return false;
        }
    }

    bool TypelessVariable::operator > (const TypelessVariable& ref) const
    {
        if (ref.m_type != m_type)
            return false;

        switch (m_type)
        {
        case  UserConstDataType::kBool:
            return m_boolValue > ref.m_boolValue;
            break;
        case  UserConstDataType::kFloat:
            return m_floatValue > ref.m_floatValue;
            break;
        case  UserConstDataType::kInt:
            return m_intValue > ref.m_intValue;
            break;
        case  UserConstDataType::kUInt:
            return m_uintValue > ref.m_uintValue;
            break;
        default:
            return false;
        }
    }

    std::string TypelessVariable::stringify() const
    {
        switch (m_type)
        {
        case UserConstDataType::kBool:      return ir::stringify(getDimensionality(), m_boolValue);
        case UserConstDataType::kFloat:     return ir::stringify(getDimensionality(), m_floatValue);
        case UserConstDataType::kInt:       return ir::stringify(getDimensionality(), m_intValue);
        case UserConstDataType::kUInt:      return ir::stringify(getDimensionality(), m_uintValue);
        default:                            return "UNDEFINED";
        }
    }

#define ColwertTo_bool(value)                   static_cast<bool>(value)
#define ColwertTo_userConstTypes_Bool(value)    (static_cast<bool>(value) ? userConstTypes::Bool::kTrue : userConstTypes::Bool::kFalse)
#define ColwertTo_userConstTypes_Int(value)     static_cast<int>(value)
#define ColwertTo_userConstTypes_UInt(value)    static_cast<unsigned int>(value)
#define ColwertTo_userConstTypes_Float(value)   static_cast<float>(value)

#define GetTypelessVariable(otherValueType, otherValueTypeEnum, otherValue, otherValueDimensions) \
    unsigned int validLength = min(otherValueDimensions, getDimensionality()); \
    for (unsigned int idx = 0; idx < validLength; ++idx) \
    { \
        switch (getType()) \
        { \
        case UserConstDataType::kBool:    otherValue[idx] = ColwertTo_##otherValueType(m_boolValue[idx]);  break; \
        case UserConstDataType::kFloat:    otherValue[idx] = ColwertTo_##otherValueType(m_floatValue[idx]);  break; \
        case UserConstDataType::kInt:    otherValue[idx] = ColwertTo_##otherValueType(m_intValue[idx]);  break; \
        case UserConstDataType::kUInt:    otherValue[idx] = ColwertTo_##otherValueType(m_uintValue[idx]);  break; \
        default:              otherValue[idx] = ColwertTo_##otherValueType(m_intValue[idx]);  break; \
        } \
    } \
     \
    if (getType() != otherValueTypeEnum || otherValueDimensions < getDimensionality()) \
    { \
        LOG_ERROR("Got TypelessVariable:\"%s\" and attempted to store it in \"%s\"", stringify().c_str(), ir::stringify(otherValueDimensions, otherValue).c_str()); \
    }

    void TypelessVariable::get(userConstTypes::Bool * ret, unsigned int maxDims) const
    {
        GetTypelessVariable(userConstTypes_Bool, UserConstDataType::kBool, ret, maxDims);
    }

    void TypelessVariable::get(bool * ret, unsigned int maxDims) const
    {
        GetTypelessVariable(bool, UserConstDataType::kBool, ret, maxDims);
    }

    void TypelessVariable::get(userConstTypes::Float * ret, unsigned int maxDims) const
    {
        GetTypelessVariable(userConstTypes_Float, UserConstDataType::kFloat, ret, maxDims);
    }

    void TypelessVariable::get(userConstTypes::UInt * ret, unsigned int maxDims) const
    {
        GetTypelessVariable(userConstTypes_UInt, UserConstDataType::kUInt, ret, maxDims);
    }

    void TypelessVariable::get(userConstTypes::Int * ret, unsigned int maxDims) const
    {
        GetTypelessVariable(userConstTypes_Int, UserConstDataType::kInt, ret, maxDims);
    }

    void * TypelessVariable::getRawMemory()
    {
        return m_intValue;
    }

    void TypelessVariable::set(const userConstTypes::Bool& val)
    {
        set(0u, val);
    }

#define SetTypelessVariableAtIndex(idx, otherValueTypeEnum, otherValue, silent) \
    if (idx < getDimensionality()) \
    { \
        switch (getType()) \
        { \
        case UserConstDataType::kBool:    m_boolValue[idx] =  (static_cast<bool>(otherValue) ? userConstTypes::Bool::kTrue : userConstTypes::Bool::kFalse);  break; \
        case UserConstDataType::kFloat:    m_floatValue[idx] =  static_cast<userConstTypes::Float>(otherValue);                          break; \
        case UserConstDataType::kInt:    m_intValue[idx] =  static_cast<userConstTypes::Int>(otherValue);                          break; \
        case UserConstDataType::kUInt:    m_uintValue[idx] =  static_cast<userConstTypes::UInt>(otherValue);                          break; \
        default:              m_intValue[idx] =  static_cast<userConstTypes::Int>(otherValue);                          break; \
        } \
        if (!silent && getType() != otherValueTypeEnum) \
        { \
            LOG_ERROR("Used \"%s\" to set index %d of TypelessVariable:\"%s\"", ir::stringify(1, &otherValue).c_str(), idx, stringify().c_str()); \
        } \
    } \
    else if (!silent) \
    { \
        LOG_ERROR("Out of bounds. Tried to set index %d of \"%s\" to %s", idx, stringify().c_str(), ir::stringify(1, &otherValue).c_str()); \
    }

#define SetTypelessVariable(otherValueTypeEnum, otherValue, otherValueDimensions) \
    for (unsigned int idx = 0; idx < otherValueDimensions; ++idx) \
    { \
        SetTypelessVariableAtIndex(idx, otherValueTypeEnum, otherValue[idx], true); \
    } \
    if (getType() != otherValueTypeEnum || getDimensionality() < otherValueDimensions) \
    { \
        LOG_ERROR("Used \"%s\" to set TypelessVariable:\"%s\"", ir::stringify(otherValueDimensions, otherValue), stringify().c_str()); \
    }

    void TypelessVariable::set(unsigned int idx, const userConstTypes::Bool& val)
    {
        SetTypelessVariableAtIndex(idx, UserConstDataType::kBool, val, false);
    }

    void TypelessVariable::set(const userConstTypes::Bool * val, unsigned int dims)
    {
        SetTypelessVariable(UserConstDataType::kBool, val, dims);
    }

    void TypelessVariable::set(const bool& val)
    {
        set(0u, val);
    }

    void TypelessVariable::set(unsigned int idx, const bool& val)
    {
        SetTypelessVariableAtIndex(idx, UserConstDataType::kBool, val, false);
    }

    void TypelessVariable::set(const bool * val, unsigned int dims)
    {
        SetTypelessVariable(UserConstDataType::kBool, val, dims);
    }

    void TypelessVariable::set(const userConstTypes::Float& val)
    {
        set(0u, val);
    }

    void TypelessVariable::set(unsigned int idx, const userConstTypes::Float& val)
    {
        SetTypelessVariableAtIndex(idx, UserConstDataType::kFloat, val, false);
    }

    void TypelessVariable::set(const userConstTypes::Float * val, unsigned int dims)
    {
        SetTypelessVariable(UserConstDataType::kFloat, val, dims);
    }

    void TypelessVariable::set(const userConstTypes::UInt& val)
    {
        set(0u, val);
    }

    void TypelessVariable::set(unsigned int idx, const userConstTypes::UInt& val)
    {
        SetTypelessVariableAtIndex(idx, UserConstDataType::kUInt, val, false);
    }

    void TypelessVariable::set(const userConstTypes::UInt * val, unsigned int dims)
    {
        SetTypelessVariable(UserConstDataType::kUInt, val, dims);
    }

    void TypelessVariable::set(const userConstTypes::Int& val)
    {
        set(0u, val);
    }

    void TypelessVariable::set(unsigned int idx, const userConstTypes::Int& val)
    {
        SetTypelessVariableAtIndex(idx, UserConstDataType::kInt, val, false);
    }

    void TypelessVariable::set(const userConstTypes::Int * val, unsigned int dims)
    {
        SetTypelessVariable(UserConstDataType::kInt, val, dims);
    }

    void TypelessVariable::reinitialize(const userConstTypes::Bool & val)
    {
        m_type = UserConstDataType::kBool;
        m_dimensions = 1;
        m_boolValue[0] = val;
    }

    void TypelessVariable::reinitialize(const userConstTypes::Float & val)
    {
        m_type = UserConstDataType::kFloat;
        m_dimensions = 1;
        m_floatValue[0] = val;
    }

    void TypelessVariable::reinitialize(const userConstTypes::Bool * val, unsigned int dims)
    {
        ReinterpretCast(UserConstDataType::kBool);
        setDimensionality(dims);
        set(val, dims);
    }
    void TypelessVariable::reinitialize(const userConstTypes::Float * val, unsigned int dims)
    {
        ReinterpretCast(UserConstDataType::kFloat);
        setDimensionality(dims);
        set(val, dims);
    }
    void TypelessVariable::reinitialize(const userConstTypes::UInt * val, unsigned int dims)
    {
        ReinterpretCast(UserConstDataType::kUInt);
        setDimensionality(dims);
        set(val, dims);
    }
    void TypelessVariable::reinitialize(const userConstTypes::Int * val, unsigned int dims)
    {
        ReinterpretCast(UserConstDataType::kInt);
        setDimensionality(dims);
        set(val, dims);
    }

}
}
