#pragma once

#include <string>

namespace shadermod
{
namespace ir
{
#define MAX_GROUPED_VARIABLE_DIMENSION 4

    class TypelessVariable
    {
    public:
        struct TypeMismatchException
        {   };

        void zeroStorage()
        {
            for (auto& v : m_intValue)
                v = 0;
        }

        TypelessVariable() : m_type(UserConstDataType::NUM_ENTRIES), m_dimensions(1) 
        {
            zeroStorage();
        }
        TypelessVariable(const userConstTypes::Bool& val) : m_type(UserConstDataType::kBool), m_dimensions(1)
        {
            zeroStorage();
            m_boolValue[0] = val;
        }
        TypelessVariable(const bool& val) : m_type(UserConstDataType::kBool), m_dimensions(1)
        {
            zeroStorage();
            m_boolValue[0] = val ? userConstTypes::Bool::kTrue : userConstTypes::Bool::kFalse;
        }
        TypelessVariable(const userConstTypes::Float& val) : m_type(UserConstDataType::kFloat), m_dimensions(1)
        {
            zeroStorage();
            m_floatValue[0] = val;
        }
        TypelessVariable(const userConstTypes::UInt& val) : m_type(UserConstDataType::kUInt), m_dimensions(1)
        {
            zeroStorage();
            m_uintValue[0] = val;
        }
        TypelessVariable(const userConstTypes::Int& val) : m_type(UserConstDataType::kInt), m_dimensions(1)
        {
            zeroStorage();
            m_intValue[0] = val;
        }

        TypelessVariable(const userConstTypes::Bool * val, unsigned int dims) : m_type(UserConstDataType::kBool), m_dimensions(dims)
        {
            zeroStorage();
            for (unsigned int idx = 0; idx < dims; ++idx)
            {
                m_boolValue[idx] = val[idx];
            }
        }
        TypelessVariable(const bool * val, unsigned int dims) : m_type(UserConstDataType::kBool), m_dimensions(dims)
        {
            zeroStorage();
            for (unsigned int idx = 0; idx < dims; ++idx)
            {
                m_boolValue[idx] = val[idx] ? userConstTypes::Bool::kTrue : userConstTypes::Bool::kFalse;
            }
        }
        TypelessVariable(const userConstTypes::Float * val, unsigned int dims) : m_type(UserConstDataType::kFloat), m_dimensions(dims)
        {
            zeroStorage();
            for (unsigned int idx = 0; idx < dims; ++idx)
            {
                m_floatValue[idx] = val[idx];
            }
        }
        TypelessVariable(const userConstTypes::UInt * val, unsigned int dims) : m_type(UserConstDataType::kUInt), m_dimensions(dims)
        {
            zeroStorage();
            for (unsigned int idx = 0; idx < dims; ++idx)
            {
                m_uintValue[idx] = val[idx];
            }
        }
        TypelessVariable(const userConstTypes::Int * val, unsigned int dims) : m_type(UserConstDataType::kInt), m_dimensions(dims)
        {
            zeroStorage();
            for (unsigned int idx = 0; idx < dims; ++idx)
            {
                m_intValue[idx] = val[idx];
            }
        }

        void setDimensionality(unsigned int dims);
        unsigned int getDimensionality() const
        {
            return m_dimensions;
        }

        UserConstDataType getType() const
        {
            return m_type;
        }

        static TypelessVariable makeZero(UserConstDataType t);
        static TypelessVariable makeMin(UserConstDataType t);
        static TypelessVariable makeMax(UserConstDataType t);
        TypelessVariable getClamped(const TypelessVariable& minimum, const TypelessVariable& maximum) const;

        void ReinterpretCast(UserConstDataType  newType) { m_type = newType; }

        void get(userConstTypes::Bool * ret, unsigned int maxDims) const;
        void get(bool * ret, unsigned int maxDims) const;
        void get(userConstTypes::Float * ret, unsigned int maxDims) const;
        void get(userConstTypes::UInt * ret, unsigned int maxDims) const;
        void get(userConstTypes::Int * ret, unsigned int maxDims) const;

        void * getRawMemory();

        bool operator < (const TypelessVariable& ref) const;
        bool operator > (const TypelessVariable& ref) const;

        std::string stringify() const;

        void set(const userConstTypes::Bool & val);
        void set(unsigned int idx, const userConstTypes::Bool & val);
        void set(const userConstTypes::Bool * val, unsigned int dims);

        void set(const bool & val);
        void set(unsigned int idx, const bool & val);
        void set(const bool * val, unsigned int dims);

        void set(const userConstTypes::Float & val);
        void set(unsigned int idx, const userConstTypes::Float & val);
        void set(const userConstTypes::Float * val, unsigned int dims);

        void set(const userConstTypes::UInt & val);
        void set(unsigned int idx, const userConstTypes::UInt & val);
        void set(const userConstTypes::UInt * val, unsigned int dims);

        void set(const userConstTypes::Int & val);
        void set(unsigned int idx, const userConstTypes::Int & val);
        void set(const userConstTypes::Int * val, unsigned int dims);

        void reinitialize(const userConstTypes::Bool & val);
        void reinitialize(const userConstTypes::Float & val);

        void reinitialize(const userConstTypes::Bool * val, unsigned int dims);
        void reinitialize(const userConstTypes::Float * val, unsigned int dims);
        void reinitialize(const userConstTypes::UInt * val, unsigned int dims);
        void reinitialize(const userConstTypes::Int * val, unsigned int dims);

    private:
        friend class UserConstant;

        union
        {
            userConstTypes::Bool        m_boolValue[MAX_GROUPED_VARIABLE_DIMENSION];
            userConstTypes::Float       m_floatValue[MAX_GROUPED_VARIABLE_DIMENSION];
            userConstTypes::Int         m_intValue[MAX_GROUPED_VARIABLE_DIMENSION];
            userConstTypes::UInt        m_uintValue[MAX_GROUPED_VARIABLE_DIMENSION];
        };

        UserConstDataType   m_type;
        unsigned int        m_dimensions;
    };

}
}