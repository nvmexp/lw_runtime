/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2018 by LWPU Corporation. All rights reserved. All information
 * contained herein is proprietary and confidential to LWPU Corporation. Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */
#pragma once

#include <string>
#include <tuple>
#include <vector>

#include <boost/hana/functional.hpp>
#include <boost/multi_index/mem_fun.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/variant.hpp>

using namespace boost::multi_index;
using namespace boost::spirit;
namespace hana = boost::hana;

// Following data structures simulate a static C-structure in runtime. I.e. a
// structure is a collection of fields. Each field is a name and a value. The
// value is either a primitive type or another structure. It's a relwrsive
// collection that is implemented using `boost::variant`.

// The basic type is a field. It has a name, a value and a list of dimensions.
// What is the value is not defined yet, it's some arbitrary template argument
// at this point.
template <typename RelwrsiveVariant>
struct UndefinedField
{
    std::string         name;
    RelwrsiveVariant    value;
    std::vector<size_t> dims;
};

template <typename RelwrsiveVariant>
using StructFieldsIndices = indexed_by<
    sequenced<>
  , ordered_unique<
        member<
            UndefinedField<RelwrsiveVariant>
          , std::string
          , &UndefinedField<RelwrsiveVariant>::name
          >
      >
  >;

// This is a collection of fields ordered by the name. Once again the type of
// value is not defined yet.
template <typename RelwrsiveVariant>
using UndefinedStructFields = multi_index_container<
    UndefinedField<RelwrsiveVariant>
  , StructFieldsIndices<RelwrsiveVariant>
  >;

// Finally, a structure is a typedef name and a set fields.
template <typename RelwrsiveVariant>
class UndefinedStruct
{
    std::string                             typedefName;
    UndefinedStructFields<RelwrsiveVariant> fields;

public:
    const std::string & GetTypedefName() const { return typedefName; }
    void SetTypedefName(const std::string &v) { typedefName = v; }

    const UndefinedStructFields<RelwrsiveVariant> & GetFields() const { return fields; }
    void SetFields(const UndefinedStructFields<RelwrsiveVariant> &v) { fields = v; }

    const RelwrsiveVariant& GetField(const std::string &name) const
    {
        std::vector<
            std::tuple<
                std::string           // field name
              , std::vector<unsigned> // array indices
              >
          > fieldNames;
        qi::parse(
            name.cbegin(), name.cend(),
            (
                qi::as_string[qi::alpha >> *(qi::alnum | qi::char_('_'))] >> // an identifier
                *('[' >> qi::uint_ >> ']')                                   // array of dimensions
            ) % '.',
            fieldNames
        );
        
        const auto *lwrFields = &fields;

        // nth_index<1> is a collection indexed by field names
        typename UndefinedStructFields<RelwrsiveVariant>::template nth_index<1>::type::const_iterator it;
        
        const RelwrsiveVariant *retVal = nullptr;

        // descend the tree of structure fields
        for (const auto &fn : fieldNames)
        {
            // get the collection of fields indexed by name
            const auto &byName = lwrFields->template get<1>();
            // get the field by its name
            it = byName.find(std::get<0>(fn));
            if (byName.cend() == it)
            {
                throw std::ilwalid_argument("Field " + name + " is not defined");
            }

            // the field is a variant, `apply_visitor` gives us access to the
            // variant that is another structure
            boost::apply_visitor(hana::overload
            (
                [&](const UndefinedStruct<RelwrsiveVariant> &nextStruct)
                {
                    // next tree node
                    lwrFields = &nextStruct.fields;
                },
                [&](const std::vector<RelwrsiveVariant> &arrayOfNextStructs)
                {
                    // the values of a multidimensional array are flattened,
                    // callwlate the correct index
                    size_t idx = 0;
                    for (size_t i = 0; it->dims.size() > i; ++i)
                    {
                        idx = idx * it->dims[i] + std::get<1>(fn)[i];
                    }
                    boost::apply_visitor(hana::overload
                    (
                        [&](const UndefinedStruct<RelwrsiveVariant> &nextStruct)
                        {
                            // next tree node
                            lwrFields = &nextStruct.fields;
                        },
                        [&](const auto &x)
                        {
                            retVal = &arrayOfNextStructs[idx];
                        }
                    ), arrayOfNextStructs[idx]);
                },
                [&](const auto &)
                {
                    retVal = &it->value;
                }
            ), it->value);
        }

        return *retVal;
    }

    template <typename ValueType>
    std::enable_if_t<std::is_integral<ValueType>::value>
    SetField(const std::string &name, ValueType v)
    {
        std::vector<
            std::tuple<
                std::string           // field name
              , std::vector<unsigned> // array indices
              >
          > fieldNames;
        qi::parse(
            name.cbegin(), name.cend(),
            (
                qi::as_string[qi::alpha >> *(qi::alnum | qi::char_('_'))] >> // an identifier
                *('[' >> qi::uint_ >> ']')                                   // array of dimensions
            ) % '.',
            fieldNames
        );

        SetField(fieldNames.cbegin(), fieldNames.cend(), v);
    }

private:
    template <typename Iterator, typename ValueType>
    void SetField(Iterator start, Iterator end, ValueType v)
    {
        // get the collection of fields indexed by name
        auto &byName = fields.template get<1>();
        // get the field by its name
        auto it = byName.find(std::get<0>(*start));
        if (byName.cend() == it)
        {
            throw std::ilwalid_argument("Field " + std::get<0>(*start) + " is not defined");
        }

        // relwrsively call SetField
        byName.modify(it, [&](auto &f)
        {
            boost::apply_visitor(hana::overload
            (
                [&](UndefinedStruct<RelwrsiveVariant> &nextStruct)
                {
                    // relwrsive step
                    nextStruct.SetField(start + 1, end, v);
                },
                [&](std::vector<RelwrsiveVariant> &arrayOfNextStructs)
                {
                    // the values of a multidimensional array are flattened,
                    // callwlate the correct index
                    size_t idx = 0;
                    for (size_t i = 0; it->dims.size() > i; ++i)
                    {
                        idx = idx * it->dims[i] + std::get<1>(*start)[i];
                    }
                    boost::apply_visitor(hana::overload
                    (
                        [&](UndefinedStruct<RelwrsiveVariant> &nextStruct)
                        {
                            // relwrsive step
                            nextStruct.SetField(start + 1, end, v);
                        },
                        [&](auto &)
                        {
                            arrayOfNextStructs[idx] = v;
                        }
                    ), arrayOfNextStructs[idx]);
                },
                [&](auto &)
                {
                    f.value = v;
                }
            ), f.value);
        });
    }
};

// This defines the C++ type of a value of a field: a primitive type, another
// structure or an array.
typedef typename boost::make_relwrsive_variant<
    int, unsigned int, bool
  , UndefinedStruct<boost::relwrsive_variant_>
  , std::vector<boost::relwrsive_variant_>
  >::type FieldValue;

// Finally, we can specialize the concrete type for the template argument.
using Field = UndefinedField<FieldValue>;
using StructFields = UndefinedStructFields<FieldValue>;
using Struct = UndefinedStruct<FieldValue>;

// We can put all structures into a container indexed by the typedef name.
typedef multi_index_container<
    Struct
  , indexed_by<
        sequenced<>
      , ordered_unique<
            const_mem_fun<
                Struct
              , const std::string &
              , &Struct::GetTypedefName
              >
          >
      >
  > Structures;
