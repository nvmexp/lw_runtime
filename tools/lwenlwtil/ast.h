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

#include <boost/hana/functional.hpp>
#include <boost/multi_index/mem_fun.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/phoenix/fusion.hpp>
#include <boost/variant.hpp>
#include <boost/wave/cpp_context.hpp>
#include <boost/wave/cpp_iteration_context.hpp>
#include <boost/wave/cpplexer/cpp_lex_iterator.hpp>

namespace AST
{
    using namespace boost::multi_index;
    using namespace boost::wave;
    namespace hana = boost::hana;

    enum UnaryOp
    {
        UN_MINUS, UN_PLUS
    };

    enum AdditiveOp
    {
        ADD, SUB
    };

    enum MultiplicativeOp
    {
        MULT, DIV, MOD
    };

    struct AdditiveExpression;
    struct PrimaryExpression
      : boost::variant<
          int
        , boost::relwrsive_wrapper<AdditiveExpression>
        >
    {
        typedef boost::variant<
            int
          , boost::relwrsive_wrapper<AdditiveExpression>
          > Base;
        PrimaryExpression() = default;
        PrimaryExpression(int v) : Base(v) {}
        PrimaryExpression(const AdditiveExpression &v) : Base(v) {}

        int operator()() const;
    };

    struct UnaryOperation
    {
        UnaryOp op;
        PrimaryExpression expr;
        int operator()() const
        {
            if (UN_MINUS == op)
            {
                return -expr();
            }
            else
            {
                return expr();
            }
        }
    };

    struct UnaryExpression
      : boost::variant<PrimaryExpression, UnaryOperation>
    {
        typedef boost::variant<PrimaryExpression, UnaryOperation> Base;
        
        UnaryExpression() = default;
        UnaryExpression(const PrimaryExpression &expr) : Base(expr) {}
        UnaryExpression(const UnaryOperation &expr) : Base(expr) {}

        int operator()() const
        {
            return boost::apply_visitor(hana::overload
            (
                [](const auto &expr) -> int
                {
                    return expr();
                }
            ), *this);
        }
    };

    struct MultiplicativeOperation
    {
        MultiplicativeOp op;
        UnaryExpression expr;
        int operator()(int first) const
        {
            switch (op)
            {
            case MULT:
                return first * expr();
            case DIV:
                return first / expr();
            case MOD:
                return first % expr();
            }
            return first;
        }
    };

    struct MultiplicativeExpression
    {
        UnaryExpression                      first;
        std::vector<MultiplicativeOperation> rest;
        int operator()() const
        {
            int res = first();
            for (const auto &expr : rest)
            {
                res = expr(res);
            }
            return res;
        }
    };

    struct AdditiveOperation
    {
        AdditiveOp op;
        MultiplicativeExpression expr;
        int operator()(int first) const
        {
            switch (op)
            {
            case ADD:
                return first + expr();
            case SUB:
                return first - expr();
            }
            return first;
        }
    };

    struct AdditiveExpression
    {
        MultiplicativeExpression       first;
        std::vector<AdditiveOperation> rest;
        int operator()() const
        {
            int res = first();
            for (const auto &expr : rest)
            {
                res = expr(res);
            }
            return res;
        }
    };

    int PrimaryExpression::operator()() const
    {
        return boost::apply_visitor(hana::overload
        (
            [](int v) -> int
            {
                return v;
            },
            [](const AdditiveExpression &expr) -> int
            {
                return expr();
            }
        ), *this);
    }

    struct DirectDeclarator
    {
        cpplexer::lex_token<> name;
        std::vector<AdditiveExpression> arrayDims;
    };

    struct BitField
    {
        DirectDeclarator decl;
        AdditiveExpression bitSize;
    };

    typedef boost::variant<DirectDeclarator, BitField> StructDeclarator;

    enum Type
    {
        LwU8, LwU16, LwU32, LwU64, LwS8, LwS16, LwS32, LwS64, LwBool, TypedefName
    };

    struct TypeSpecifier
    {
        Type         type;
        unsigned int alignment;
        unsigned int typeSize;
        bool         isSigned;
        std::string  typedefName;
    };

    struct StructDeclaration
    {
        TypeSpecifier typeSpecifier;
        StructDeclarator structDeclarator;
        std::string GetName() const
        {
            return boost::apply_visitor(hana::overload
            (
                [](const DirectDeclarator &dd) -> std::string
                {
                    return dd.name.get_value();
                },
                [](const BitField &bf) -> std::string
                {
                    return bf.decl.name.get_value();
                }
            ), structDeclarator);
        }
    };

    struct seq {};
    struct by_name {};

    typedef multi_index_container<
        StructDeclaration
      , indexed_by<
        sequenced<boost::multi_index::tag<seq>>
          , ordered_unique<
                boost::multi_index::tag<by_name>
              , const_mem_fun<
                    StructDeclaration
                  , std::string
                  , &StructDeclaration::GetName
                  >
              >
           >
      > StructDeclarations;

    struct StructOrUnionSpecifier
    {
        cpplexer::lex_token<> structName;
        StructDeclarations decls;
        unsigned int alignment = 0;

        std::string GetName() const
        {
            return structName.get_value();
        }
    };

    typedef multi_index_container<
        StructOrUnionSpecifier
      , indexed_by<
            sequenced<boost::multi_index::tag<seq>>
          , ordered_unique<
                boost::multi_index::tag<by_name>
              , const_mem_fun<
                    StructOrUnionSpecifier
                  , std::string
                  , &StructOrUnionSpecifier::GetName
                  >
              >
          >
      > Structs;
}

BOOST_FUSION_ADAPT_STRUCT(
    AST::UnaryOperation,
    op,
    expr
)

BOOST_FUSION_ADAPT_STRUCT(
    AST::MultiplicativeOperation,
    op,
    expr
)

BOOST_FUSION_ADAPT_STRUCT(
    AST::MultiplicativeExpression,
    first,
    rest
)

BOOST_FUSION_ADAPT_STRUCT(
    AST::AdditiveOperation,
    op,
    expr
)

BOOST_FUSION_ADAPT_STRUCT(
    AST::AdditiveExpression,
    first,
    rest
)

BOOST_FUSION_ADAPT_STRUCT(
    AST::DirectDeclarator,
    name,
    arrayDims
)

BOOST_FUSION_ADAPT_STRUCT(
    AST::BitField,
    decl,
    bitSize
)

BOOST_FUSION_ADAPT_STRUCT(
    AST::TypeSpecifier,
    type,
    alignment,
    typeSize,
    isSigned,
    typedefName
)

BOOST_FUSION_ADAPT_STRUCT(
    AST::StructDeclaration,
    typeSpecifier,
    structDeclarator
)

BOOST_FUSION_ADAPT_STRUCT(
    AST::StructOrUnionSpecifier,
    decls,
    structName
)
