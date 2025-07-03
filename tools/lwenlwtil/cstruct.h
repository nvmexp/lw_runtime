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

#include <boost/phoenix/fusion.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/wave/cpp_context.hpp>
#include <boost/wave/cpp_iteration_context.hpp>
#include <boost/wave/cpplexer/cpp_lex_iterator.hpp>

#include "ast.h"
#include "clexer.h"

using namespace boost::spirit;
using namespace boost::wave;

using namespace WaveParsers;

template <typename Iterator>
class WhiteSpace : public qi::grammar<Iterator>
{
public:
    WhiteSpace() : WhiteSpace::base_type(skip)
    {
        skip = qi::lit(T_SPACE) | T_NEWLINE;
    }

private:
    typename WhiteSpace::start_type skip;
};

template <typename Iterator>
struct CStructsGrammar : qi::grammar<Iterator, WhiteSpace<Iterator>, AST::Structs()>
{
    typedef typename CStructsGrammar::skipper_type skipper_type;
    typedef typename CStructsGrammar::start_type   start_type;
    typedef typename CStructsGrammar::base_type    base_type;

    template <typename RuleSignature = unused_type>
    using rule = qi::rule<Iterator, WhiteSpace<Iterator>, RuleSignature>;

    CStructsGrammar() : base_type(start)
    {
        namespace ph = boost::phoenix;
        using boost::phoenix::at_c;

        start %= *struct_or_union_specifier;

        type_specifier =
            Identifier("LwU8")   [at_c<0>(_val) = AST::LwU8,   at_c<1>(_val) = 1, at_c<2>(_val) = 1, at_c<3>(_val) = false] |
            Identifier("LwU16")  [at_c<0>(_val) = AST::LwU16,  at_c<1>(_val) = 2, at_c<2>(_val) = 2, at_c<3>(_val) = false] |
            Identifier("LwU32")  [at_c<0>(_val) = AST::LwU32,  at_c<1>(_val) = 4, at_c<2>(_val) = 4, at_c<3>(_val) = false] |
            Identifier("LwU64")  [at_c<0>(_val) = AST::LwU64,  at_c<1>(_val) = 8, at_c<2>(_val) = 8, at_c<3>(_val) = false] |
            Identifier("LwS8")   [at_c<0>(_val) = AST::LwS8,   at_c<1>(_val) = 1, at_c<2>(_val) = 1, at_c<3>(_val) = true]  |
            Identifier("LwS16")  [at_c<0>(_val) = AST::LwS16,  at_c<1>(_val) = 2, at_c<2>(_val) = 2, at_c<3>(_val) = true]  |
            Identifier("LwS32")  [at_c<0>(_val) = AST::LwS32,  at_c<1>(_val) = 4, at_c<2>(_val) = 4, at_c<3>(_val) = true]  |
            Identifier("LwS64")  [at_c<0>(_val) = AST::LwS64,  at_c<1>(_val) = 8, at_c<2>(_val) = 8, at_c<3>(_val) = true]  |
            Identifier("LwBool") [at_c<0>(_val) = AST::LwBool, at_c<1>(_val) = 1, at_c<2>(_val) = 1, at_c<3>(_val) = false] |
            typedef_name
            [
                at_c<0>(_val) = AST::TypedefName,
                at_c<1>(_val) = 0,
                at_c<2>(_val) = 0,
                at_c<4>(_val) = ph::bind(&cpplexer::lex_token<>::get_value, qi::_1)
            ];

        struct_or_union_specifier %=
            T_TYPEDEF >>
            qi::omit[struct_or_union] >> T_LEFTBRACE >> *struct_declaration >> T_RIGHTBRACE >> typedef_name >> T_SEMICOLON;

        struct_or_union =
            qi::lit(T_STRUCT) | T_UNION;

        struct_declaration %=
            type_specifier >> struct_declarator >> T_SEMICOLON;

        struct_declarator %=
            (direct_declarator >> !qi::lit(T_COLON)) |
            (direct_declarator >> T_COLON >> constant_expression);

        direct_declarator %=
            Token(T_IDENTIFIER) >> *(T_LEFTBRACKET >> constant_expression >> T_RIGHTBRACKET);

        constant_expression = expression.alias();

        typedef_name = Token(T_IDENTIFIER);

        expression = additive_expression.alias();

        additive_expression %=
            multiplicative_expression >>
           *(additive_operator >> multiplicative_expression);

        additive_operator =
            (T_PLUS >> qi::attr(AST::ADD)) |
            (T_MINUS >> qi::attr(AST::SUB));

        multiplicative_expression %=
            unary_expression >>
           *(multiplicative_operator >> unary_expression);
        
        multiplicative_operator =
            (T_STAR >> qi::attr(AST::MULT)) |
            (T_DIVIDE >> qi::attr(AST::DIV)) |
            (T_PERCENT >> qi::attr(AST::MOD));

        unary_expression %=
            primary_expression |
            (unary_operator >> primary_expression);

        unary_operator =
            (T_MINUS >> qi::attr(AST::UN_MINUS)) |
            (T_PLUS >> qi::attr(AST::UN_PLUS));

        primary_expression %=
            Int |
            (T_LEFTPAREN >> expression >> T_RIGHTPAREN);

        BOOST_SPIRIT_DEBUG_NODES(
            (type_specifier)(struct_or_union_specifier)(struct_or_union)(struct_declaration)
            (struct_declarator)(direct_declarator)(constant_expression)(typedef_name)
            (primary_expression)(unary_operator)(unary_expression)(multiplicative_operator)
            (multiplicative_expression)(additive_expression)(additive_operator)(expression)
        );
    }

    start_type start;

    rule<AST::TypeSpecifier()>            type_specifier;
    rule<AST::StructOrUnionSpecifier()>   struct_or_union_specifier;
    rule<>                                struct_or_union;
    rule<AST::StructDeclaration()>        struct_declaration;
    rule<AST::StructDeclarator()>         struct_declarator;
    rule<AST::DirectDeclarator()>         direct_declarator;
    rule<AST::AdditiveExpression()>       constant_expression;
    rule<cpplexer::lex_token<>()>         typedef_name;
    rule<AST::PrimaryExpression()>        primary_expression;
    rule<AST::UnaryOp()>                  unary_operator;
    rule<AST::UnaryExpression()>          unary_expression;
    rule<AST::MultiplicativeOp()>         multiplicative_operator;
    rule<AST::MultiplicativeExpression()> multiplicative_expression;
    rule<AST::AdditiveExpression()>       additive_expression;
    rule<AST::AdditiveOp()>               additive_operator;
    rule<AST::AdditiveExpression()>       expression;
};
