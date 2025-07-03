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

#include <boost/mpl/bool.hpp>
#include <boost/phoenix/fusion.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/wave/cpp_context.hpp>
#include <boost/wave/cpp_iteration_context.hpp>
#include <boost/wave/cpplexer/cpp_lex_iterator.hpp>

using namespace boost::wave;

typedef cpplexer::lex_token<> LexToken;
typedef cpplexer::lex_iterator<LexToken> LexIterator;
typedef context<
    std::string::const_iterator
  , LexIterator
  , iteration_context_policies::load_file_to_string
  > WaveContext;

namespace WaveParsers
{
    BOOST_SPIRIT_TERMINAL_EX(Token);
    BOOST_SPIRIT_TERMINAL_EX(Int);
    BOOST_SPIRIT_TERMINAL_EX(Identifier)
}

namespace boost { namespace spirit {
    // Enables T_TOKEN
    template <>
    struct use_terminal<qi::domain, token_id> : mpl::true_ {};

    // Enables Token(T_TOKEN).
    template <typename A0>
    struct use_terminal<
        qi::domain
      , terminal_ex<
            WaveParsers::tag::Token
          , fusion::vector1<A0>
          >
      > : mpl::true_
    {};

    // Enables qi::lit(T_TOKEN). By convention qi::lit(T_TOKEN) is equivalent to
    // T_TOKEN. qi::lit(T_TOKEN) is usefull when the compiler cannot guess we
    // are generating a parser. For example T_TOKEN1 >> T_TOKEN2 is just an
    // arithmetic expression.
    template <typename A0>
    struct use_terminal<
        qi::domain
      , terminal_ex<tag::lit, fusion::vector1<A0>>
      , typename enable_if<is_same<A0, token_id>>::type
      > : mpl::true_
    {};

    // Enables Int.
    template <>
    struct use_terminal<qi::domain, WaveParsers::tag::Int> : mpl::true_ {};

    // Enables Identifier("someIdentifier")
    template <typename A0>
    struct use_terminal<
        qi::domain
      , terminal_ex<
            WaveParsers::tag::Identifier
          , fusion::vector1<A0>
          >
      > : mpl::true_
    {};
}}

namespace WaveParsers
{
    using namespace boost::spirit;
    class TokenParser : qi::primitive_parser<TokenParser>
    {
    public:
        explicit TokenParser(const token_id id)
          : id(id)
        {}

        template <typename Context, typename Iterator>
        struct attribute
        {
            typedef LexToken type;
        };

        template <typename Iterator, typename Context, typename Skipper, typename Attribute>
        bool parse(Iterator& first, Iterator const& last, Context&, Skipper const& skipper,
                   Attribute& attr) const
        {
            qi::skip_over(first, last, skipper);

            if (first != last && token_id(*first) == id)
            {
                traits::assign_to(*first, attr);
                ++first;
                return true;
            }

            return false;
        }

        template <typename Context>
        info what(Context&) const
        {
            std::string what = "Token ";
            what += get_token_name(id).c_str();
            return info(what);
        }

    private:
        token_id id;
    };

    class LiteralTokenParser : qi::primitive_parser<LiteralTokenParser>
    {
    public:
        LiteralTokenParser(token_id id)
          : id(id)
        {}

        template <typename Context, typename Iterator>
        struct attribute
        {
            typedef unused_type type;
        };

        template <typename Iterator, typename Context, typename Skipper, typename Attribute>
        bool parse(Iterator& first, Iterator const& last, Context&, Skipper const& skipper,
                   Attribute& attr) const
        {
            qi::skip_over(first, last, skipper);

            if (first != last && token_id(*first) == id)
            {
                ++first;
                return true;
            }

            return false;
        }

        template <typename Context>
        info what(Context&) const
        {
            std::string what = "Token ";
            what += get_token_name(id).c_str();
            return what;
        }

    private:
        token_id id;
    };

    class IntParser : qi::primitive_parser<IntParser>
    {
    public:
        template <typename Context, typename Iterator>
        struct attribute
        {
            typedef int type;
        };

        template <typename Iterator, typename Context, typename Skipper, typename Attribute>
        bool parse(Iterator& first, Iterator const& last, Context&, Skipper const& skipper,
                   Attribute& attr) const
        {
            qi::skip_over(first, last, skipper);
            if (first != last && token_id(*first) == T_INTLIT)
            {
                using ascii::no_case;
                int v = 0;

                qi::parse(
                    first->get_value().begin(),
                    first->get_value().end(),
                    (no_case["0x"] >> qi::hex) | ('0' >> qi::oct) | qi::uint_,
                    v
                );
                traits::assign_to(v, attr);
                ++first;
                return true;
            }

            return false;
        }

        template <typename Context>
        info what(Context&) const
        {
            return info("integer");
        }
    };

    class IdentifierParser : qi::primitive_parser<IdentifierParser>
    {
    public:
        IdentifierParser(const char *name)
          : idName(name)
        {}

        template <typename Context, typename Iterator>
        struct attribute
        {
            typedef LexToken type;
        };

        template <typename Iterator, typename Context, typename Skipper, typename Attribute>
        bool parse(Iterator& first, Iterator const& last, Context&, Skipper const& skipper,
                   Attribute& attr) const
        {
            qi::skip_over(first, last, skipper);

            if (
                first != last &&
                T_IDENTIFIER == token_id(*first) &&
                first->get_value() == idName
            )
            {
                traits::assign_to(*first, attr);
                ++first;
                return true;
            }

            return false;
        }

        template <typename Context>
        info what(Context&) const
        {
            std::string what = "Identifier ";
            what += idName;
            return boost::spirit::info(what);
        }

    private:
        const char *idName;
    };
}

namespace boost { namespace spirit { namespace qi {
    template <typename Modifiers, typename A0>
    struct make_primitive<
        terminal_ex<
            WaveParsers::tag::Token
          , fusion::vector1<A0>
          >
      , Modifiers
      >
    {
        typedef WaveParsers::TokenParser result_type;

        template <typename Terminal>
        result_type operator()(Terminal const& term, unused_type) const
        {
            return result_type(fusion::at_c<0>(term.args));
        }
    };

    template <typename Modifiers>
    struct make_primitive<token_id, Modifiers>
    {
        typedef WaveParsers::LiteralTokenParser result_type;

        template <typename TokenId>
        result_type operator()(TokenId id, unused_type) const
        {
            return result_type(id);
        }
    };

    template <typename Modifiers>
    struct make_primitive<WaveParsers::tag::Int, Modifiers>
    {
        typedef WaveParsers::IntParser result_type;

        result_type operator()(unused_type, unused_type) const
        {
            return result_type();
        }
    };

    template <typename Modifiers, typename A0>
    struct make_primitive<
        terminal_ex<tag::lit, fusion::vector1<A0>>
      , Modifiers
      , typename enable_if<is_same<A0, token_id>>::type
      >
    {
        typedef WaveParsers::LiteralTokenParser result_type;

        template <typename Terminal>
        result_type operator()(Terminal const& term, unused_type) const
        {
            return result_type(fusion::at_c<0>(term.args));
        }
    };

    template <typename Modifiers, typename A0>
    struct make_primitive<
        terminal_ex<
            WaveParsers::tag::Identifier
          , fusion::vector1<A0>
          >
      , Modifiers
      >
    {
        typedef WaveParsers::IdentifierParser result_type;

        template <typename Terminal>
        result_type operator()(Terminal const& term, unused_type) const
        {
            return result_type(fusion::at_c<0>(term.args));
        }
    };
}}}
