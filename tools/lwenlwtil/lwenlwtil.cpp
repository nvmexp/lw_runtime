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
#include "pch.hpp"

#include <string>
#include <vector>

#include <boost/container/flat_map.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/fusion/adapted/std_tuple.hpp>
#include <boost/hana/functional.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/multi_index/global_fun.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/mem_fun.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/phoenix/bind/bind_member_function.hpp>
#include <boost/phoenix/fusion.hpp>
#include <boost/phoenix/operator.hpp>
#include <boost/program_options.hpp>
#include <boost/range/adaptor/indirected.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/range/algorithm/find_if.hpp>
#include <boost/range/algorithm/max_element.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/variant/relwrsive_variant.hpp>
#include <boost/wave/cpp_context.hpp>
#include <boost/wave/cpp_iteration_context.hpp>
#include <boost/wave/cpplexer/cpp_lex_iterator.hpp>

#include "core/include/types.h"

#include "h265dpbmanager.h"
#include "h265parser.h"

#include "ast.h"
#include "cstruct.h"
#include "dynstruct.h"
#include "inbitstream.h"

// Besides printing out the values of control structures this utility can also
// help filling in the values of output pictures indices and reference pictures
// lists based on a video stream (--ref-from option). This operation requires a
// special explanation. 
//
// For the LWENC engine to work we have to tell it where to find reference
// pictures and where to output the restored (i.e. encoded and then decoded)
// current picture. A collection of the restored pictures used for reference is
// called DPB (Decoded Picture Buffer). The process of managing DPB is described
// in section C.3 of ITU-T H.265. Here is how it works for the purpose of LWENC.
// We have a fixed size set of surfaces. For encoding a frame you tell the
// engine the index of each surface using SET_IN_REF_PICXX method in the push
// buffer. Then these fields in the control structures:
// `lwenc_h265_drv_pic_setup_s::pic_control::l0` and `l1` use these indices to
// tell the engine what to use for reference.
// The index for the output pictures is chosen in a way to overwrite only
// surfaces that are not used for reference anymore.
// Therefore, first you chose slots in the DPB array for output. This is done by
// `PrintH265OutIdx`. Then for the frame you are printing the control structures
// for you have to update the reference lists according to the current content
// of the the DPB.

using namespace std::string_literals;

using namespace boost::spirit;
using namespace boost::wave;
using namespace boost::multi_index;

namespace fs = boost::filesystem;
namespace po = boost::program_options;
namespace hana = boost::hana;

template <typename InputIterator>
using InBitStream = BitsIO::GenericInBitStream<InputIterator, BitsIO::LittleEndian>;

template <typename InputIterator>
InBitStream<InputIterator> MakeInBitStream(InputIterator start, InputIterator finish)
{
    return InBitStream<InputIterator>(start, finish);
}

template <typename Cont, typename Iterator>
void UpdateAlignment(Cont &allStructs, Iterator &where)
{
    unsigned lwrAlignment = 0;
    for (const auto &decl : where->decls)
    {
        if (AST::TypedefName != decl.typeSpecifier.type)
        {
            lwrAlignment = std::max(lwrAlignment, decl.typeSpecifier.alignment);
        }
        else
        {
            auto it = allStructs.find(decl.typeSpecifier.typedefName);
            if (allStructs.end() != it)
            {
                UpdateAlignment(allStructs, it);
            }
            else
            {
                throw std::ilwalid_argument(
                    "Structure " + decl.typeSpecifier.typedefName + " is not defined");
            }
            lwrAlignment = std::max(lwrAlignment, it->alignment);
        }
    }
    allStructs.modify(where, [lwrAlignment](auto &s) {
        s.alignment = lwrAlignment;
    });
}

template <typename AstCont, typename Iterator, typename BitStream>
void ReadStructure(const AstCont &astStructs, const Iterator &lwrStruct, BitStream &is, Struct &out)
{
    out.SetTypedefName(lwrStruct->structName.get_value());
    StructFields fields;
    for (const auto &decl : lwrStruct->decls)
    {
        boost::apply_visitor(hana::overload
        (
            [&](const AST::DirectDeclarator &dd)
            {
                Field newField;
                newField.name = dd.name.get_value();
                if (AST::TypedefName != decl.typeSpecifier.type)
                {
                    if (dd.arrayDims.empty())
                    {
                        is.AlignTo(decl.typeSpecifier.alignment);
                        if (decl.typeSpecifier.isSigned)
                        {
                            auto value = is.ReadSigned(decl.typeSpecifier.typeSize * 8);
                            if (AST::LwBool == decl.typeSpecifier.type)
                            {
                                newField.value = 0 == value ? false : true;
                            }
                            else
                            {
                                newField.value = static_cast<int>(value);
                            }
                        }
                        else
                        {
                            auto value = is.Read(decl.typeSpecifier.typeSize * 8);
                            if (AST::LwBool == decl.typeSpecifier.type)
                            {
                                newField.value = 0 == value ? false : true;
                            }
                            else
                            {
                                newField.value = static_cast<unsigned int>(value);
                            }
                        }
                    }
                    else
                    {
                        using namespace boost::adaptors;
                        boost::copy(
                            dd.arrayDims | transformed([](const auto &v) { return v(); }),
                            std::back_inserter(newField.dims)
                        );
                        unsigned numElems = boost::accumulate(
                            dd.arrayDims,
                            1U,
                            [](auto v, const auto &e) { return v * e(); }
                        );
                        std::vector<FieldValue> fv(numElems);
                        for (unsigned i = 0; numElems > i; ++i)
                        {
                            is.AlignTo(decl.typeSpecifier.alignment);
                            if (decl.typeSpecifier.isSigned)
                            {
                                auto value = is.ReadSigned(decl.typeSpecifier.typeSize * 8);
                                if (AST::LwBool == decl.typeSpecifier.type)
                                {
                                    fv[i] = 0 == value ? false : true;
                                }
                                else
                                {
                                    fv[i] = static_cast<int>(value);
                                }
                            }
                            else
                            {
                                auto value = is.Read(decl.typeSpecifier.typeSize * 8);
                                if (AST::LwBool == decl.typeSpecifier.type)
                                {
                                    fv[i] = 0 == value ? false : true;
                                }
                                else
                                {
                                    fv[i] = static_cast<unsigned int>(value);
                                }
                            }
                        }
                        newField.value = fv;
                    }
                }
                else
                {
                    auto it = astStructs.find(decl.typeSpecifier.typedefName);
                    if (astStructs.end() != it)
                    {
                        if (dd.arrayDims.empty())
                        {
                            Struct newStruct;
                            newStruct.SetTypedefName(decl.typeSpecifier.typedefName);
                            is.AlignTo(it->alignment);
                            ReadStructure(astStructs, it, is, newStruct);
                            newField.value = newStruct;
                        }
                        else
                        {
                            using namespace boost::adaptors;
                            boost::copy(
                                dd.arrayDims | transformed([](const auto &v) { return v(); }),
                                std::back_inserter(newField.dims)
                            );
                            unsigned numElems = boost::accumulate(
                                dd.arrayDims,
                                1U,
                                [](auto v, const auto &e) { return v * e(); }
                            );
                            std::vector<FieldValue> fv(numElems);
                            for (unsigned i = 0; numElems > i; ++i)
                            {
                                Struct newStruct;
                                newStruct.SetTypedefName(decl.typeSpecifier.typedefName);
                                is.AlignTo(it->alignment);
                                ReadStructure(astStructs, it, is, newStruct);
                                fv[i] = newStruct;
                            }
                            newField.value = fv;
                        }
                    }
                    else
                    {
                        throw std::ilwalid_argument(
                            "Structure " + decl.typeSpecifier.typedefName + " is not defined");
                    }
                }
                fields.push_back(newField);
            },
            [&](const AST::BitField &bf)
            {
                Field newField;
                newField.name = bf.decl.name.get_value();

                auto lwrOffset = is.GetLwrrentOffset();
                auto bitsInWord = decl.typeSpecifier.alignment * 8;
                auto lwrBitInWord = lwrOffset % bitsInWord;

                // bitfield cannot cross alignment boundary
                if ((lwrBitInWord + bf.bitSize() - 1) > bitsInWord)
                {
                    is.AlignTo(decl.typeSpecifier.alignment);
                }
                if (decl.typeSpecifier.isSigned)
                {
                    auto value = is.ReadSigned(bf.bitSize());
                    if (AST::LwBool == decl.typeSpecifier.type)
                    {
                        newField.value = 0 != value;
                    }
                    else
                    {
                        newField.value = static_cast<int>(value);
                    }
                }
                else
                {
                    auto value = is.Read(bf.bitSize());
                    if (AST::LwBool == decl.typeSpecifier.type)
                    {
                        newField.value = 0 != value;
                    }
                    else
                    {
                        newField.value = static_cast<unsigned int>(value);
                    }
                }
                fields.push_back(newField);
            }
        ), decl.structDeclarator);
    }
    out.SetFields(fields);
}

template <typename AstCont, typename BinIt, typename OutputIterator>
void ReadStructsAtOffset
(
    const Struct &drvCtrl,
    const char *offsName,
    const char *sizeName,
    const char *structName,
    AstCont &astStructs,
    BinIt start,
    BinIt finish,
    OutputIterator outIt
)
{
    auto offs = boost::get<unsigned>(drvCtrl.GetField(offsName));
    auto size = boost::get<unsigned>(drvCtrl.GetField(sizeName)) + 1;

    if (0 != offs)
    {
        auto is = MakeInBitStream(start + offs, finish);
        auto it = astStructs.find(structName);
        UpdateAlignment(astStructs, it);
        for (unsigned i = 0; size > i; ++i)
        {
            Struct sc;
            ReadStructure(astStructs, it, is, sc);
            *outIt++ = sc;
        }
    }
}

template <typename OutputIterator>
void GetStructFields(const Struct &s, const std::string &prefix, OutputIterator outIt)
{
    for (const auto &f : s.GetFields())
    {
        boost::apply_visitor(hana::overload
        (
            [&](auto v) // all types besides Struct and an array
            {
                if (0 != v)
                {
                    std::string qualifiedName = prefix + '.' + f.name;
                    if ("magic" == f.name || "gop_length" == f.name ||
                        "slice_stat_offset" == f.name || "mpec_stat_offset" == f.name)
                    {
                        *outIt++ = std::make_tuple(
                            qualifiedName,
                            (boost::format("0x%08x") % v).str()
                        );
                    }
                    else
                    {
                        *outIt++ = std::make_tuple(
                            qualifiedName,
                            (boost::format("%||") % v).str()
                        );
                    }
                }
            },
            [&](const Struct &v)
            {
                GetStructFields(v, prefix + '.' + f.name, outIt);
            },
            [&](const std::vector<FieldValue> &v)
            {
                size_t numElems = boost::accumulate(
                    f.dims,
                    1ULL,
                    [](auto v, const auto &e) { return v * e; }
                );
                for (size_t i = 0; numElems > i; ++i)
                {
                    std::string qualifiedName = (boost::format("%s.%s") % prefix % f.name).str();
                    size_t d = numElems;
                    size_t m = i;
                    for (size_t idx = 0; f.dims.size() > idx; ++idx)
                    {
                        d /= f.dims[idx];
                        qualifiedName += (boost::format("[%u]") % (m / d)).str();
                        m %= d;
                    }
                    boost::apply_visitor(hana::overload
                    (
                        [&](auto v)
                        {
                            if (0 != v)
                            {
                                if ("bitmask" == f.name || "diff_pic_order_cnt_zero" == f.name)
                                {
                                    *outIt++ = std::make_tuple(
                                        qualifiedName,
                                        (boost::format("0x%08x") % v).str()
                                    );
                                }
                                else
                                {
                                    *outIt++ = std::make_tuple(
                                        qualifiedName,
                                        (boost::format("%||") % v).str()
                                    );
                                }
                            }
                        },
                        [&](const Struct &v)
                        {
                            GetStructFields(v, qualifiedName, outIt);
                        },
                        [&](const std::vector<FieldValue> &) { }
                    ), v[i]);
                }
            }
        ), f.value);
    }
}

template <typename InIterator, typename OutputIterator>
void GetStructFields(InIterator start, InIterator finish, const std::string &prefix, OutputIterator outIt)
{
    auto size = std::distance(start, finish);
    size_t count = 0;
    for (auto it = start; finish > it; ++it, ++count)
    {
        if (1 == size)
        {
            GetStructFields(*it, prefix, outIt);
        }
        else
        {
            GetStructFields(*it, (boost::format("%||[%||]") % prefix % count).str(), outIt);
        }
    }
}

template <typename AstCont, typename BinIt, typename OutputIterator>
void ReadH264DrvStructs
(
    AstCont &astStructs,
    const Struct &lwEncDevStruct,
    const std::string &slicePrefix,
    const std::string &mePrefix,
    const std::string &mdPrefix,
    const std::string &qPrefix,
    const std::string &wpPrefix,
    BinIt start,
    BinIt finish,
    OutputIterator outIt
)
{
    std::vector<Struct> slices;
    std::vector<Struct> me;
    std::vector<Struct> md;
    std::vector<Struct> qCtrl;
    std::vector<Struct> wpCtrl;

    ReadStructsAtOffset(
        lwEncDevStruct,
        "pic_control.slice_control_offset",
        "pic_control.num_forced_slices_minus1",
        "lwenc_h264_slice_control_s",
        astStructs,
        start,
        finish,
        std::back_inserter(slices)
    );
    GetStructFields(slices.begin(), slices.end(), slicePrefix, outIt);

    ReadStructsAtOffset(
        lwEncDevStruct,
        "pic_control.me_control_offset",
        "pic_control.num_me_controls_minus1",
        "lwenc_h264_me_control_s",
        astStructs,
        start,
        finish,
        std::back_inserter(me)
    );
    GetStructFields(me.begin(), me.end(), mePrefix, outIt);

    ReadStructsAtOffset(
        lwEncDevStruct,
        "pic_control.md_control_offset",
        "pic_control.num_md_controls_minus1",
        "lwenc_h264_md_control_s",
        astStructs,
        start,
        finish,
        std::back_inserter(md)
    );
    GetStructFields(md.begin(), md.end(), mdPrefix, outIt);

    ReadStructsAtOffset(
        lwEncDevStruct,
        "pic_control.q_control_offset",
        "pic_control.num_q_controls_minus1",
        "lwenc_h264_quant_control_s",
        astStructs,
        start,
        finish,
        std::back_inserter(qCtrl)
    );
    GetStructFields(qCtrl.begin(), qCtrl.end(), qPrefix, outIt);

    ReadStructsAtOffset(
        lwEncDevStruct,
        "pic_control.wp_control_offset",
        "pic_control.num_wp_controls_minus1",
        "lwenc_pred_weight_table_s",
        astStructs,
        start,
        finish,
        std::back_inserter(wpCtrl)
    );
    GetStructFields(wpCtrl.begin(), wpCtrl.end(), wpPrefix, outIt);
}

template <typename AstCont, typename BinIt, typename OutputIterator>
void ReadH265DrvStructs
(
    AstCont &astStructs,
    const Struct &lwEncDevStruct,
    const std::string &slicePrefix,
    const std::string &mePrefix,
    const std::string &mdPrefix,
    const std::string &qPrefix,
    const std::string &wpPrefix,
    BinIt start,
    BinIt finish,
    OutputIterator outIt
)
{
    std::vector<Struct> slices;
    std::vector<Struct> me;
    std::vector<Struct> md;
    std::vector<Struct> qCtrl;
    std::vector<Struct> wpCtrl;

    ReadStructsAtOffset(
        lwEncDevStruct,
        "pic_control.slice_control_offset",
        "pic_control.num_forced_slices_minus1",
        "lwenc_h265_slice_control_s",
        astStructs,
        start,
        finish,
        std::back_inserter(slices)
    );
    GetStructFields(slices.begin(), slices.end(), slicePrefix, outIt);

    ReadStructsAtOffset(
        lwEncDevStruct,
        "pic_control.me_control_offset",
        "pic_control.num_me_controls_minus1",
        "lwenc_h264_me_control_s",
        astStructs,
        start,
        finish,
        std::back_inserter(me)
    );
    GetStructFields(me.begin(), me.end(), mePrefix, outIt);

    ReadStructsAtOffset(
        lwEncDevStruct,
        "pic_control.md_control_offset",
        "pic_control.num_md_controls_minus1",
        "lwenc_h265_md_control_s",
        astStructs,
        start,
        finish,
        std::back_inserter(md)
    );
    GetStructFields(md.begin(), md.end(), mdPrefix, outIt);

    ReadStructsAtOffset(
        lwEncDevStruct,
        "pic_control.q_control_offset",
        "pic_control.num_q_controls_minus1",
        "lwenc_h265_quant_control_s",
        astStructs,
        start,
        finish,
        std::back_inserter(qCtrl)
    );
    GetStructFields(qCtrl.begin(), qCtrl.end(), qPrefix, outIt);

    ReadStructsAtOffset(
        lwEncDevStruct,
        "pic_control.wp_control_offset",
        "pic_control.num_wp_controls_minus1",
        "lwenc_pred_weight_table_s",
        astStructs,
        start,
        finish,
        std::back_inserter(wpCtrl)
    );
    GetStructFields(wpCtrl.begin(), wpCtrl.end(), wpPrefix, outIt);
}

class FakeOuputIterator
{
public:
    using iterator_category = std::output_iterator_tag;
    using value_type = void;
    using difference_type = void;
    using pointer = void;
    using reference = void;

    template <typename Anything>
    FakeOuputIterator& operator=(Anything) { return *this; }
    FakeOuputIterator& operator*() { return *this; }
    FakeOuputIterator& operator++() { return *this; }
    FakeOuputIterator operator++(int) { return *this; }
};

void PrintH265OutIdx(const H265::Parser &parser)
{
    H265DPBManager dpb(static_cast<UINT08>(parser.GetMaxDecPicBuffering()));

    bool firstIter = true;
    for (auto picIt = parser.pic_begin(); parser.pic_end() != picIt; ++picIt)
    {
        int outIdx = dpb.GetSlotForNewPic(*picIt, FakeOuputIterator());
        if (!firstIter)
        {
            printf(", ");
        }
        printf("%d", outIdx);
        if (firstIter)
        {
            firstIter = false;
        }
    }
    printf("\n");
}

void CorrectRefLists(Struct &lwEncDevStruct, const H265::Parser &parser, unsigned frameNum)
{
    H265DPBManager dpb(static_cast<UINT08>(parser.GetMaxDecPicBuffering()));

    auto picIt = parser.pic_begin();
    for (unsigned count = 0; frameNum > count && parser.pic_end() != picIt; ++count, ++picIt)
    {
        dpb.GetSlotForNewPic(*picIt, FakeOuputIterator());
    }
    if (parser.pic_end() != picIt)
    {
        int l0[H265::maxNumRefPics] = {};
        int l1[H265::maxNumRefPics] = {};

        // NumPocTotalLwrr is the maximum for all slices and for all lists. The
        // real amount ref pictures is defined in the slice header for each
        // slice.
        for (size_t i = 0; picIt->NumPocTotalLwrr > i; ++i)
        {
            l0[i] = dpb.GetLwdecIdx(picIt->GetRefPicListTemp0()[i]);
            l1[i] = dpb.GetLwdecIdx(picIt->GetRefPicListTemp1()[i]);
        }
        if (0 < picIt->NumPocTotalLwrr)
        {
            for (size_t i = picIt->NumPocTotalLwrr; H265::maxNumRefPics > i; ++i)
            {
                l0[i] = l0[i - picIt->NumPocTotalLwrr];
                l1[i] = l1[i - picIt->NumPocTotalLwrr];
            }
        }

        auto &frontSlice = picIt->slices_front();

        size_t refCount = 0;
        if (!frontSlice.l0_empty())
        {
            for (auto l0it = frontSlice.l0_begin(); frontSlice.l0_end() != l0it; ++l0it, ++refCount)
            {
                lwEncDevStruct.SetField(
                    "pic_control.l0[" + std::to_string(refCount) + "]", 2 * l0[refCount]);
                lwEncDevStruct.SetField(
                    "pic_control.poc_entry_l0[" + std::to_string(refCount) + "]", l0[refCount]
                );
            }
        }
        for (size_t i = refCount; 8 > i; ++i)
        {
            lwEncDevStruct.SetField("pic_control.l0[" + std::to_string(i) + "]", -2);
            lwEncDevStruct.SetField("pic_control.poc_entry_l0[" + std::to_string(i) + "]", -1);
        }
        refCount = 0;
        if (!frontSlice.l1_empty())
        {
            for (auto l1it = frontSlice.l1_begin(); frontSlice.l1_end() != l1it; ++l1it, ++refCount)
            {
                lwEncDevStruct.SetField(
                    "pic_control.l1[" + std::to_string(refCount) + "]", 2 * l1[refCount]);
                lwEncDevStruct.SetField(
                    "pic_control.poc_entry_l1[" + std::to_string(refCount) + "]", l1[refCount]
                );
            }
        }
        for (size_t i = refCount; 8 > i; ++i)
        {
            lwEncDevStruct.SetField("pic_control.l1[" + std::to_string(i) + "]", -2);
            lwEncDevStruct.SetField("pic_control.poc_entry_l1[" + std::to_string(i) + "]", -1);
        }
    }
}

// The following code will print out a table like this:
//        # POC type|  l0|  l1| idx| DPB        | Output
//        0   0  IDR|   *|   *|   0|  *  *  *  *|  *
//        1   4    P|   0|   *|   1|  0  *  *  *|  *
//        2   2    b|   0|   4|   2|  0  4  *  *|  *
//        3   1    b|   0|   2|   3|  0  4  2  *|  *
//        4   3    b|   2|   4|   0|  0  4  2  1|  0
//        5   8    P|   4|   *|   3|  3  4  2  1|  3
//        6   6    b|   4|   8|   2|  3  4  2  8|  2
//        7   5    b|   4|   6|   0|  3  4  6  8|  0
//        8   7    b|   6|   8|   1|  5  4  6  8|  1
//        9  12    P|   8|   *|   0|  5  7  6  8|  0
//       10  10    b|   8|  12|   2| 12  7  6  8|  2
//       11   9    b|   8|  10|   1| 12  7 10  8|  1
//       12  11    b|  10|  12|   3| 12  9 10  8|  3
//       13  16    P|  12|   *|   1| 12  9 10 11|  1
//       14  14    b|  12|  16|   2| 12 16 10 11|  2
//       15  13    b|  12|  14|   3| 12 16 14 11|  3
//       16  15    b|  14|  16|   0| 12 16 14 13|  0
//       17  20    P|  16|   *|   3| 15 16 14 13|  3
//       18  18    b|  16|  20|   2| 15 16 14 20|  2
//       19  17    b|  16|  18|   0| 15 16 18 20|  0
//       20  19    b|  18|  20|   1| 17 16 18 20|  1
//       21  24    P|  20|   *|   0| 17 19 18 20|  0
//       22  22    b|  20|  24|   2| 24 19 18 20|  2
//       23  21    b|  20|  22|   1| 24 19 22 20|  1
//       24  23    b|  22|  24|   3| 24 21 22 20|  3
//       25  28    P|  24|   *|   1| 24 21 22 23|  1
//       26  26    b|  24|  28|   2| 24 28 22 23|  2
//       27  25    b|  24|  26|   3| 24 28 26 23|  3
//       28  27    b|  26|  28|   0| 24 28 26 25|  0
//       29  29    b|  28|  28|   3| 27 28 26 25|  3
// Remaining output pictures:   2  0  1  3
//
// #, POC and 'type' describe a picture to be decoded. # is its number in the
// decoding order, POC is Picture Order Count, i.e. the picture number in the
// displaying order. 'type' is either IDR (Instantaneous Decoding Refresh, i.e.
// it doesn't depend on anything previously decoded), P -- the picture depends
// on pictures located before it in the displaying order, and B -- the picture
// depends both on pictures located before and after it in the displaying order.
// Calling it a picture type is not technically correct, since a picture can
// have an arbitrary number of slices, each having its own type. However it's an
// accepted in the company slang, since most of the time a picture has just one
// type of slices. Lowercase p and b mean that these pictures are not used as a
// reference.
//
// 'l0' and 'l1' are the lists of POCs of pictures used as a reference to decode
// this picture.
//
// 'idx' is where the picture is going to be placed into DPB (Decoded Picture
// Buffer) after decoding.
//
// 'DPB' shows POCs of the pictures in the DPB before decoding this picture.
//
// 'Output' shows the indices of DPB slots that can be output before decoding
// this picture.
void PrintDPBInfo(const H265::Parser &parser)
{
    H265DPBManager dpb(static_cast<UINT08>(parser.GetMaxDecPicBuffering()));

    typedef std::vector<size_t> OutSurfIDsCont;
    typedef std::map<size_t, OutSurfIDsCont> OutputPicsMap;

    OutputPicsMap outputPics;
    OutSurfIDsCont lastOutPics;
    std::vector<unsigned> outIndices;

    printf("Reference lists, DPB and output order of to be decoded pictures:\n");
    for (auto picIt = parser.pic_begin(); parser.pic_end() != picIt; ++picIt)
    {
        H265::PicIdx lwrPicIdx = picIt->GetPicIdx();
        auto outIdx = dpb.GetSlotForNewPic(*picIt, std::back_inserter(outputPics[lwrPicIdx]));
        outIndices.push_back(static_cast<unsigned>(outIdx));
    }
    dpb.Clear(std::back_inserter(lastOutPics));

    size_t list0MaxSize = 0;
    size_t list1MaxSize = 0;
    size_t outPicsMax = 0;
    for (auto picIt = parser.pic_begin(); parser.pic_end() != picIt; ++picIt)
    {
        size_t list0Size = picIt->slices_front().GetL0Size();
        size_t list1Size = picIt->slices_front().GetL1Size();

        list0MaxSize = std::max(list0MaxSize, list0Size);
        list1MaxSize = std::max(list1MaxSize, list1Size);

        const OutSurfIDsCont &outPics = outputPics[picIt->GetPicIdx()];
        outPicsMax = std::max(outPicsMax, outPics.size());
    }
    printf("       # POC type|");
    printf("%*s", static_cast<int>(list0MaxSize * 4), "l0");
    printf("|");
    printf("%*s", static_cast<int>(list1MaxSize * 4), "l1");
    printf("| idx|");
    if (0 < outPicsMax)
    {
        printf(
            "%*s",
            -static_cast<int>(parser.GetMaxDecPicBuffering() * 3),
            " DPB"
        );
        printf("| Output");
    }
    else
    {
        printf(" DPB");
    }

    size_t count = 0;
    printf("\n");
    for (auto picIt = parser.pic_begin(); parser.pic_end() != picIt; ++picIt, ++count)
    {
        auto &frontSlice = picIt->slices_front();

        printf(
            "    %4d%4d%5s%s",
            static_cast<int>(count),
            static_cast<int>(picIt->PicOrderCntVal),
            frontSlice.GetSliceTypeStr(),
            "|"
        );

        H265::Slice::ref_iterator l0it;
        H265::Slice::ref_iterator l1it;
        
        // print reference list 0
        size_t refCount = 0;
        if (!frontSlice.l0_empty())
        {
            for (l0it = frontSlice.l0_begin(); frontSlice.l0_end() != l0it; ++l0it, ++refCount)
            {
                printf("%4d", parser.GetPicture(*l0it)->PicOrderCntVal);
            }
        }
        for (size_t i = refCount; list0MaxSize > i; ++i)
        {
            printf("%4s", "*");
        }
        printf("|");
        
        // print reference list 1
        refCount = 0;
        if (!frontSlice.l1_empty())
        {
            for (l1it = frontSlice.l1_begin();
                 frontSlice.l1_end() != l1it;
                 ++l1it, ++refCount)
            {
                printf("%4d", parser.GetPicture(*l1it)->PicOrderCntVal);
            }
        }
        for (size_t i = refCount; list1MaxSize > i; ++i)
        {
            printf("%4s", "*");
        }
        printf("|");

        // current picture index in DPB
        printf("%4u|", outIndices[count]);

        std::vector<int> lwrDpb(parser.GetMaxDecPicBuffering(), -1);
        for (size_t i = 0; count > i; ++i)
        {
            lwrDpb[outIndices[i]] = parser.GetPicture(i)->PicOrderCntVal;
        }
        for (size_t i = 0; parser.GetMaxDecPicBuffering() > i; ++i)
        {
            if (0 > lwrDpb[i])
            {
                printf("%3s", "*");
            }
            else
            {
                printf("%3u", lwrDpb[i]);
            }
        }
        if (0 < outPicsMax)
        {
            printf("|");
            // output pictures
            size_t outPicsCount = 0;
            const OutSurfIDsCont &outPics = outputPics[picIt->GetPicIdx()];
            if (!outPics.empty())
            {
                OutSurfIDsCont::const_iterator it;
                for (it = outPics.begin(); outPics.end() != it; ++it, ++outPicsCount)
                {
                    printf("%3d", static_cast<int>(*it));
                }
            }
            for (size_t i = outPicsCount; outPicsMax > i; ++i)
            {
                printf("%3s", "*");
            }
        }
        printf("\n");
    }
    printf("Remaining output pictures:\n");
    if (!lastOutPics.empty())
    {
        OutSurfIDsCont::const_iterator it;
        for (it = lastOutPics.begin(); lastOutPics.end() != it; ++it)
        {
            printf("%3d", static_cast<int>(*it));
        }
    }
    else
    {
        printf("none");
    }
    printf("\n");
}

int main(int argc, char* argv[])
{
    po::positional_options_description positional;
    po::options_description cmdLineOptions;
    po::variables_map vm;

    std::string inFileName;
    std::string compareFileName;
    std::string outFileName;
    std::string lwEncDrvFileName;
    std::string videoFileName;
    std::string structName;
    std::string prefixName;
    std::string slicePrefix;
    std::string mePrefix;
    std::string mdPrefix;
    std::string qPrefix;
    std::string wpPrefix;

    unsigned frameNum = 0;
    unsigned compFrameNum = 0;
    
    std::unique_ptr<FILE, decltype(&fclose)> inFile(nullptr, &fclose);
    std::unique_ptr<FILE, decltype(&fclose)> compareFile(nullptr, &fclose);
    std::unique_ptr<FILE, decltype(&fclose)> outFile(nullptr, &fclose);
    std::unique_ptr<FILE, decltype(&fclose)> lwEncDrvFile(nullptr, &fclose);
    std::unique_ptr<FILE, decltype(&fclose)> videoFile(nullptr, &fclose);

    cmdLineOptions.add_options()
        (
            "input",
            po::value<std::string>(&inFileName)->required(),
            "input file"
        )
        (
            "compare-with",
            po::value<std::string>(&compareFileName),
            "prefix for fields output, i.e. the name of the root variable"
        )
        (
            "output,o",
            po::value<std::string>(&outFileName),
            "output file"
        )
        (
            "lwenc_drv",
            po::value<std::string>(&lwEncDrvFileName)->required(),
            "structures definition file"
        )
        (
            "struct",
            po::value<std::string>(&structName)->required(),
            "driver structure name"
        )
        (
            "prefix",
            po::value<std::string>(&prefixName)->default_value("picSetup"s),
            "prefix for fields output, i.e. the name of the root variable"
        )
        (
            "slice_prefix",
            po::value<std::string>(&slicePrefix)->default_value("sliceControl"s),
            "prefix for slice_control"
        )
        (
            "me_prefix",
            po::value<std::string>(&mePrefix)->default_value("meControl"s),
            "prefix for me_control"
        )
        (
            "md_prefix",
            po::value<std::string>(&mdPrefix)->default_value("mdControl"s),
            "prefix for md_control"
        )
        (
            "q_prefix",
            po::value<std::string>(&qPrefix)->default_value("qControl"s),
            "prefix for q_control"
        )
        (
            "wp_prefix",
            po::value<std::string>(&wpPrefix)->default_value("wpControl"s),
            "prefix for wp_control"
        )
        (
            "ref-from",
            po::value<std::string>(&videoFileName),
            "set values for l0 and l1 from a video file, print out DPB"
        )
        (
            "outidx",
            "if ref-from is specified, print out output indexes of pictures"
        )
        (
            "dpb",
            "if ref-from is specified, print out DPB and ref lists for all pictures"
        )
      ;
    positional.add("input", 1);

    try
    {
        po::store(po::command_line_parser(argc, argv).
                  positional(positional).
                  options(cmdLineOptions).run(), vm);
        po::notify(vm);
    }
    catch (po::error &x)
    {
        fprintf(stderr, "Command line error: %s\n", x.what());
        return 1;
    }

    try
    {
        if (!fs::exists(inFileName))
        {
            fprintf(stderr, "Input file %s doesn't exist\n", inFileName.c_str());
            return 1;
        }
        inFile.reset(fopen(inFileName.c_str(), "rb"));
        if (!inFile) throw boost::system::system_error(errno, boost::system::system_category());

        if (0 != vm.count("compare-with"))
        {
            if (!fs::exists(compareFileName))
            {
                fprintf(stderr, "Input file %s doesn't exist\n", compareFileName.c_str());
                return 1;
            }
            compareFile.reset(fopen(compareFileName.c_str(), "rb"));
            if (!compareFile) throw boost::system::system_error(errno, boost::system::system_category());
        }

        if (!fs::exists(lwEncDrvFileName))
        {
            fprintf(stderr, "Input file %s doesn't exist\n", lwEncDrvFileName.c_str());
            return 1;
        }
        lwEncDrvFile.reset(fopen(lwEncDrvFileName.c_str(), "rb"));
        if (!lwEncDrvFile) throw boost::system::system_error(errno, boost::system::system_category());

        if (0 != vm.count("ref-from"))
        {
            if (!fs::exists(videoFileName))
            {
                fprintf(stderr, "Input file %s doesn't exist\n", inFileName.c_str());
                return 1;
            }
            videoFile.reset(fopen(videoFileName.c_str(), "rb"));
            if (!videoFile) throw boost::system::system_error(errno, boost::system::system_category());
            auto fn = fs::path(inFileName).filename().string();
            qi::parse(fn.cbegin(), fn.cend(), "DRVPIC_" >> qi::uint_ >> ".bin", frameNum);
            if (0 != vm.count("compare-with"))
            {
                auto fn = fs::path(compareFileName).filename().string();
                qi::parse(fn.cbegin(), fn.cend(), "DRVPIC_" >> qi::uint_ >> ".bin", compFrameNum);
            }
        }

        if (0 == vm.count("output"))
        {
            outFile = decltype(outFile)(stdout, [](FILE *)->int { return 0; });
        }
        else
        {
            outFile.reset(fopen(outFileName.c_str(), "w"));
            if (!outFile) throw boost::system::system_error(errno, boost::system::system_category());
        }

        std::string lwEncDrv;
        while (!feof(lwEncDrvFile.get()))
        {
            std::array<char, 4096> buf;
            size_t wasRead = fread(&buf[0], 1, boost::size(buf), lwEncDrvFile.get());
            lwEncDrv.append(&buf[0], &buf[wasRead]);
        }
        lwEncDrvFile.reset();

        std::vector<boost::uint8_t> lwEncDrvBin;
        while (!feof(inFile.get()))
        {
            std::array<boost::uint8_t, 4096> buf;
            size_t wasRead = fread(&buf[0], 1, boost::size(buf), inFile.get());
            lwEncDrvBin.insert(lwEncDrvBin.end(), &buf[0], &buf[wasRead]);
        }
        inFile.reset();

        std::vector<boost::uint8_t> compBin;
        if (0 != vm.count("compare-with"))
        {
            while (!feof(compareFile.get()))
            {
                std::array<boost::uint8_t, 4096> buf;
                size_t wasRead = fread(&buf[0], 1, boost::size(buf), compareFile.get());
                compBin.insert(compBin.end(), &buf[0], &buf[wasRead]);
            }
            compareFile.reset();
        }

        std::vector<boost::uint8_t> videoBin;
        if (0 != vm.count("ref-from"))
        {
            while (!feof(videoFile.get()))
            {
                std::array<boost::uint8_t, 4096> buf;
                size_t wasRead = fread(&buf[0], 1, boost::size(buf), videoFile.get());
                videoBin.insert(videoBin.end(), &buf[0], &buf[wasRead]);
            }
            videoFile.reset();
        }

        WaveContext ctx(lwEncDrv.cbegin(), lwEncDrv.cend());
        ctx.set_language(boost::wave::enable_long_long(ctx.get_language(), true));
        ctx.set_language(enable_emit_line_directives(ctx.get_language(), false));
        ctx.set_language(enable_insert_whitespace(ctx.get_language(), false));
        ctx.set_language(enable_single_line(ctx.get_language(), true));

        const CStructsGrammar<WaveContext::iterator_type> grammar;
        const WhiteSpace<WaveContext::iterator_type> whiteSpace;

        AST::Structs lwEncDevAstStructs;
        if (qi::phrase_parse(ctx.begin(), ctx.end(), grammar, whiteSpace, lwEncDevAstStructs))
        {
            std::cout << "Parsing succeeded" << std::endl;
        }
        else
        {
            std::cout << "Parsing failed" << std::endl;
            return 1;
        }

        std::vector<std::tuple<std::string, std::string>> nameValuePairs;

        auto &byName = lwEncDevAstStructs.get<AST::by_name>();
        auto it = byName.find(structName);
        if (byName.end() != it)
        {
            H265::Parser parser;

            Struct lwEncDevStruct;
            UpdateAlignment(byName, it);
            auto is = MakeInBitStream(lwEncDrvBin.cbegin(), lwEncDrvBin.cend());
            ReadStructure(byName, it, is, lwEncDevStruct);
            if ("lwenc_h265_drv_pic_setup_s" == structName && 0 != vm.count("ref-from"))
            {
                if (OK != parser.ParseStream(videoBin.cbegin(), videoBin.cend()))
                {
                    fprintf(stderr, "Error parsing %s\n", videoFileName.c_str());
                    return 1;
                }
                if (0 != vm.count("dpb"))
                {
                    PrintDPBInfo(parser);
                }

                if (0 != vm.count("outidx"))
                {
                    printf("Insert this into streamDescriptions.outPictureIndex:\n");
                    PrintH265OutIdx(parser);
                }

                CorrectRefLists(lwEncDevStruct, parser, frameNum);
            }
            GetStructFields(lwEncDevStruct, prefixName, std::back_inserter(nameValuePairs));

            if ("lwenc_h264_drv_pic_setup_s" == structName)
            {
                ReadH264DrvStructs(
                    byName,
                    lwEncDevStruct,
                    slicePrefix,
                    mePrefix,
                    mdPrefix,
                    qPrefix,
                    wpPrefix,
                    lwEncDrvBin.cbegin(),
                    lwEncDrvBin.cend(),
                    std::back_inserter(nameValuePairs)
                );
            }
            else if ("lwenc_h265_drv_pic_setup_s" == structName)
            {
                ReadH265DrvStructs(
                    byName,
                    lwEncDevStruct,
                    slicePrefix,
                    mePrefix,
                    mdPrefix,
                    qPrefix,
                    wpPrefix,
                    lwEncDrvBin.cbegin(),
                    lwEncDrvBin.cend(),
                    std::back_inserter(nameValuePairs)
                );
            }

            if (0 != vm.count("compare-with"))
            {
                std::vector<std::tuple<std::string, std::string>> comp;

                Struct compStruct;
                UpdateAlignment(byName, it);
                auto is = MakeInBitStream(compBin.cbegin(), compBin.cend());
                ReadStructure(byName, it, is, compStruct);
                if ("lwenc_h265_drv_pic_setup_s" == structName && 0 != vm.count("ref-from"))
                {
                    CorrectRefLists(compStruct, parser, compFrameNum);
                }
                GetStructFields(compStruct, prefixName, std::back_inserter(comp));

                if ("lwenc_h264_drv_pic_setup_s" == structName)
                {
                    ReadH264DrvStructs(
                        byName,
                        compStruct,
                        slicePrefix,
                        mePrefix,
                        mdPrefix,
                        qPrefix,
                        wpPrefix,
                        compBin.cbegin(),
                        compBin.cend(),
                        std::back_inserter(comp)
                    );
                }
                else if ("lwenc_h265_drv_pic_setup_s" == structName)
                {
                    ReadH265DrvStructs(
                        byName,
                        compStruct,
                        slicePrefix,
                        mePrefix,
                        mdPrefix,
                        qPrefix,
                        wpPrefix,
                        compBin.cbegin(),
                        compBin.cend(),
                        std::back_inserter(comp)
                    );
                }

                typedef std::vector<std::tuple<std::string, std::string>>::const_iterator It;
                std::vector<std::tuple<It, It>> intersect;
                for (auto it1 = comp.cbegin(); comp.cend() != it1; ++it1)
                {
                    for (auto it2 = nameValuePairs.cbegin(); nameValuePairs.cend() != it2; ++it2)
                    {
                        if (std::get<0>(*it1) == std::get<0>(*it2))
                        {
                            intersect.push_back(std::make_tuple(it1, it2));
                        }
                    }
                }
                std::vector<std::tuple<std::string, std::string>> newList;
                auto it1 = comp.cbegin();
                auto it2 = nameValuePairs.cbegin();
                for (auto it = intersect.cbegin(); intersect.cend() != it; ++it, ++it1, ++it2)
                {
                    for (; std::get<0>(*it) > it1; ++it1)
                    {   // deleted
                        newList.push_back(make_tuple(std::get<0>(*it1), "0"s));
                    }
                    for (; std::get<1>(*it) > it2; ++it2)
                    {   // added
                        newList.push_back(make_tuple(std::get<0>(*it2), std::get<1>(*it2)));
                    }
                    if (std::get<1>(*it1) != std::get<1>(*it2))
                    {   // changed
                        newList.push_back(make_tuple(std::get<0>(*it2), std::get<1>(*it2)));
                    }
                }
                nameValuePairs = newList;
            }

            if (!nameValuePairs.empty())
            {
                auto longestFieldIt = boost::max_element(
                    nameValuePairs,
                    [](const auto &t1, const auto &t2)
                    {
                        return std::get<0>(t1).size() < std::get<0>(t2).size();
                    }
                );
                auto longestField = std::get<0>(*longestFieldIt).size();
                for (const auto &t : nameValuePairs)
                {
                    fprintf(
                        outFile.get(),
                        "    %-*s = %s;\n",
                        static_cast<int>(longestField),
                        std::get<0>(t).c_str(),
                        std::get<1>(t).c_str()
                    );
                }
            }
            else
            {
                printf("All values of %s are zero\n", structName.c_str());
            }
        }
        else
        {
            throw std::ilwalid_argument("Structure " + structName + " is not defined");
        }
    }
    catch (std::exception &x)
    {
        fprintf(stderr, "%s\n", x.what());
        return 1;
    }

    return 0;
}
