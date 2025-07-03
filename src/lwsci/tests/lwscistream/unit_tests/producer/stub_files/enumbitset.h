//! \file
//! \brief Simple enum-based bitset utility
//!
//! \copyright
//! Copyright (c) 2019-2021 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef ENUMBITSET_H
#define ENUMBITSET_H

namespace LwSciStream {

//! \brief A template class which provides utility functions to set and get
//!   the bits of given BitType based on the given EnumType.
//!
//!   It is used by all the units for validation purpose. On entry point of
//!   every APIBlockInterface, SrcBlockInterface and DstBlockInterface
//!   interfaces, to validate the state of the stream
//!   (ValidateCtrl::Complete, ValidateCtrl::CompleteQueried) and input
//!   parameters such as srcIndex (source connection index), dstIndex
//!   (destination connection index), blocks create an instance of EnumBitSet
//!   class and initialize it with the ValidateCtrl enum values of the items
//!   to be validated. Then this instance will be passed to
//!   Block::validateWithEvent() or Block::validateWithError() utility
//!   interfaces which will get the items to be validated from the EnumBitSet
//!   instance and do the required validation.
//!
//! \tparam EnumType: Enum type, like Block::ValidateCtrl,
//!         each value of which corresponds to one bit.
//!
//! \tparam BitType: uint32_t or uint64_t
//!
//! \implements{19765842}
LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A14_1_1), "LwSciStream-ADV-AUTOSARC++14-002")
template <class EnumType, class BitType=uint32_t>
class EnumBitset final
{
public:
    //! \brief Constructs an instance of EnumBitset class
    //!        and initializes the data fields.
    EnumBitset(void) noexcept = default;

    //! \brief Constructs an instance of EnumBitset class
    //!        and copies the given EnumBitset reference's
    //!        data fields to it.
    //!
    //! \param [in] r: EnumBitset reference object.
    //!
    //! \return None
    //!
    //! \implements{19775574}
    EnumBitset(const EnumBitset& r) noexcept : bits(r.bits) {};

    // default copy operator
    auto operator=(const EnumBitset& r) noexcept -> EnumBitset& = delete;

    LWCOV_ALLOWLIST_BEGIN(LWCOV_CERTCPP(DCL51_CPP), "LwSciStream-ADV-CERTCPP-001")
    //! \brief Sets the N-th bit where N is the value
    //!        of given EnumType.
    //!
    //! \param [in] v: EnumType
    //!
    //! \return void.
    //!
    //! \implements{19775580}
    void set(EnumType const v) noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(INT34_C), "LwSciStream-ADV-CERTC-004")
        bits |= (OneInBitType << static_cast<uint32_t>(v));
    };
    LWCOV_ALLOWLIST_END(LWCOV_CERTCPP(DCL51_CPP))

    //! \brief Returns the boolean value of N-th bit
    //!        where N is the value of given EnumType.
    //!
    //! \param [in] v: EnumType.
    //!
    //! \return bool
    //!  * true: if the N-th bit is set.
    //!  * false: otherwise.
    //!
    //! \implements{19775583}
    bool operator[](EnumType const v) const noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_CERTC(INT34_C), "LwSciStream-ADV-CERTC-004")
        return (ZeroInBitType != ((OneInBitType << static_cast<uint32_t>(v)) & bits));
    };

    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    EnumBitset(EnumBitset&& r) = delete;
    LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A8_4_8), "Bug 2745784")
    auto operator=(EnumBitset&& r) -> EnumBitset& = delete;

    //! \brief Default destructor
    ~EnumBitset(void) = default;

private:
    LWCOV_ALLOWLIST_BEGIN(LWCOV_AUTOSAR(A8_5_2), "Bug 2738296")
    // Not exported to doxygen. See Readme-doxygen-criteria.txt for details.
    //! \cond
    //! \brief Constant values used within EnumBitset.
    static constexpr BitType    ZeroInBitType { static_cast<BitType>(0) };
    // Constant values used within EnumBitset.
    static constexpr BitType    OneInBitType  { static_cast<BitType>(1) };
    LWCOV_ALLOWLIST_END(LWCOV_AUTOSAR(A8_5_2))
    //! \endcond

    //! \brief A bitset tracks whether the bits corresponding to EnumType(s)
    //!   are set. It is initialized when an EnumBitset instance is created
    //!   and it can be updated with EnumBitset::operator=() and
    //!   EnumBitset::set() interfaces.
    BitType                     bits { ZeroInBitType };
};

} // namespace LwSciStream

#endif // ENUMBITSET_H
