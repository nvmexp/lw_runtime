//! \file
//! \brief LwSciStream branch tracking for multicast.
//!
//! \copyright
//! Copyright (c) 2021-2022 LWPU Corporation. All rights reserved.
//!
//! LWPU Corporation and its licensors retain all intellectual property and
//! proprietary rights in and to this software, related documentation and any
//! modifications thereto. Any use, reproduction, disclosure or distribution
//! of this software and related documentation without an express license
//! agreement from LWPU Corporation is strictly prohibited.

#ifndef BRANCH_H
#define BRANCH_H

#include <cstdint>
#include <bitset>
#include "lwscistream_common.h"

namespace LwSciStream {

//! \brief Simple object used by multicast to keep track of which
//!   downstream branches have provided any given piece of information,
//!   so that we can ensure each one does so exactly once before the
//!   information is sent upstream.
//!
//! If support is ever added for multiple producers, this same object
//!   will be used to track information flowing downstream through the
//!   block that combines it.
//!
//! This object does not provide thread safety, and the owner is
//!   expected to provide protection through a mutex or other means.
//!   The intended usage pattern is:
//!   * Mutex lock (or other thread-safety mechanism)
//!   * BranchTrack::set(index)
//!   * Set information for index
//!   * BranchTrack::done()
//!   * Mutex unlock
//!   * If done() call returned true, pass complete info onwards.
class BranchTrack final
{
public:
    //! \brief Constructor
    //!
    //! \param [in] paramBranchCount: Number of branches.
    explicit BranchTrack(size_t const paramBranchCount) noexcept :
        branchCount(paramBranchCount),
        branchBits(),
        branchDone(false)
    {
        assert(branchBits.size() >= branchCount);
    };

    //! \brief Default destructor.
    ~BranchTrack(void) noexcept                           = default;

    // Other operations not needed
    BranchTrack(BranchTrack const&) noexcept              = delete;
    BranchTrack(BranchTrack&&) noexcept                   = delete;
    BranchTrack& operator=(BranchTrack const&) & noexcept = delete;
    BranchTrack& operator=(BranchTrack&&) & noexcept      = delete;

    //! \brief Indicate information for indexed branch is being set.
    //!
    //! \param [in] index: Index of branch being set.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! - LwSciError_Success: Operation completed successfully.
    //! - LwSciError_IndexOutOfRange: Index is too large.
    //! - LwSciError_AlreadyDone: The branch was already set.
    LwSciError set(size_t const index) noexcept
    {
        if (branchCount <= index) {
            return LwSciError_IndexOutOfRange;
        }
        bool const val { branchBits[index] };
        branchBits[index] = true;
        return val ? LwSciError_AlreadyDone : LwSciError_Success;
    };

    //! \brief Returns whether or not information from branch already
    //!  received.
    //!
    //! \param [in] index: Index of branch.
    //!
    //! \return bool,
    //! - true: If information from branch already received
    //! - false: Otherwise.
    bool get(size_t const index) const noexcept
    {
        assert (index <= branchCount);
        return (branchBits[index]);
    };

    //! \brief Returns the status of all branches
    //!
    //! \return std::bitset
    std::bitset<MAX_DST_CONNECTIONS> getAll(void) const noexcept
    {
        return branchBits;
    };

    //! \brief Checks whether all branches are finished, and if the first
    //!   caller, takes responsibility for sending info onwards.
    //!
    //! \param [in] mask: bitset value
    //!
    //! \return bool, whether the caller should handle the completed info.
    bool done(std::bitset<MAX_DST_CONNECTIONS> const mask=
        std::bitset<MAX_DST_CONNECTIONS>{}) noexcept
    {
        // Check if already done. Only the first caller receives true.
        if (branchDone) {
            return false;
        }

        // Check all bits
        std::bitset<MAX_DST_CONNECTIONS> const bits { branchBits | mask };

        branchDone = (bits.count() == branchCount);
        return branchDone;
    };

    //! \brief Resets the tracking to be used again.
    //!   Lwrrently only relevant for runtime information (i.e. fences),
    //!   but might be needed for setup someday if we support dynamic
    //!   configuration.
    //!
    //! \return void
    void reset(void) noexcept
    {
        branchDone = false;
        branchBits.reset();
    };

private:
    //! \brief Number of branches to track.
    //!   Initialized during construction to input value.
    size_t const                        branchCount;

    //! \brief Bits tracking which branches are set.
    //!   Initialized during construction to all unset.
    // TODO: A dynamically sized bitset (e.g. boost::dynamic_bitset) would
    //       be prefereable, but std c++ does not provide one.
    std::bitset<MAX_DST_CONNECTIONS>    branchBits;

    //! \brief Tracks whether info is complete.
    //!   Initialized during construction to false.
    bool                                branchDone;
};

//! \brief Simple object used by multicast to keep track of the range
//!   of consumer indices accessible through each of its downstream
//!   branches, in order to collate information coming from them.
//!
//! If support is ever added for multiple producers, this same object
//!   will be used to track producer index ranges in the block that
//!   combines them.
//!
//! No thread-safety is provided. When setting up the object, the owner
//!   is expected to provide protection. After that, queries don't modify
//!   the object so are thread-safe.
class BranchMap final
{
public:
    //! \brief Representation for a range of endpoints within a list
    using Range = struct {
        //! \brief Number of endpoints in the range.
        size_t count;
        //! \brief Start of the range within the list.
        size_t start;
    };

    //! \brief Constructor for branch map
    //!
    //! \param [in] paramConnCount: Number of connections mapped.
    explicit BranchMap(size_t const paramConnCount) noexcept;
    //! \brief Default destructor.
    ~BranchMap(void) noexcept                         = default;

    BranchMap(BranchMap const&) noexcept              = delete;
    BranchMap(BranchMap&&) noexcept                   = delete;
    BranchMap& operator=(BranchMap const&) & noexcept = delete;
    BranchMap& operator=(BranchMap&&) & noexcept      = delete;

    //! \brief Retrieve any initialization failures.
    //!
    //! return LwSciError, any failure during constructor.
    //! * LwSciError_Success: Construction was successful.
    //! * LwSciError_InsufficientMemory: Unable to allocate vectors.
    LwSciError initErrorGet(void) const noexcept
    {
        return initError;
    }

    //! \brief Set up indexed connection if not already done, adjusting
    //!   the start of all other ranges.
    //!
    //! \param [in] index: Index of connection.
    //! \param [in] count: Number of endpoints accessible through connection.
    //!
    //! \return LwSciError, the completion code of this operation.
    //! * LwSciError_Success: Connection mapped successfully.
    //! * LwSciError_Overflow: Count is too large or would cause total count
    //!   to be too large.
    //! * Any error returned by TrackBranch::set().
    LwSciError set(
        size_t const index,
        size_t const count) noexcept;

    //! \brief Check whether all branches have set their info, and if so
    //!   take responsibility for passing along.
    //!
    //! \return bool, whether the caller should handle the completed info.
    inline bool done(void) noexcept
    {
        return connTrack.done();
    }

    //! \brief Query range for a connection.
    //!
    //! \param [in] index: Index of connection to query.
    //!
    //! \return The range for the connection, or an empty range if the
    //!   index is invalid.
    inline Range get(
        size_t const index) const noexcept
    {
        LWCOV_ALLOWLIST_LINE(LWCOV_AUTOSAR(A5_1_1), "TID-300")
        return (connRange.size() > index) ? connRange[index] : emptyRange;
    };

private:
    //! \brief Empty range used for initialization and errors.
    static constexpr Range  emptyRange { 0U, 0U };

    //! \brief Tracker of which connections have finished setup.
    //!   Initialized at construction with no connections set.
    BranchTrack             connTrack;

    //! \brief Vector of ranges for each connection.
    //!   Intialized at construction with size equal to the number of
    //!   connections and all entries with zero count and start.
    std::vector<Range>      connRange;

    //! \brief Any error encountered during construction.
    //!   Initialized at construction based on success of setup.
    LwSciError              initError;
};

} // namespace LwSciStream
#endif // BRANCH_H
