/***************************************************************************************************
 * Copyright (c) 2011-2019, LWPU CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LWPU CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <xmma/xmma.h>

#include <xmma/layout.h>
#include <xmma/utils.h>
#include <xmma/numeric_types.h>

namespace xmma {

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// E X T E N D E D   A P I   F O R   L D G
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< int BYTES_PER_LDG >
struct Fragment_ldg {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_ldg<1> {
    template< typename Fragment >
    static inline __device__ void ldg(Fragment &f,
                                      int ii,
                                      const void *ptr,
                                      uint64_t mem_desc = MEM_DESC_DEFAULT) {
        uint8_t tmp;
        xmma::ldg(tmp, ptr, mem_desc);
        f.u8(ii) = tmp;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_ldg<2> {
    template< typename Fragment >
    static inline __device__ void ldg(Fragment &f,
                                      int ii,
                                      const void *ptr,
                                      uint64_t mem_desc = MEM_DESC_DEFAULT) {
        uint16_t tmp;
        xmma::ldg(tmp, ptr, mem_desc);
        f.u16(ii) = tmp;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_ldg<4> {
    template< typename Fragment >
    static inline __device__ void ldg(Fragment &f,
                                      int ii,
                                      const void *ptr,
                                      uint64_t mem_desc = MEM_DESC_DEFAULT) {
        uint32_t tmp;
        xmma::ldg(tmp, ptr, mem_desc);
        f.reg(ii) = tmp;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_ldg<8> {
    template< typename Fragment >
    static inline __device__ void ldg(Fragment &f,
                                      int ii,
                                      const void *ptr,
                                      uint64_t mem_desc = MEM_DESC_DEFAULT) {
        uint2 tmp;
        xmma::ldg(tmp, ptr, mem_desc);
        f.reg(2*ii+0) = tmp.x;
        f.reg(2*ii+1) = tmp.y;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_ldg<16> {
    template< typename Fragment >
    static inline __device__ void ldg(Fragment &f,
                                      int ii,
                                      const void *ptr,
                                      uint64_t mem_desc = MEM_DESC_DEFAULT) {
        uint4 tmp;
        xmma::ldg(tmp, ptr, mem_desc);
        f.reg(4*ii+0) = tmp.x;
        f.reg(4*ii+1) = tmp.y;
        f.reg(4*ii+2) = tmp.z;
        f.reg(4*ii+3) = tmp.w;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_ldg<32> {
    template< typename Fragment >
    static inline __device__ void ldg(Fragment &f,
                                      int ii,
                                      const void *ptr,
                                      uint64_t mem_desc = MEM_DESC_DEFAULT) {
        uint4 tmp0;
        xmma::ldg(tmp0, ptr, mem_desc);
        f.reg(8*ii+0) = tmp0.x;
        f.reg(8*ii+1) = tmp0.y;
        f.reg(8*ii+2) = tmp0.z;
        f.reg(8*ii+3) = tmp0.w;

        uint4 tmp1;
        xmma::ldg(tmp1, static_cast<const char*>(ptr)+sizeof(uint4), mem_desc);
        f.reg(8*ii+4) = tmp1.x;
        f.reg(8*ii+5) = tmp1.y;
        f.reg(8*ii+6) = tmp1.z;
        f.reg(8*ii+7) = tmp1.w;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// E X T E N D E D   A P I   F O R   L D S
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< int BYTES_PER_LDS >
struct Fragment_lds {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_lds<2> {
    template< typename Fragment >
    static inline __device__ void lds(Fragment &f, int ii, uint32_t ptr) {
        uint16_t tmp;
        xmma::lds(tmp, ptr);
        f.u16(ii) = tmp;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_lds<4> {
    template< typename Fragment >
    static inline __device__ void lds(Fragment &f, int ii, uint32_t ptr) {
        uint32_t tmp;
        xmma::lds(tmp, ptr);
        f.reg(ii) = tmp;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_lds<8> {
    template< typename Fragment >
    static inline __device__ void lds(Fragment &f, int ii, uint32_t ptr) {
        uint2 tmp;
        xmma::lds(tmp, ptr);
        f.reg(2*ii+0) = tmp.x;
        f.reg(2*ii+1) = tmp.y;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_lds<16> {
    template< typename Fragment >
    static inline __device__ void lds(Fragment &f, int ii, uint32_t ptr) {
        uint4 tmp;
        xmma::lds(tmp, ptr);
        f.reg(4*ii+0) = tmp.x;
        f.reg(4*ii+1) = tmp.y;
        f.reg(4*ii+2) = tmp.z;
        f.reg(4*ii+3) = tmp.w;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
template<>
struct Fragment_lds<32> {
    template< typename Fragment >
    static inline __device__ void lds(Fragment &f, int ii, uint32_t ptr) {
        uint4 tmp0;
        xmma::lds(tmp0, ptr);
        f.reg(8*ii+0) = tmp0.x;
        f.reg(8*ii+1) = tmp0.y;
        f.reg(8*ii+2) = tmp0.z;
        f.reg(8*ii+3) = tmp0.w;

        uint4 tmp1;
        xmma::lds(tmp1, ptr+sizeof(uint4));
        f.reg(8*ii+4) = tmp1.x;
        f.reg(8*ii+5) = tmp1.y;
        f.reg(8*ii+6) = tmp1.z;
        f.reg(8*ii+7) = tmp1.w;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// E X T E N D E D   A P I   F O R   S T G
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< int BYTES_PER_STG >
struct Fragment_stg {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_stg<1> {
    template< typename Fragment >
    static inline __device__ void stg(void *ptr,
                                      const Fragment &f,
                                      int ii,
                                      uint64_t mem_desc = MEM_DESC_DEFAULT) {
        xmma::stg(ptr, f.u8(ii), mem_desc);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_stg<2> {
    template< typename Fragment >
    static inline __device__ void stg(void *ptr,
                                      const Fragment &f,
                                      int ii,
                                      uint64_t mem_desc = MEM_DESC_DEFAULT) {
        xmma::stg(ptr, f.u16(ii), mem_desc);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_stg<4> {
    template< typename Fragment >
    static inline __device__ void stg(void *ptr,
                                      const Fragment &f,
                                      int ii,
                                      uint64_t mem_desc = MEM_DESC_DEFAULT) {
        xmma::stg(ptr, f.reg(ii), mem_desc);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_stg<8> {
    template< typename Fragment >
    static inline __device__ void stg(void *ptr,
                                      const Fragment &f,
                                      int ii,
                                      uint64_t mem_desc = MEM_DESC_DEFAULT) {
        xmma::stg(ptr, make_uint2(f.reg(2*ii+0), f.reg(2*ii+1)), mem_desc);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_stg<16> {
    template< typename Fragment >
    static inline __device__ void stg(void *ptr,
                                      const Fragment &f,
                                      int ii,
                                      uint64_t mem_desc = MEM_DESC_DEFAULT) {
        xmma::stg(ptr,
                      make_uint4(f.reg(4*ii+0), f.reg(4*ii+1), f.reg(4*ii+2), f.reg(4*ii+3)),
                      mem_desc);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Fragment_stg<32> {
    template< typename Fragment >
    static inline __device__ void stg(void *ptr,
                                      const Fragment &f,
                                       int ii,
                                       uint64_t mem_desc = MEM_DESC_DEFAULT) {
        xmma::stg(ptr,
                      make_uint4(f.reg(8*ii+0), f.reg(8*ii+1), f.reg(8*ii+2), f.reg(8*ii+3)),
                      mem_desc);
        xmma::stg(static_cast<char *>(ptr)+sizeof(uint4),
                      make_uint4(f.reg(8*ii+4), f.reg(8*ii+5), f.reg(8*ii+6), f.reg(8*ii+7)),
                      mem_desc);
     }
 };

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// S E R I A L I Z A T I O N
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< int BYTES_PER_LDG, int M, int N >
inline __device__ void deserialize_( uint32_t (&dst)[N],
                                     const void *ptr,
                                     int tidx,
                                     int step,
                                     uint64_t mem_desc = MEM_DESC_DEFAULT ) {
    // The base pointer.
    const char *ptr_ = reinterpret_cast<const char*>(ptr);

    // Pack as many LDG.128 as we can.
    const int STEPS_4 = M / (BYTES_PER_LDG / 4);
    const int offset_uint4 = tidx * sizeof( uint4 );
    const int offset_uint2 = tidx * sizeof( uint2 );
    const int offset_uint1 = tidx * sizeof( uint32_t );
#pragma unroll
    for( int ii = 0; ii < STEPS_4; ++ii ) {
        if (BYTES_PER_LDG == 16) {
            uint4 dst_;
            xmma::ldg_cg( dst_, ptr_ + offset_uint4, mem_desc );
            dst[4 * ii + 0] = dst_.x;
            dst[4 * ii + 1] = dst_.y;
            dst[4 * ii + 2] = dst_.z;
            dst[4 * ii + 3] = dst_.w;
            ptr_ += step * sizeof( uint4 );
        } else if (BYTES_PER_LDG == 8) {
            uint2 dst_;
            xmma::ldg_cg( dst_, ptr_ + offset_uint2, mem_desc );
            dst[2 * ii + 0] = dst_.x;
            dst[2 * ii + 1] = dst_.y;
            ptr_ += step * sizeof( uint2 );
        } else if (BYTES_PER_LDG == 4) {
            xmma::ldg_cg( dst[ii], ptr_ + offset_uint1, mem_desc );
            ptr_ += step * sizeof( uint32_t );
        }
    }

    // Pack as many LDG.64 as we can.
    const int STEPS_2 = M % (BYTES_PER_LDG / 4) / 2;
    if( STEPS_2 == 1 ) {
        uint2 dst_;
        xmma::ldg_cg( dst_, ptr_ + offset_uint2, mem_desc );
        dst[4 * STEPS_4 + 0] = dst_.x;
        dst[4 * STEPS_4 + 1] = dst_.y;
        ptr_ += step * sizeof( uint2 );
    }

    // Finalize with LDG.32.
    const int STEPS_1 = M % (BYTES_PER_LDG / 4) % 2;
    if( STEPS_1 == 1 ) {
        xmma::ldg_cg(
            dst[( BYTES_PER_LDG / 4 ) * STEPS_4 + 2 * STEPS_2], ptr_ + offset_uint1, mem_desc );
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int BYTES_PER_STG, int M, int N >
inline __device__ void serialize_(void *ptr, const uint32_t (&src)[N], int tidx, int step) {
    // The base pointer.
    char *ptr_ = reinterpret_cast<char*>(ptr);

    // Pack as many STG.128 as we can.
    const int STEPS_4 = M / (BYTES_PER_STG / 4);
    #pragma unroll
    for( int ii = 0; ii < STEPS_4; ++ii ) {
        if (BYTES_PER_STG == 16) {
            asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};\n"
            :
            : "l"(ptr_ + tidx*sizeof(uint4))
            , "r"(src[4*ii + 0])
            , "r"(src[4*ii + 1])
            , "r"(src[4*ii + 2])
            , "r"(src[4*ii + 3]));
            ptr_ += step*sizeof(uint4);
        } else if (BYTES_PER_STG == 8) {
            asm volatile("st.global.v2.b32 [%0], {%1, %2};\n"
            :
            : "l"(ptr_ + tidx*sizeof(uint2))
            , "r"(src[2*ii + 0])
            , "r"(src[2*ii + 1]));
            ptr_ += step*sizeof(uint2);
        } else if (BYTES_PER_STG == 4) {
            asm volatile("st.global.b32 [%0], %1;\n"
            :
            : "l"(ptr_ + tidx*sizeof(uint32_t))
            , "r"(src[ii]));
            ptr_ += step*sizeof(uint32_t);
        }
    }

    // Pack as many STG.64 as we can.
    const int STEPS_2 = M % (BYTES_PER_STG / 4) / 2;
    if( STEPS_2 == 1 ) {
        asm volatile("st.global.v2.b32 [%0], {%1, %2};\n"
            :
            : "l"(ptr_ + tidx*sizeof(uint2))
            , "r"(src[4*STEPS_4 + 0])
            , "r"(src[4*STEPS_4 + 1]));
        ptr_ += step*sizeof(uint2);
    }

    // Finalize with STG.32.
    const int STEPS_1 = M % (BYTES_PER_STG / 4) % 2;
    if( STEPS_1 == 1 ) {
        asm volatile("st.global.b32 [%0], %1;\n"
            :
            : "l"(ptr_ + tidx*sizeof(uint32_t))
            , "r"(src[(BYTES_PER_STG / 4)*STEPS_4 + 2*STEPS_2]));
    }
}

template< int BYTES_PER_STG, int M, int N >
inline __device__ void serialize_atomic_add_(void *ptr, const uint32_t (&src)[N], int tidx, int step) {
    // The base pointer.
    char *ptr_ = reinterpret_cast<char*>(ptr);

    // Pack as many STG.128 as we can.
    const int STEPS_4 = M / (BYTES_PER_STG / 4);
    const int offset_uint4 = tidx*sizeof(uint4);
    const int offset_uint2 = tidx*sizeof(uint2);
    const int offset_uint1 = tidx*sizeof(uint32_t);
    #pragma unroll
    for( int ii = 0; ii < STEPS_4; ++ii ) {
        if (BYTES_PER_STG == 16) {
            red_add_f32(ptr_ + offset_uint4, src[4*ii + 0]);
            red_add_f32(ptr_ + offset_uint4 + 4, src[4*ii + 1]);
            red_add_f32(ptr_ + offset_uint4 + 8, src[4*ii + 2]);
            red_add_f32(ptr_ + offset_uint4 + 12, src[4*ii + 3]);
            ptr_ += step*sizeof(uint4);

        } else if (BYTES_PER_STG == 8) {
            red_add_f32(ptr_ + offset_uint2, src[2*ii + 0]);
            red_add_f32(ptr_ + offset_uint2 + 4, src[2*ii + 1]);
            ptr_ += step*sizeof(uint2);

        } else if (BYTES_PER_STG == 4) {
            red_add_f32(ptr_ + offset_uint1, src[ii]);
            ptr_ += step*sizeof(uint32_t);
        }
    }

    // Pack as many STG.64 as we can.
    const int STEPS_2 = M % (BYTES_PER_STG / 4) / 2;
    if( STEPS_2 == 1 ) {
        red_add_f32(ptr_ + offset_uint2, src[4*STEPS_4 + 0]);
        red_add_f32(ptr_ + offset_uint2 + 4, src[4*STEPS_4 + 1]);
        ptr_ += step*sizeof(uint2);

    }

    // Finalize with STG.32.
    const int STEPS_1 = M % (BYTES_PER_STG / 4) % 2;
    if( STEPS_1 == 1 ) {
        red_add_f32(ptr_ + offset_uint1, src[(BYTES_PER_STG / 4)*STEPS_4 + 2*STEPS_2]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Data_type_, int NUM_ELTS_, int BITS_PER_ELT_, int ALIGNMENT_ >
struct Fragment_base_ {

    // The data type.
    using Data_type = Data_type_;
    // default input type
    using Input_type_ = Data_type_;
    // Does it store the array of elements.
    enum { HAS_ELTS = BITS_PER_ELT_ >= 8 };
    // The number of elements.
    enum { NUM_ELTS = NUM_ELTS_ };
    // The size of element in bits.
    enum { BITS_PER_ELT = BITS_PER_ELT_ };
    // The size of byte of a single register.
    enum { BYTES_PER_REG = 4 };
    // The size in bits.
    enum { BITS_PER_REG = BYTES_PER_REG * 8 };
    // The number of registers needed to store the fragment.
    enum { NUM_REGS = Div_up<NUM_ELTS * BITS_PER_ELT, BITS_PER_REG>::VALUE };
    // The size in bytes (as returned by sizeof(Fragment_base<>).
    enum { SIZE_IN_BYTES = NUM_REGS * BYTES_PER_REG };
    // The alignment.
    enum { ALIGNMENT = ALIGNMENT_ > 0 ? ALIGNMENT_ : Min<NUM_REGS * BYTES_PER_REG, 16>::VALUE };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<
    // The type of the elements.
    typename Data_type_,
    // The number of elements.
    int NUM_ELTS_,
    // The size of each element in bits.
    int BITS_PER_ELT_,
    // The alignment if you want to force a value -- use 0 otherwise.
    int ALIGNMENT_,
    // The base class.
    typename Base_ = Fragment_base_<Data_type_, NUM_ELTS_, BITS_PER_ELT_, ALIGNMENT_>
>
struct alignas(static_cast<int>(Base_::ALIGNMENT)) Fragment_base : public Base_ {

    // The size of a load/store.
    enum { BYTES_PER_LOAD_STORE = Base_::NUM_REGS * sizeof(uint32_t) };

    // Clear the fragment.
    inline __device__ void clear() {
        #pragma unroll
        for( int ii = 0; ii < Base_::NUM_REGS; ++ii ) {
    //      this->reg(ii) = uint32_t(0);
            //inline PTX matters here as it will generates better PTXs than lwvm does for acc initilization.
            asm   volatile ( "mov.u32 %0, 0; \n"  : "=r"( this->reg(ii) ) :  );

        }
    }

    // Set the fragment with a scalar
    inline __device__ void set(uint32_t value) {
        #pragma unroll
        for( int ii = 0; ii < Base_::NUM_REGS; ++ii ) {
            this->reg(ii) = value;
        }
    }

    // Load from global memory (for inter-CTA split-k).
    template < int BYTES_PER_LDG = 16 >
    inline __device__ void deserialize( const void *ptr,
                                        int tidx,
                                        int threads,
                                        uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        xmma::deserialize_<BYTES_PER_LDG, Base_::NUM_REGS>(
            this->regs_, ptr, tidx, threads, mem_desc );
    }

    // Load from global memory.
    inline __device__ void ldg(const void *ptr, uint64_t mem_desc = MEM_DESC_DEFAULT) {
        Fragment_ldg<Base_::NUM_ELTS*Base_::BITS_PER_ELT/8>::ldg(*this, 0, ptr, mem_desc);
    }

    // Immutable access to a register.
    inline __device__ const uint32_t& reg(int ii) const {
        return this->regs_[ii];
    }

    // Mutable access to a register.
    inline __device__ uint32_t& reg(int ii) {
        return this->regs_[ii];
    }

    // Store to global memory (for inter-CTA split-k).
    template < int BYTES_PER_STG = 16>
    inline __device__ void serialize(void *ptr, int tidx, int threads) const {
        xmma::serialize_<BYTES_PER_STG, Base_::NUM_REGS>(ptr, this->regs_, tidx, threads);
    }

    template < int BYTES_PER_STG = 16>
    inline __device__ void serialize_atomic_add(void *ptr, int tidx, int threads) const {
        xmma::serialize_atomic_add_<BYTES_PER_STG, Base_::NUM_REGS>(ptr, this->regs_, tidx, threads);

    }

    // Store to global memory.
    inline __device__ void stg(void *ptr, uint64_t mem_desc = MEM_DESC_DEFAULT) const {
        Fragment_stg<Base_::NUM_ELTS*Base_::BITS_PER_ELT/8>::stg(ptr, *this, 0, mem_desc);
    }

    // Immutable access to a byte.
    inline __device__ uint8_t u8(int ii) const {
        return reinterpret_cast<const uint8_t*>(&this->regs_[0])[ii];
    }

    // Mutable access to a u8.
    inline __device__ uint8_t& u8(int ii) {
        return reinterpret_cast<uint8_t*>(&this->regs_[0])[ii];
    }

    // Immutable access to a half-word..
    inline __device__ uint16_t u16(int ii) const {
        return reinterpret_cast<const uint16_t*>(&this->regs_[0])[ii];
    }

    // Mutable access to a half-word.
    inline __device__ uint16_t& u16(int ii) {
        return reinterpret_cast<uint16_t*>(&this->regs_[0])[ii];
    }

    // Immutable access to a word.
    inline __device__ uint32_t u32(int ii) const {
        return reinterpret_cast<const uint32_t*>(&this->regs_[0])[ii];
    }

    // Mutable access to a word.
    inline __device__ uint32_t& u32(int ii) {
        return reinterpret_cast<uint32_t*>(&this->regs_[0])[ii];
    }

    // The storage in registers.
    //
    // NOTE: Instead of using only an array of uint32_t, we could use a union so we could either
    // access the registers or the elements. We found that for:
    //
    // union {
    //   uint16_t elts_[4]; uint32_t regs_[2];
    // };
    //
    // The compiler does not always produce a final structure of 8B. So, for the moment we are
    // going to go only with the regs_ array and use reinterpret_cast<> to access elements (see
    // below). It may be worth revisiting that when time permits.
    uint32_t regs_[Base_::NUM_REGS];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Data_type_, int NUM_ELTS_, int ALIGNMENT_ = 0 >
struct Fragment : public Fragment_base<Data_type_, NUM_ELTS_, 8*static_cast<int>(sizeof(Data_type_)), ALIGNMENT_> {

    // Immutable access to the elements.
    inline __device__ const Data_type_& elt(int ii) const {
        return reinterpret_cast<const Data_type_*>(&this->regs_[0])[ii];
    }

    // Mutable access to the elements.
    inline __device__ Data_type_& elt(int ii) {
        return reinterpret_cast<Data_type_*>(&this->regs_[0])[ii];
    }

    // Immutable access to the elements with a cast.
    template< typename Cast_type >
    inline __device__ const Cast_type& elt_as(int ii) const {
        return reinterpret_cast<const Cast_type*>(&this->regs_[0])[ii];
    }

    // Mutable access to the elements.
    template< typename Cast_type >
    inline __device__ Cast_type& elt_as(int ii) {
        return reinterpret_cast<Cast_type*>(&this->regs_[0])[ii];
    }

    // Add another fragment.
    inline __device__ void add(const Fragment &other) {
        #pragma unroll
        for( int ii = 0; ii < NUM_ELTS_; ++ii ) {
            this->elt(ii) += other.elt(ii);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int NUM_ELTS_, int ALIGNMENT_ >
struct Fragment<lwtlass::int4_t, NUM_ELTS_, ALIGNMENT_>
    : public Fragment_base<lwtlass::int4_t, NUM_ELTS_, 4, ALIGNMENT_> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int NUM_ELTS_, int ALIGNMENT_ >
struct Fragment<lwtlass::uint4_t, NUM_ELTS_, ALIGNMENT_>
    : public Fragment_base<lwtlass::uint4_t, NUM_ELTS_, 4, ALIGNMENT_> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int NUM_ELTS_, int ALIGNMENT_ >
struct Fragment<lwtlass::int2_t, NUM_ELTS_, ALIGNMENT_>
    : public Fragment_base<lwtlass::int2_t, NUM_ELTS_, 2, ALIGNMENT_> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int NUM_ELTS_, int ALIGNMENT_ >
struct Fragment<lwtlass::uint2_t, NUM_ELTS_, ALIGNMENT_>
    : public Fragment_base<lwtlass::uint2_t, NUM_ELTS_, 2, ALIGNMENT_> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int NUM_ELTS_, int ALIGNMENT_ >
struct Fragment<bool, NUM_ELTS_, ALIGNMENT_>
    : public Fragment_base<bool, NUM_ELTS_, 1, ALIGNMENT_> {
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H E L P E R   F U N C T O R S   T O   C R E A T E   F R A G M E N T S
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Data_type_, int SIZE_IN_BYTES_ >
struct Fragment_from_size_in_bytes {
    using Type = Fragment<Data_type_, SIZE_IN_BYTES_ / static_cast<int>(sizeof(Data_type_))>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int SIZE_IN_BYTES_ >
struct Fragment_from_size_in_bytes<bool, SIZE_IN_BYTES_> {
    using Type = Fragment<bool, SIZE_IN_BYTES_ * 8>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int SIZE_IN_BYTES_ >
struct Fragment_from_size_in_bytes<lwtlass::int2_t, SIZE_IN_BYTES_> {
    using Type = Fragment<lwtlass::int2_t, SIZE_IN_BYTES_ * 8 / 2>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int SIZE_IN_BYTES_ >
struct Fragment_from_size_in_bytes<lwtlass::uint2_t, SIZE_IN_BYTES_> {
    using Type = Fragment<lwtlass::uint2_t, SIZE_IN_BYTES_ * 8 / 2>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int SIZE_IN_BYTES_ >
struct Fragment_from_size_in_bytes<lwtlass::int4_t, SIZE_IN_BYTES_> {
    using Type = Fragment<lwtlass::int4_t, SIZE_IN_BYTES_ * 8 / 4>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int SIZE_IN_BYTES_ >
struct Fragment_from_size_in_bytes<lwtlass::uint4_t, SIZE_IN_BYTES_> {
    using Type = Fragment<lwtlass::uint4_t, SIZE_IN_BYTES_ * 8 / 4>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// F R A G M E N T S   W I T H   A   R O L E
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Layout >
struct Fragment_a {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Layout >
struct Fragment_b {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits >
struct Fragment_aclwmulator {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, bool = (Cta_tile::WARPS_K > 1) >
struct Fragment_epilogue_pre_swizzle {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, bool = (Cta_tile::WARPS_K > 1) >
struct Fragment_epilogue_post_swizzle {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, bool = (Cta_tile::WARPS_K > 1) >
struct Fragment_epilogue_interleaved_post_swizzle {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, bool = (Cta_tile::WARPS_K > 1) >
struct Fragment_c {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, bool = (Cta_tile::WARPS_K > 1) >
struct Fragment_interleaved_c {
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A . F 1 6 / F 3 2
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, int NUM_ELTS = 8 >
struct Fragment_hmma_base_c {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Fragment_hmma_base_c<Traits, Cta_tile, 2> : public Fragment<lwtlass::half_t, 2> {

    // The base class.
    using Base = Fragment<lwtlass::half_t, 2>;
    // Make sure the size of the fragment is what we expect.
    static_assert(sizeof(Base) == 4 && Base::SIZE_IN_BYTES == 4 && Base::NUM_REGS == 1, "");

    // Compute the sum between two fragments.
    inline __device__ void add(const Fragment_hmma_base_c &other) {
        this->regs[0] = hadd2(this->regs[0], other.regs[0]);
    }

    // Extract from an int.
    inline __device__ void from_int(const int &x) {
        this->regs[0] = x;
    }

    // Get an int from it.
    inline __device__ int to_int() const {
        return this->reg(0);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Fragment_hmma_base_c<Traits, Cta_tile, 4> : public Fragment<lwtlass::half_t, 4> {

    // The base class.
    using Base = Fragment<lwtlass::half_t, 4>;
    // Make sure the size of the fragment is what we expect.
    static_assert(sizeof(Base) == 8 && Base::SIZE_IN_BYTES == 8 && Base::NUM_REGS == 2, "");

    // Compute the sum between two fragments.
    inline __device__ void add(const Fragment_hmma_base_c &other) {
        this->regs[0] = hadd2(this->regs[0], other.regs[0]);
        this->regs[1] = hadd2(this->regs[1], other.regs[1]);
    }

    // Extract from an int2.
    inline __device__ void from_int2(const int2 &x) {
        this->regs[0] = x.x;
        this->regs[1] = x.y;
    }

    // Get an int2 from it.
    inline __device__ int2 to_int2() const {
        return make_int2(this->regs[0], this->regs[1]);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Fragment_hmma_base_c<Traits, Cta_tile, 8> : public Fragment<lwtlass::half_t, 8> {

    // The base class.
    using Base = Fragment<lwtlass::half_t, 8>;
    // Make sure the size of the fragment is what we expect.
    static_assert(sizeof(Base) == 16 && Base::SIZE_IN_BYTES == 16 && Base::NUM_REGS == 4, "");

    // Compute the sum between two fragments.
    inline __device__ void add(const Fragment_hmma_base_c &other) {
        this->reg(0) = hadd2(this->reg(0), other.reg(0));
        this->reg(1) = hadd2(this->reg(1), other.reg(1));
        this->reg(2) = hadd2(this->reg(2), other.reg(2));
        this->reg(3) = hadd2(this->reg(3), other.reg(3));
    }

    // Extract from an int2.
    inline __device__ void from_int2(const uint2 &x) {
        this->reg(0) = x.x;
        this->reg(1) = x.y;
    }

    // Extract from an int4.
    inline __device__ void from_int4(const uint4 &x) {
        this->reg(0) = x.x;
        this->reg(1) = x.y;
        this->reg(2) = x.z;
        this->reg(3) = x.w;
    }

    // Get an int2 from it.
    inline __device__ uint2 to_int2() const {
        return make_uint2(this->reg(0), this->reg(1));
    }

    // Get an int4 from it.
    inline __device__ uint4 to_int4() const {
        return make_uint4(this->reg(0), this->reg(1), this->reg(2), this->reg(3));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A . F 1 6
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Fragment_hmma_fp16_epilogue_pre_swizzle : public Fragment_hmma_base_c<Traits, Cta_tile> {

    // The aclwmulators from the main loop.
    using Aclwmulators = Fragment_aclwmulator<Traits>;

    // Colwert from fp16 aclwmulators to fp16 outputs.
    inline __device__ void colwert(lwtlass::half_t alpha, const Aclwmulators &c) {
        ushort2 alpha_ = make_ushort2(alpha, alpha);

        this->reg(0) = hmul2(reinterpret_cast<const uint32_t&>(alpha_), c.reg(0));
        this->reg(1) = hmul2(reinterpret_cast<const uint32_t&>(alpha_), c.reg(1));
        this->reg(2) = hmul2(reinterpret_cast<const uint32_t&>(alpha_), c.reg(2));
        this->reg(3) = hmul2(reinterpret_cast<const uint32_t&>(alpha_), c.reg(3));
    }

    // Colwert from fp16 aclwmulators to fp16 outputs.
    inline __device__ void scaled_colwert(lwtlass::half_t alpha, const Aclwmulators &c) {
        ushort2 alpha_ = make_ushort2(alpha, alpha);

        this->reg(0) = hmul2(reinterpret_cast<const uint32_t&>(alpha_), c.reg(0));
        this->reg(1) = hmul2(reinterpret_cast<const uint32_t&>(alpha_), c.reg(1));
        this->reg(2) = hmul2(reinterpret_cast<const uint32_t&>(alpha_), c.reg(2));
        this->reg(3) = hmul2(reinterpret_cast<const uint32_t&>(alpha_), c.reg(3));
    }

    inline __device__ void shuffle_groups(Aclwmulators &acc) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Fragment_hmma_fp16_epilogue_post_swizzle
    : public Fragment<lwtlass::half_t, Cta_tile::WARPS_K*8> {

    // The base class.
    using Base = Fragment<lwtlass::half_t, Cta_tile::WARPS_K*8>;

    // The number of registers after reduction.
    enum { NUM_REGS_AFTER_REDUCTION = 4 };
    // Make sure the fragment oclwpies 4 registers after reduction.
    static_assert(Base::NUM_REGS == NUM_REGS_AFTER_REDUCTION*Cta_tile::WARPS_K, "");
    // The number of bytes for load/store -- we only load/store the 1st 16 bytes.
    enum { BYTES_PER_LOAD_STORE = NUM_REGS_AFTER_REDUCTION*sizeof(uint32_t) };

    // Add two fragments together.
    template< typename Other_fragment >
    inline __device__ void add(const Other_fragment &other) {
        #pragma unroll
        for( int ii = 0; ii < 4; ++ii ) {
            this->reg(ii) = hadd2(this->reg(ii), other.reg(ii));
        }
    }

    // The residual is added later.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res, lwtlass::half_t beta) {
    }

    // The bias is added later.
    template< typename Fragment_bias >
    inline __device__ void add_bias(const Fragment_bias &bias) {
    }

    // The bias is added later.
    template< typename Fragment_bias >
    inline __device__ void add_bias_nchw(const Fragment_bias &bias, int i) {
    }

    // ReLu.
    inline __device__ void relu(lwtlass::half_t relu_lb) {
    }

    // Clip-ReLu.
    inline __device__ void relu_ub(lwtlass::half_t relu_ub) {
    }

    // Gelu_erf activation.
    inline __device__ void gelu_erf(float gelu_scale) {
    }

    // Load from global memory (for inter-CTA split-k).
    // Load from global memory (for inter-CTA split-k).
    template < int BYTES_PER_LDG = 16 >
    inline __device__ void deserialize( const void *ptr,
                                        int tidx,
                                        int threads,
                                        uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        xmma::deserialize_<BYTES_PER_LDG, NUM_REGS_AFTER_REDUCTION>(
            this->regs_, ptr, tidx, threads, mem_desc );
    }

    // Do the reduction for in-CTA split-K.
    inline __device__ void reduce(lwtlass::half_t) {
        #pragma unroll
        for( int ki = 1; ki < Cta_tile::WARPS_K; ++ki ) {
            this->reg(0) = hadd2(this->reg(0), this->reg(4*ki+0));
            this->reg(1) = hadd2(this->reg(1), this->reg(4*ki+1));
            this->reg(2) = hadd2(this->reg(2), this->reg(4*ki+2));
            this->reg(3) = hadd2(this->reg(3), this->reg(4*ki+3));
        }
    }

    // Store to global memory (for inter-CTA split-k).
    template < int BYTES_PER_STG = 16>
    inline __device__ void serialize(void *ptr, int tidx, int threads) const {
        xmma::serialize_<BYTES_PER_STG, NUM_REGS_AFTER_REDUCTION>(ptr, this->regs_, tidx, threads);
    }

    //Atomic add to global memory (for inter-CTA split-k).
    template < int BYTES_PER_STG = 16>
    inline __device__ void serialize_atomic_add(void *ptr, int tidx, int threads) const {
        xmma::serialize_atomic_add_<BYTES_PER_STG, NUM_REGS_AFTER_REDUCTION>(ptr, this->regs_, tidx, threads);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Fragment_hmma_fp16_c : public Fragment_hmma_base_c<Traits, Cta_tile> {
    // The residual.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res_, lwtlass::half_t beta) {
        uint4 res = res_.to_int4();
        ushort2 beta_ = make_ushort2(beta, beta);

        this->reg(0) = hfma2(reinterpret_cast<const uint32_t&>(beta_), res.x, this->reg(0));
        this->reg(1) = hfma2(reinterpret_cast<const uint32_t&>(beta_), res.y, this->reg(1));
        this->reg(2) = hfma2(reinterpret_cast<const uint32_t&>(beta_), res.z, this->reg(2));
        this->reg(3) = hfma2(reinterpret_cast<const uint32_t&>(beta_), res.w, this->reg(3));
    }

    // Compute the sum between two fragments.
    template<typename Fragment_post_swizzle>
    inline __device__ void pack(lwtlass::half_t, const Fragment_post_swizzle &frag) {

        this->reg(0) = frag.reg(0);
        this->reg(1) = frag.reg(1);
        this->reg(2) = frag.reg(2);
        this->reg(3) = frag.reg(3);
    }

    // The bias.
    template< typename Fragment_bias >
    inline __device__ void add_bias(const Fragment_bias &bias_) {

        this->reg(0) = hadd2(bias_.reg(0), this->reg(0));
        this->reg(1) = hadd2(bias_.reg(1), this->reg(1));
        this->reg(2) = hadd2(bias_.reg(2), this->reg(2));
        this->reg(3) = hadd2(bias_.reg(3), this->reg(3));
    }

    template< typename Fragment_bias >
    inline __device__ void add_bias_relu(const Fragment_bias &bias_,
                                         int32_t with_relu,
                                         lwtlass::half_t relu_lb,
                                         float one) {
        uint32_t one2 = float2_to_half2(one, one);

        ushort2 tmp = make_ushort2(relu_lb, relu_lb);
        uint32_t relu_lb_ = reinterpret_cast<uint32_t&>(tmp);

#pragma unroll
        for( int ii = 0; ii < 4; ++ii ) {
            this->reg(ii) =
                xmma::hfma2_relu(this->reg(ii), one2, bias_.reg(ii), with_relu, relu_lb_);
        }
    }

    // The bias is added later.
    template< typename Fragment_bias >
    inline __device__ void add_bias_nchw(const Fragment_bias &bias, int i) {
    }

    // ReLu.
    inline __device__ void relu(lwtlass::half_t relu_lb) {
        ushort2 relu_lb_ = make_ushort2(relu_lb, relu_lb);
        this->reg(0) = relu_fp16x2(this->reg(0), reinterpret_cast<const uint32_t&>(relu_lb_));
        this->reg(1) = relu_fp16x2(this->reg(1), reinterpret_cast<const uint32_t&>(relu_lb_));
        this->reg(2) = relu_fp16x2(this->reg(2), reinterpret_cast<const uint32_t&>(relu_lb_));
        this->reg(3) = relu_fp16x2(this->reg(3), reinterpret_cast<const uint32_t&>(relu_lb_));
    }

    // Clip-ReLu.
    inline __device__ void relu_ub(lwtlass::half_t relu_ub) {
        ushort2 relu_ub_ = make_ushort2(relu_ub, relu_ub);
        this->reg(0) = relu_ub_fp16x2(this->reg(0), reinterpret_cast<const uint32_t&>(relu_ub_));
        this->reg(1) = relu_ub_fp16x2(this->reg(1), reinterpret_cast<const uint32_t&>(relu_ub_));
        this->reg(2) = relu_ub_fp16x2(this->reg(2), reinterpret_cast<const uint32_t&>(relu_ub_));
        this->reg(3) = relu_ub_fp16x2(this->reg(3), reinterpret_cast<const uint32_t&>(relu_ub_));
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// H M M A . F 3 2
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Fragment_hmma_fp32_epilogue_pre_swizzle : public Fragment<float, 8> {

    // The aclwmulators.
    using Aclwmulators = Fragment_aclwmulator<Traits>;

    // Compute the sum between two fragments.
    inline __device__ void add(const Fragment_hmma_fp32_epilogue_pre_swizzle &other) {
        // Not needed as it happens after the swizzle!
    }

    // Quantize the aclwmulators -- actually simply do a scaled copy.
    inline __device__ void colwert(float, const Aclwmulators &acc) {
        this->elt(0) = acc.elt(0);
        this->elt(1) = acc.elt(1);
        this->elt(2) = acc.elt(4);
        this->elt(3) = acc.elt(5);
        this->elt(4) = acc.elt(2);
        this->elt(5) = acc.elt(3);
        this->elt(6) = acc.elt(6);
        this->elt(7) = acc.elt(7);
    }

    // Quantize the aclwmulators -- actually simply do a scaled copy.
    inline __device__ void scaled_colwert(float alpha, const Aclwmulators &acc) {
        this->elt(0) = alpha * acc.elt(0);
        this->elt(1) = alpha * acc.elt(1);
        this->elt(2) = alpha * acc.elt(4);
        this->elt(3) = alpha * acc.elt(5);
        this->elt(4) = alpha * acc.elt(2);
        this->elt(5) = alpha * acc.elt(3);
        this->elt(6) = alpha * acc.elt(6);
        this->elt(7) = alpha * acc.elt(7);
    }


    inline __device__ void colwert(Fragment<float, 8>, const Aclwmulators &acc) {
        this->elt(0) = acc.elt(0);
        this->elt(1) = acc.elt(1);
        this->elt(2) = acc.elt(4);
        this->elt(3) = acc.elt(5);
        this->elt(4) = acc.elt(2);
        this->elt(5) = acc.elt(3);
        this->elt(6) = acc.elt(6);
        this->elt(7) = acc.elt(7);
    }


    inline __device__ void shuffle_groups(Aclwmulators &acc) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Fragment_hmma_fp32_epilogue_post_swizzle : public Fragment<float, Cta_tile::WARPS_K*8> {

    using Base = Fragment<float, Cta_tile::WARPS_K*8>;

    // The number of registers after reduction.
    enum { NUM_REGS_AFTER_REDUCTION = 8 };
    // Make sure the fragment oclwpies 8 registers after reduction.
    static_assert(Base::NUM_REGS == NUM_REGS_AFTER_REDUCTION*Cta_tile::WARPS_K, "");
    // The number of bytes for load/store -- we only load/store the 1st 16 bytes.
    enum { BYTES_PER_LOAD_STORE = NUM_REGS_AFTER_REDUCTION*sizeof(uint32_t) };

    // Add two fragments together.
    template< typename Other_fragment >
    inline __device__ void add(const Other_fragment &other) {
        #pragma unroll
        for( int ii = 0; ii < 8; ++ii ) {
            this->elt(ii) = this->elt(ii) + other.elt(ii);
        }
    }

    // The residual is added.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res, float beta) {
        for ( int ii = 0; ii < Fragment_c::NUM_REGS; ii++ ) {
            float2 tmp = half2_to_float2(res.reg(ii));
            this->elt(ii * 2 + 0) += beta * tmp.x;
            this->elt(ii * 2 + 1) += beta * tmp.y;
        }
    }

    // The bias is added.
    template< typename Fragment_bias >
    inline __device__ void add_bias(const Fragment_bias &bias) {
        #pragma unroll
        for (int ii = 0; ii < Fragment_bias::NUM_REGS; ii++) {
            float2 tmp = half2_to_float2(bias.reg(ii));
            this->elt(ii * 2 + 0) += tmp.x;
            this->elt(ii * 2 + 1) += tmp.y;
        }
    }

    // The bias is added later.
    template< typename Fragment_bias >
    inline __device__ void add_bias_nchw(const Fragment_bias &bias, int i) {
    }

    // ReLu.
    inline __device__ void relu(float relu_lb=0.f) {
        #pragma unroll
        for ( int ii = 0; ii < Base::NUM_ELTS; ++ii ) {
            this->elt(ii) = xmma::relu_fp32(this->elt(ii), relu_lb);
        }
    }

    // Clip-ReLu.
    inline __device__ void relu_ub(float relu_ub) {
        #pragma unroll
        for ( int ii = 0; ii < Base::NUM_ELTS; ++ii ) {
            this->elt(ii) = xmma::relu_ub_fp32(this->elt(ii), relu_ub);
        }
    }
    
    // Gelu_erf activation.
    inline __device__ void gelu_erf(float gelu_scale) {
    }

    // Load from global memory (for inter-CTA split-k).
    template < int BYTES_PER_LDG = 16 >
    inline __device__ void deserialize( const void *ptr,
                                        int tidx,
                                        int threads,
                                        uint64_t mem_desc = MEM_DESC_DEFAULT ) {
        xmma::deserialize_<BYTES_PER_LDG, NUM_REGS_AFTER_REDUCTION>(
            this->regs_, ptr, tidx, threads, mem_desc );
    }

    // Do the parallel reduction.
    inline __device__ void reduce(float alpha) {
        #pragma unroll
        for( int ni = 0; ni < 8; ++ni ) {
            #pragma unroll
            for( int ki = 1; ki < Cta_tile::WARPS_K; ++ki ) {
              this->elt(ni) += this->elt(ki*8 + ni);
            }
            this->elt(ni) *= alpha;
        }
    }

    // Store to global memory (for inter-CTA split-k).
    template < int BYTES_PER_STG = 16>
    inline __device__ void serialize(void *ptr, int tidx, int threads) const {
        xmma::serialize_<BYTES_PER_STG, NUM_REGS_AFTER_REDUCTION>(ptr, this->regs_, tidx, threads);
    }

    // Atomic add to global memory (for inter-CTA split-k).
    template < int BYTES_PER_STG = 16>
    inline __device__ void serialize_atomic_add(void *ptr, int tidx, int threads) const {
        xmma::serialize_atomic_add_<BYTES_PER_STG, NUM_REGS_AFTER_REDUCTION>(ptr, this->regs_, tidx, threads);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, bool = (Cta_tile::WARPS_K > 1) >
struct Fragment_hmma_fp32_epilogue_interleaved_post_swizzle {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Fragment_hmma_fp32_epilogue_interleaved_post_swizzle<Traits, Cta_tile, false>
    : public Fragment<lwtlass::half_t, 2> {

    // Add two fragments for inter-CTA split-k.
    template< typename Other_fragment >
    inline __device__ void add(const Other_fragment &other) {
        this->reg(0) = hadd2(this->reg(0), other.reg(0));
    }

    // The residual is added later.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &, float) {
    }

    // The bias is added later.
    template< typename Fragment_bias >
    inline __device__ void add_bias(const Fragment_bias &bias) {
    }

    // The bias is added later.
    template< typename Fragment_bias >
    inline __device__ void add_bias_nchw(const Fragment_bias &bias, int i) {
    }

    // Do the parallel reduction.
    inline __device__ void reduce(float) {
    }

    // RELU activation, not implemented.
    inline __device__ void relu(float relu_lb=0.f) {
        assert(false);
    }

    // Clip-ReLu, not implemented.
    inline __device__ void relu_ub(float) {
        assert(false);
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////
template< typename Traits, typename Cta_tile >
struct Fragment_hmma_fp32_c : public Fragment_hmma_base_c<Traits, Cta_tile> {

    // The residual.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res_, float beta) {
    }

    // Compute the sum between two fragments.
    template<typename Fragment_post_swizzle>
    inline __device__ void pack(float, const Fragment_post_swizzle &frag) {

        this->reg(0) = float2_to_half2(frag.elt(0), frag.elt(1));
        this->reg(1) = float2_to_half2(frag.elt(2), frag.elt(3));
        this->reg(2) = float2_to_half2(frag.elt(4), frag.elt(5));
        this->reg(3) = float2_to_half2(frag.elt(6), frag.elt(7));
    }

    // The bias.
    template< typename Fragment_bias >
    inline __device__ void add_bias(const Fragment_bias &bias_) {
    }

    template< typename Fragment_bias >
    inline __device__ void add_bias_relu(const Fragment_bias &bias_, uint32_t with_relu,
                                         float relu_lb, float one) {
    }

    // The bias is added later.
    template< typename Fragment_bias >
    inline __device__ void add_bias_nchw(const Fragment_bias &bias, int i) {
    }

    // ReLu.
    inline __device__ void relu(float relu_lb) {
    }

    // Clip-ReLu.
    inline __device__ void relu_ub(float relu_ub) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, bool = (Cta_tile::WARPS_K > 1) >
struct Fragment_hmma_fp32_interleaved_c {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Fragment_hmma_fp32_interleaved_c<Traits, Cta_tile, false>
    : public Fragment_hmma_base_c<Traits, Cta_tile, 1> {

    // The residual.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res, float beta) {
        uint32_t beta_beta = float2_to_half2(beta, beta);
        this->reg(0) = hfma2(beta_beta, res.to_int(), this->reg(0));
    }

    // Compute the sum between two fragments.
    template<typename Fragment_post_swizzle>
    inline __device__ void pack(float, const Fragment_post_swizzle &frag) {
        this->reg(0) = frag.reg(0);
    }

    // The bias.
    template< typename Fragment_bias >
    inline __device__ void add_bias(const Fragment_bias &bias_) {
        this->reg(0) = hadd2(bias_.reg(0), this->reg(0));
    }

    // The bias+relu.
    template< typename Fragment_bias >
    inline __device__ void add_bias_relu(const Fragment_bias &bias_,
                                         int32_t with_relu,
                                         float relu_lb,
                                         float one) {
    }

    // The bias is added later.
    template< typename Fragment_bias >
    inline __device__ void add_bias_nchw(const Fragment_bias &bias, int i) {
    }

    // ReLu.
    inline __device__ void relu(float relu_lb) {
        half2 relu_lb_ = xmma::colwert<half2>(relu_lb);
        this->reg(0) = relu_fp16x2(this->reg(0), reinterpret_cast<uint32_t&>(relu_lb_));
    }

    // Clip-ReLu.
    inline __device__ void relu_ub(float relu_ub) {
        half2 relu_ub_ = xmma::colwert<half2>(relu_ub);
        this->reg(0) = relu_ub_fp16x2(this->reg(0), reinterpret_cast<uint32_t&>(relu_ub_));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// I M M A . x -> I N T 3 2
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Fragment_imma_int32_epilogue_pre_swizzle : public Fragment<float, 8> {

    // The aclwmulators.
    using Aclwmulators = Fragment_aclwmulator<Traits>;

    // Compute the sum between two fragments.
    inline __device__ void add(const Fragment_imma_int32_epilogue_pre_swizzle &other) {
    }

    // Quantize the aclwmulators -- actually simply do a scaled copy.
    inline __device__ void colwert( float &alpha, const Aclwmulators &acc ) {
        // This is for per-tensor scaling.
        #pragma unroll
        for( int ii = 0; ii < Aclwmulators::NUM_REGS; ++ii ) {
            this->elt( ii ) = xmma::i2f( acc.elt( ii ) );
        }
    }

    // Quantize the aclwmulators -- per_channel scaling.
    inline __device__ void colwert( Fragment<float, 8> &alpha, const Aclwmulators &acc ) {
        #pragma unroll
        for( int ii = 0; ii < Aclwmulators::NUM_REGS; ++ii ) {
            this->elt( ii ) = xmma::i2f( acc.elt( ii ) );
        }
    }

    inline __device__ void shuffle_groups(Aclwmulators &acc) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Fragment_imma_int32_epilogue_post_swizzle : public Fragment<float, 16> {

    // The base class.
    using Base = Fragment<float, 16>;

    // Do the parallel reduction.
    inline __device__ void reduce(float &alpha) {
        #pragma unroll
        for( int ni = 0; ni < 16; ++ni ) {
            this->elt(ni) *= alpha;
        }
    }
    inline __device__ void reduce(Fragment<float, 16> &alpha) {
        #pragma unroll
        for( int ni = 0; ni < 16; ++ni ) {
            this->elt(ni) *= alpha.elt(ni);
        }
    }

    // The residual.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res, float beta) {
        for( int ii = 0; ii < Fragment_c::NUM_REGS; ++ii ) {
            char4 tmp;
            tmp = reinterpret_cast<const char4&>(res.reg(ii));
            this->elt(ii * 4 + 0) += beta * float(tmp.x);
            this->elt(ii * 4 + 1) += beta * float(tmp.y);
            this->elt(ii * 4 + 2) += beta * float(tmp.z);
            this->elt(ii * 4 + 3) += beta * float(tmp.w);
        }

    }

    // Per-channel beta scaling.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res, Fragment<float, 16> &beta) {
        #pragma unroll
        for( int ii = 0; ii < Fragment_c::NUM_REGS; ++ii ) {
            float4 tmp = s8x4_to_float4(res.reg(ii));
            this->elt(ii * 4 + 0) += tmp.x * beta.elt((ii * 4 + 0));
            this->elt(ii * 4 + 1) += tmp.y * beta.elt((ii * 4 + 1));
            this->elt(ii * 4 + 2) += tmp.z * beta.elt((ii * 4 + 2));
            this->elt(ii * 4 + 3) += tmp.w * beta.elt((ii * 4 + 3));
        }
    }

    // The bias is added later.
    template< typename Fragment_bias >
    inline __device__ void add_bias(const Fragment_bias &bias) {
        #pragma unroll
        for ( int ii = 0; ii < Base::NUM_ELTS; ++ii ) {
            this->elt(ii) += bias.elt(ii);
        }
    }

    // The bias is added later.
    template< typename Fragment_bias >
    inline __device__ void add_bias_nchw(const Fragment_bias &bias, int i) {
    }

    // RELU activation.
    inline __device__ void relu(float relu_lb=0.f) {
        #pragma unroll
        for ( int ii = 0; ii < Base::NUM_ELTS; ++ii ) {
            this->elt(ii) = xmma::relu_fp32(this->elt(ii), relu_lb);
        }
    }

    // Clip-RELU activation.
    inline __device__ void relu_ub(float relu_ub) {
        #pragma unroll
        for ( int ii = 0; ii < Base::NUM_ELTS; ++ii ) {
            this->elt(ii) = xmma::relu_ub_fp32(this->elt(ii), relu_ub);
        }
    }

    // Gelu_erf activation.
    inline __device__ void gelu_erf(float gelu_scale) {
        #pragma unroll
        for ( int ii = 0; ii < Base::NUM_ELTS; ++ii ) {
            this->elt(ii) = xmma::gelu_erf(this->elt(ii), gelu_scale);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Fragment_imma_int32_c : public Fragment<int32_t, 4> {

    // Compute the sum between two fragments.
    inline __device__ void add(const Fragment_imma_int32_c &other) {
        this->elt(0) += other.elt(0);
        this->elt(1) += other.elt(1);
        this->elt(2) += other.elt(2);
        this->elt(3) += other.elt(3);
    }

    // Add the residual.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res_, float beta) {
    }

    // Extract from an int2.
    inline __device__ void from_int2(const uint2 &x) {
        this->elt(0) = x.x;
        this->elt(1) = x.y;
    }

    // Extract from an int4.
    inline __device__ void from_int4(const uint4 &x) {
        this->elt(0) = x.x;
        this->elt(1) = x.y;
        this->elt(2) = x.z;
        this->elt(3) = x.w;
    }

    // Compute the sum between two fragments.
    template<typename Fragment_post_swizzle>
    inline __device__ void pack(int32_t,
                                const Fragment_post_swizzle &frag) {
        int32_t tmp[4];
        #pragma unroll
        for( int ii = 0; ii < Fragment_post_swizzle::NUM_REGS/4; ++ii ) {
            tmp[0] = f2i(frag.elt(4 * ii    ));
            tmp[1] = f2i(frag.elt(4 * ii + 1));
            tmp[2] = f2i(frag.elt(4 * ii + 2));
            tmp[3] = f2i(frag.elt(4 * ii + 3));

            this->reg(ii) = pack_int8x4(tmp);
        }
    }

    // The bias.
    template< typename Fragment_bias >
    inline __device__ void add_bias(const Fragment_bias &bias_) {
    }

    // The bias+relu.
    template< typename Fragment_bias >
    inline __device__ void add_bias_relu(const Fragment_bias &bias_,
                                         int32_t with_relu,
                                         float relu_lb,
                                         float one) {
    }

    // The bias is added later.
    template< typename Fragment_bias >
    inline __device__ void add_bias_nchw(const Fragment_bias &bias, int i) {
    }

    // RELU activation.
    inline __device__ void relu(float) {
    }

    // Clip-RELU activation.
    inline __device__ void relu_ub(float) {
    }

    // Get an int2 from it.
    inline __device__ uint2 to_int2() const {
        return make_uint2(this->reg(0), this->reg(1));
    }

    // Colwert to an int4.
    inline __device__ uint4 to_int4() const {
        return make_uint4(this->reg(0), this->reg(1), this->reg(2), this->reg(3));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Fragment_imma_interleaved_int32_epilogue_pre_swizzle
    : public Fragment<float, 16> {

    // The aclwmulators.
    using Aclwmulators = Fragment_aclwmulator<Traits>;

    // Compute the sum between two fragments.
    inline __device__ void add(
        const Fragment_imma_interleaved_int32_epilogue_pre_swizzle &other) { }

    // Quantize the aclwmulators -- actually simply do a scaled copy.
    inline __device__ void colwert(Fragment<float, 16> &alpha, const Aclwmulators &acc) {
       #pragma unroll
       for( int ii = 0; ii < Aclwmulators::NUM_REGS; ++ii ) {
           asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(this->elt(ii)) : "r"(acc.elt(ii)));
           this->elt(ii) = alpha.elt(ii) * this->elt(ii);
       }
    }

    inline __device__ void shuffle_groups(Aclwmulators &acc) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Fragment_ampere_imma_interleaved_int32_epilogue_pre_swizzle
    : public Fragment<float, 16> {

    // The aclwmulators.
    using Aclwmulators = Fragment_aclwmulator<Traits>;

    // Compute the sum between two fragments.
    inline __device__ void add(
        const Fragment_ampere_imma_interleaved_int32_epilogue_pre_swizzle &other) { }

    // Quantize the aclwmulators -- actually simply do a scaled copy.
    inline __device__ void colwert(Fragment<float, 16> &alpha, const Aclwmulators &acc) {

        // This is for per-tensor scaling.
        #pragma unroll
        for( int ii = 0; ii < Aclwmulators::NUM_REGS; ++ii ) {
            int map_ii = (ii % 8) / 2 * 4 + ii % 2 + (ii / 8) % 2 * 2;
            asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(this->elt(ii)) : "r"(acc.elt(map_ii)));
            this->elt(ii) = alpha.elt(ii) * this->elt(ii);
        }
    }

    inline __device__ void shuffle_groups(Aclwmulators &acc) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile, bool = (Cta_tile::WARPS_K > 1) >
struct Fragment_imma_fp32_epilogue_interleaved_post_swizzle : public Fragment<float, 8> {

    // The base class.
    using Base = Fragment<float, 8>;

    // Add two fragments for inter-CTA split-k.
    template< typename Other_fragment >
    inline __device__ void add(const Other_fragment &other) {
        #pragma unroll
        for( int ii = 0; ii < 8; ++ii ) {
            this->elt(ii) = this->elt(ii) + other.elt(ii);
        }
    }

    // The residual.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res, Fragment<float, 8> &beta) {
        #pragma unroll
        for ( int ri = 0; ri < Base::NUM_REGS / 4; ++ri ) {
            float4 tmp = s8x4_to_float4(res.reg(ri));
            this->elt(ri * 4 + 0) += tmp.x * beta.elt((ri * 4 + 0));
            this->elt(ri * 4 + 1) += tmp.y * beta.elt((ri * 4 + 1));
            this->elt(ri * 4 + 2) += tmp.z * beta.elt((ri * 4 + 2));
            this->elt(ri * 4 + 3) += tmp.w * beta.elt((ri * 4 + 3));
        }
    }

    // The bias is added later.
    template< typename Fragment_bias >
    inline __device__ void add_bias(const Fragment_bias &bias) {
        #pragma unroll
        for ( int ii = 0; ii < Base::NUM_ELTS; ++ii ) {
            this->elt(ii) += bias.elt(ii);
        }
    }

    // The bias is added later.
    template< typename Fragment_bias >
    inline __device__ void add_bias_nchw(const Fragment_bias &bias, int i) {
    }

    // Do the parallel reduction.
    inline __device__ void reduce(Fragment<float, 8> &alpha) {
    }

    // RELU activation.
    inline __device__ void relu(float relu_lb=0.f) {
        #pragma unroll
        for ( int ii = 0; ii < Base::NUM_ELTS; ++ii ) {
            this->elt(ii) = fmax(this->elt(ii), relu_lb);
        }
    }

    // Clip-RELU activation.
    inline __device__ void relu_ub(float relu_ub) {
        #pragma unroll
        for ( int ii = 0; ii < Base::NUM_ELTS; ++ii ) {
            this->elt(ii) = xmma::relu_ub_fp32(this->elt(ii), relu_ub);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// used in implicit_gemm_imma_nhwc and gemm_imma
template< typename Traits, typename Cta_tile >
struct Fragment_imma_nhwc_int8_c : public Fragment<int8_t, 16> {

    // Compute the sum between two fragments.
    inline __device__ void add(const Fragment_imma_nhwc_int8_c &other) {
        #pragma unroll
        for( int ii = 0; ii < 16; ii++ ) {
            this->elt(ii) += other.elt(ii);
        }
    }

    // Add the residual.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res_, float beta) {
    }
    // Add the residual.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res_, Fragment<float, 16>) {
    }

    // Extract from an int2.
    inline __device__ void from_int2(const uint2 &x) {
        this->reg(0) = x.x;
        this->reg(1) = x.y;
    }

    // Extract from an int4.
    inline __device__ void from_int4(const uint4 &x) {
        this->reg(0) = x.x;
        this->reg(1) = x.y;
        this->reg(2) = x.z;
        this->reg(3) = x.w;
    }

    // Compute the sum between two fragments.
    template<typename Fragment_post_swizzle>
    inline __device__ void pack(int32_t,
                                const Fragment_post_swizzle &frag) {
        int32_t tmp[4];
        #pragma unroll
        for( int ii = 0; ii < Fragment_post_swizzle::NUM_REGS/4; ++ii ) {
            tmp[0] = f2i(frag.elt(4 * ii    ));
            tmp[1] = f2i(frag.elt(4 * ii + 1));
            tmp[2] = f2i(frag.elt(4 * ii + 2));
            tmp[3] = f2i(frag.elt(4 * ii + 3));

            this->reg(ii) = pack_int8x4(tmp);
        }
    }

    template<typename Fragment_post_swizzle>
    inline __device__ void pack(Fragment<float, Fragment_post_swizzle::NUM_REGS>,
                                const Fragment_post_swizzle &frag) {
        int32_t tmp[4];
        #pragma unroll
        for( int ii = 0; ii < Fragment_post_swizzle::NUM_REGS/4; ++ii ) {
            tmp[0] = f2i(frag.elt(4 * ii + 0));
            tmp[1] = f2i(frag.elt(4 * ii + 1));
            tmp[2] = f2i(frag.elt(4 * ii + 2));
            tmp[3] = f2i(frag.elt(4 * ii + 3));

            this->reg(ii) = pack_int8x4(tmp);
        }
    }

    // The bias.
    template< typename Fragment_bias >
    inline __device__ void add_bias(const Fragment_bias &bias_) {
    }

    // The bias+relu.
    template< typename Fragment_bias >
    inline __device__ void add_bias_relu(const Fragment_bias &bias_,
                                         int32_t with_relu,
                                         float relu_lb,
                                         float one) {
    }

    // The bias is added later.
    template< typename Fragment_bias >
    inline __device__ void add_bias_nchw(const Fragment_bias &bias, int i) {
    }

    // RELU activation.
    inline __device__ void relu(float) {
    }

    // Clip-RELU activation.
    inline __device__ void relu_ub(float) {
    }

    // Get an int2 from it.
    inline __device__ uint2 to_int2() const {
        return make_uint2(this->reg(0), this->reg(1));
    }

    // Colwert to an int4.
    inline __device__ uint4 to_int4() const {
        return make_uint4(this->reg(0), this->reg(1), this->reg(2), this->reg(3));
    }
};


// used in implicit_gemm_interleaved_imma
template< typename Traits, typename Cta_tile >
struct Fragment_imma_int8_c : public Fragment<int8_t, 8> {

    // Compute the sum between two fragments.
    inline __device__ void add(const Fragment_imma_int8_c &other) {
        this->reg(0) += other.reg(0);
        this->reg(1) += other.reg(1);
    }

    // Add the residual.
    template< typename Fragment_c >
    inline __device__ void add_residual(const Fragment_c &res_, Fragment<float, 8> beta) {
    }

    // Extract from an int2.
    inline __device__ void from_int2(const uint2 &x) {
        this->reg(0) = x.x;
        this->reg(1) = x.y;
    }

    // Compute the sum between two fragments.
    template<typename Fragment_post_swizzle>
    inline __device__ void pack(int32_t, const Fragment_post_swizzle &frag) {
        static_assert(std::is_same< Fragment_post_swizzle,
                                    Fragment_epilogue_interleaved_post_swizzle<Traits, Cta_tile> >::value,
                      "Fragment_post_swizzle must be interleaved_post_swizzle");
        this->reg(0) = frag.reg(0);
        this->reg(1) = frag.reg(1);
    }

    // The bias is added later.
    template< typename Fragment_bias>
    inline __device__ void add_bias(const Fragment_bias &bias_) {
    }

    // The bias+relu.
    template< typename Fragment_bias >
    inline __device__ void add_bias_relu(const Fragment_bias &bias_,
                                         int32_t with_relu,
                                         float relu_lb,
                                         float one) {
    }

    // The bias is added later.
    template< typename Fragment_bias >
    inline __device__ void add_bias_nchw(const Fragment_bias &bias, int i) {
    }

    // RELU activation.
    inline __device__ void relu(float) {
    }

    // Clip-RELU activation.
    inline __device__ void relu_ub(float) {
    }

    // Colwert to an int2.
    inline __device__ uint2 to_int2() const {
        return make_uint2(this->reg(0), this->reg(1));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Fragment_imma_int32_interleaved_c : public Fragment_imma_int8_c<Traits, Cta_tile> {

    // Compute the sum between two fragments.
    template<typename Fragment_post_swizzle>
    inline __device__ void pack(Fragment<float, Fragment_post_swizzle::NUM_REGS>,
                                const Fragment_post_swizzle &frag) {
        int32_t tmp[4];
        #pragma unroll
        for( int ii = 0; ii < Fragment_post_swizzle::NUM_REGS/4; ++ii ) {
            tmp[0] = f2i(frag.elt(4 * ii    ));
            tmp[1] = f2i(frag.elt(4 * ii + 1));
            tmp[2] = f2i(frag.elt(4 * ii + 2));
            tmp[3] = f2i(frag.elt(4 * ii + 3));
            this->reg(ii) = pack_int8x4(tmp);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Fragment_ampere_imma_interleaved_int32_epilogue_fadd_pre_swizzle
    : public Fragment<float, 16> {

    // The aclwmulators.
    using Aclwmulators = Fragment_aclwmulator<Traits>;

    // Compute the sum between two fragments.
    inline __device__ void add(
        const Fragment_ampere_imma_interleaved_int32_epilogue_fadd_pre_swizzle &other) { }

    // Quantize the aclwmulators -- actually simply do a scaled copy.
    inline __device__ void colwert(Fragment<float, 16> &alpha, const Aclwmulators &acc) {

        // This is for per-tensor scaling.
        #pragma unroll
        for( int ii = 0; ii < Aclwmulators::NUM_REGS; ++ii ) {
            int map_ii = (ii % 8) / 2 * 4 + ii % 2 + (ii / 8) % 2 * 2;
            this->elt(ii) = reinterpret_cast<const float&>(acc.elt(map_ii)) - 12582912.0f;
            this->elt(ii) = alpha.elt(ii) * this->elt(ii);
        }
    }

    inline __device__ void shuffle_groups(Aclwmulators &acc) {
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename Traits, typename Cta_tile >
struct Fragment_imma_epilogue_fadd_int32_interleaved_c : public Fragment_imma_int8_c<Traits, Cta_tile> {

    // The post swizzle fragment.
    using Fragment_post_swizzle = Fragment_epilogue_interleaved_post_swizzle<Traits, Cta_tile>;

    // Compute the sum between two fragments.
    inline __device__ void pack(Fragment<float, Fragment_post_swizzle::NUM_REGS>,
                                Fragment_post_swizzle &frag) {
        #pragma unroll
        for( int i = 0; i < Fragment_post_swizzle::NUM_REGS / 4; ++i ) {
            float4 f;
            f.x = frag.elt( 4 * i + 0 );
            f.y = frag.elt( 4 * i + 1 );
            f.z = frag.elt( 4 * i + 2 );
            f.w = frag.elt( 4 * i + 3 );

            this->reg( i ) = float4_to_s8x4( f );
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace xmma
