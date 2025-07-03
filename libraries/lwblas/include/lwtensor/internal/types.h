#pragma once

#include <cassert>

#include <algorithm>
#include <array>
#include <bitset>
#include <string>
#include <utility>

#include <lwda_fp16.h>
#include <lwComplex.h>
#include <lwda_runtime.h>

#include <lwtlass/complex.h>
#include <lwtensor/types.h>
#include <lwtensor/internal/initializable.h>
#include <lwtensor/internal/deviceProp.h>
#include <lwtensor/internal/defines.h>
#include <lwtensor/internal/lwtensor.h>

namespace LWTENSOR_NAMESPACE
{ 
    class Context; 

    typedef int32_t mode_type; /**< Data type of tensor modes is int32_t */
    typedef int64_t stride_type; /**< Data type of tensor strides is int32_t */
    typedef int32_t extent_type; /**< Data type of tensor extents is int32_t */

    static constexpr mode_type LWTENSOR_FIRST_INTERNAL_MODE = kMaxNumModesExternal + 4; /**< Reserved mode for internal use */
    static constexpr mode_type LWTENSOR_ILWALID_MODE = LWTENSOR_FIRST_INTERNAL_MODE; /**< Reserved mode for internal use */
    static constexpr mode_type RESERVED_M_MODE = LWTENSOR_FIRST_INTERNAL_MODE + 1; /**< Reserved mode for internal use */
    static constexpr mode_type RESERVED_N_MODE = LWTENSOR_FIRST_INTERNAL_MODE + 2; /**< Reserved mode for internal use */
    static constexpr mode_type RESERVED_K_MODE = LWTENSOR_FIRST_INTERNAL_MODE + 3; /**< Reserved mode for internal use */
    static constexpr mode_type RESERVED_L_MODE = LWTENSOR_FIRST_INTERNAL_MODE + 4; /**< Reserved mode for internal use */
    static constexpr mode_type RESERVED_M_MODE_PW = LWTENSOR_FIRST_INTERNAL_MODE + 5; /**< Reserved mode for internal use */
    static constexpr mode_type RESERVED_N_MODE_PW = LWTENSOR_FIRST_INTERNAL_MODE + 6; /**< Reserved mode for internal use */
    static constexpr mode_type RESERVED_A_MODE_PW = LWTENSOR_FIRST_INTERNAL_MODE + 7; /**< Reserved mode for internal use */
    static constexpr mode_type RESERVED_B_MODE_PW = LWTENSOR_FIRST_INTERNAL_MODE + 8; /**< Reserved mode for internal use */
    static constexpr mode_type RESERVED_C_MODE_PW = LWTENSOR_FIRST_INTERNAL_MODE + 9; /**< Reserved mode for internal use */
    static constexpr mode_type RESERVED_GROUP_MODE = LWTENSOR_FIRST_INTERNAL_MODE + 10; /**< Reserved mode for internal use */
    static constexpr mode_type LWTENSOR_LAST_INTERNAL_MODE = LWTENSOR_FIRST_INTERNAL_MODE + 11; /**< Reserved mode for internal use */

    static const int kUpperBoundModes = LWTENSOR_LAST_INTERNAL_MODE;

    struct ModeList
    {
        typedef std::array<mode_type, kUpperBoundModes> Storage;
        Storage storage_;
        size_t size_ = 0;
        typedef Storage::iterator iterator;
        typedef Storage::const_iterator const_iterator;

        ModeList() {}
        ModeList(std::initializer_list<mode_type> list)
        {
            for (auto elem : list)
            {
                push_back(elem);
            }
        }
        ModeList(const ModeList& other) : size_(other.size_)
        {
            std::copy(other.begin(), other.end(), storage_.begin());
        }
        ModeList& operator=(const ModeList& other)
        {
            if (this != &other)
            {
                size_ = other.size_;
                std::copy(other.begin(), other.end(), storage_.begin());
            }
            return *this;
        }

        mode_type& back() { return storage_[size_ - 1]; }
        const mode_type& back() const { return storage_[size_ - 1]; }

        const_iterator begin() const { return storage_.begin(); }
        iterator begin() { return storage_.begin(); }

        const_iterator cbegin() const { return storage_.cbegin(); }

        const_iterator cend() const { return storage_.cbegin() + size_; }

        bool empty() const { return size_ == 0; }

        iterator end() { return storage_.begin() + size_; }
        const_iterator end() const { return storage_.begin() + size_; }

        iterator erase(iterator pos)
        {
            assert(size_ >= 1);
            for (size_t at = pos - begin(); at < size_ - 1; at++)
            {
                storage_[at] = storage_[at + 1];
            }
            size_ -= 1;
            return pos;
        }

        mode_type& front() { return storage_.front(); }
        const mode_type& front() const { return storage_.front(); }

        iterator insert(iterator pos, mode_type value)
        {
            assert(pos <= end());
            for (auto at = end(); at != pos; at--) {
                *at = *(at - 1);
            }
            *pos = value;
            size_ += 1;
            assert(size_ <= storage_.max_size());
            return pos;
        }

        void push_back(mode_type value)
        {
            insert(end(), value);
        }

        template< class... Args >
        void emplace_back( Args&&... args )
        {
            push_back(mode_type(args...));
        }

        void push_front(mode_type value) { insert(begin(), value); }

        void remove(mode_type value) 
        {
            size_t new_size = 0;
            for (size_t at = 0; at < size_; at++)
            {
                if (storage_[at] != value) {
                    storage_[new_size] = storage_[at];
                    new_size += 1;
                }
            }
            size_ = new_size;
        }

        size_t size() const { return size_; }

        void clear() {
            size_ = 0;
        }

        void reserve(size_t size)
        {
            (void) size;
        }

        mode_type* data() { return storage_.data(); }
        const mode_type* data() const { return storage_.data(); }
    };

    struct ModeRenamer
    {
        ModeList rename(const mode_type* modes, uint32_t num)
        {
            ModeList result;
            for (uint32_t i = 0; i < num; i++)
            {
                auto it = std::find(map_.begin(), map_.end(), modes[i]);
                if (it == map_.end())
                {
                    map_.push_back(modes[i]);
                    it = std::prev(map_.end());
                }
                result.push_back(it - map_.begin());
            }
            return result;
        }
        bool valid()
        {
            return map_.size() <= LWTENSOR_FIRST_INTERNAL_MODE;
        }
        ModeList map_;
    };

    template<typename V>
    struct ModeMap
    {
        std::array<V, kUpperBoundModes> storage_;
        std::bitset<kUpperBoundModes> oclwpied_;
        size_t size_ = 0;

        template<typename U, typename P>
        struct Iterator
        {
            P* parent_;
            mode_type pos_;
            struct Value
            {
                mode_type first;
                typename std::remove_const<U>::type second;
            };
            mutable Value value_;

            Iterator(P* parent, mode_type pos) : parent_(parent), pos_(pos) {}

            // Prefix ++ overload 
            Iterator& operator++() 
            {
                while (pos_ < kUpperBoundModes)
                {
                    pos_ += 1;
                    if (pos_ == kUpperBoundModes || parent_->oclwpied_[pos_]) break;
                }
                return *this; 
            } 
  
            // Postfix ++ overload 
            Iterator operator++(int) 
            { 
                Iterator iterator = *this; 
                ++*this; 
                return iterator; 
            } 
  
            bool operator==(const Iterator& other) const 
            { 
                return parent_ == other.parent_ && pos_ == other.pos_;
            } 
  
            bool operator!=(const Iterator& other) const
            { 
                return ! this->operator==(other);
            } 
  
            const Value& operator*() const
            {
                value_.first = pos_;
                assert(pos_ >= 0 && pos_ < kUpperBoundModes);
                value_.second = parent_->storage_.at(pos_);
                return value_;
            } 

            const Value* operator->() const
            {
                value_.first = pos_;
                assert(pos_ >= 0 && pos_ < kUpperBoundModes);
                value_.second = parent_->storage_.at(pos_);
                return &value_;
            }
        };

        typedef Iterator<V, ModeMap<V>> iterator;
        typedef Iterator<const V, const ModeMap<V>> const_iterator;

        ModeMap() { }

        mode_type findSmallestOclwpiedMode() const
        {
            mode_type result = 0;
            while (result < kUpperBoundModes)
            {
                if (oclwpied_[result]) break;
                result += 1;
            }
            return result;
        }

        iterator end() { return {this, kUpperBoundModes}; }
        const_iterator end() const { return {this, kUpperBoundModes}; }
        iterator begin() { return {this, findSmallestOclwpiedMode()}; }
        const_iterator begin() const { return {this, findSmallestOclwpiedMode()}; }

        iterator find(mode_type key)
        {
            if (key < 0 || key >= kUpperBoundModes || ! oclwpied_[key])
            {
                return end();
            }
            return {this, key};
        }
        const_iterator find(mode_type key) const
        {
            if (key < 0 || key >= kUpperBoundModes || ! oclwpied_[key])
            {
                return end();
            }
            return {this, key};
        }

        V& at(mode_type key)
        {
            assert(key >= 0 && key < kUpperBoundModes);
            assert(oclwpied_[key]);
            return storage_[key];
        }
        const V& at(mode_type key) const
        {
            assert(key >= 0 && key < kUpperBoundModes);
            assert(oclwpied_[key]);
            return storage_[key];
        }

        size_t size() const { return size_; }

        void clear()
        {
            size_ = 0;
            oclwpied_.reset();
        }

        std::pair<iterator, bool> insert(const std::pair<mode_type, V> &value)
        {
            assert(value.first >= 0 && value.first < kUpperBoundModes);
            if (oclwpied_[value.first])
            {
                return {{this, value.first}, false};
            }
            oclwpied_[value.first] = true;
            storage_[value.first] = value.second;
            size_ += 1;            
            return {{this, value.first}, true};
        }

        V& operator[](mode_type key)
        {
            auto it = find(key);
            if (it == end())
            {
                insert({key, 0});
            }
            assert(key >= 0 && key < kUpperBoundModes);
            return storage_.at(key);
        }

        size_t erase(mode_type pos)
        {
            assert(pos >= 0 && pos < kUpperBoundModes);
            if (! oclwpied_[pos]) return 0;
            oclwpied_[pos] = false;
            size_ -= 1;
            return 1;
        }

        void reserve(size_t size) { (void) size; }
    };

    typedef ModeMap<stride_type> StrideMap;
    typedef ModeMap<extent_type> ExtentMap;

    /**
     * \ingroup runtimeDataStructurePLC3
     * \brief This struct contains all data types of an elementwise operation 
     */
    struct ElementwiseTypePack
    {
        /// Data type of tensor A
        lwdaDataType_t typeA_;
        /// Data type of tensor B
        lwdaDataType_t typeB_;
        /// Data type of tensor C
        lwdaDataType_t typeC_;
        /// Data type in which all the computation happens (input values will be translated into this data type)
        lwdaDataType_t typeCompute_;
    };

    /**
     * \ingroup runtimeDataStructurePLC3
     * \brief This struct encodes all unary and binary operators the elementwise kernel will perform.
     */ 
    struct ElementwiseOpPack
    {
        /// Unary operator applied to each element of tensor A
        lwtensorOperator_t opA_;
        /// Unary operator applied to each element of tensor B
        lwtensorOperator_t opB_;
        /// Unary operator applied to each element of tensor C 
        lwtensorOperator_t opC_;
        /// Binary operator applied to a pair of elements from tensor A and B
        lwtensorOperator_t opAB_;
        /// Unary operator applied in opAB
        lwtensorOperator_t opUnaryAfterBinary_;
        /// Binary operator applied to a pair of elements from the result of opAB and tensor C
        lwtensorOperator_t opABC_; 

        inline constexpr bool operator == (const ElementwiseOpPack &other) const
        {
            return (other.opA_ == opA_) &&
                   (other.opB_ == opB_) &&
                   (other.opC_ == opC_) &&
                   (other.opAB_ == opAB_) &&
                   (other.opUnaryAfterBinary_ == opUnaryAfterBinary_) &&
                   (other.opABC_ == opABC_);
        }
    };

    /**
     * \brief This struct encapsulates all information related to accessing single entries of the tensors w.r.t. the element-wise kernel.
     *
     * \details This struct encapsulates all information related to accessing single entries of the
     * tensors; to this end this struct stores the extent and stride of each mode as well as the order in which the
     * modes are processed within the element-wise kernels.
     * \ingroup runtimeDataStructurePLC3
     *
     * \req None
     */
    class ElementwiseParameters
    {
        public: 
            /**
             * \param[in] mode_order This list encodes the order in which the modes will be
             * traversed (processed) inside the LWCA kernel. The mode-order is both important
             * for performance reasons (since it affects the spatial locality) as well as for
             * correctness reasons (since the blocked modes (if any) _must_ be the first
             * modes).
             * \param[in] extent Holds the extent for each mode in mode_order.
             * \param[in] modeA modes of A (sorted w.r.t. strides they exhibit w.r.t. A)
             * \param[in] modeB modes of B (sorted w.r.t. strides they exhibit w.r.t. B)
             * \param[in] modeC modes of C (sorted w.r.t. strides they exhibit w.r.t. C)
             * \param[in] strideA holds the strides of each mode in modeA. This map may be empty,
             * in that case a generalized column-major memory layout is assumed (i.e., modeA.front()
             * is the stride-1 mode, and modeA.back() is the slowest-varying mode).
             * \param[in] strideB similar to strideA but w.r.t. B
             * \param[in] strideC similar to strideA but w.r.t. C
             * \param[in] info Holds all necessary information related to vectorization.
             *
             * \req None
             * \pre modeA all modes must be sorted w.r.t. the strides in A
             * \pre modeB all modes must be sorted w.r.t. the strides in B
             * \pre modeC all modes must be sorted w.r.t. the strides in C
             * \pre The blocked modes (if any) must be the first modes in 'mode_order'.
             * \changes_elw None
             * \exception-guarantee nothrow
             * \behavior blocking, not reentrant, and thread safe
             */
            lwtensorStatus_t init(
                    const Context* ctx,
                    const ModeList &mode_order,
                    const ExtentMap &extent,
                    const bool useA, lwdaDataType_t typeA, const ModeList &modeA, lwtensorOperator_t opA,
                    const bool useB, lwdaDataType_t typeB, const ModeList &modeB, lwtensorOperator_t opB,
                    const bool useC, lwdaDataType_t typeC, const ModeList &modeC, lwtensorOperator_t opC, lwtensorOperator_t opAB, lwtensorOperator_t opABC, lwdaDataType_t typeCompute,
                    const uint32_t alignmentRequirementA,
                    const uint32_t alignmentRequirementB,
                    const uint32_t alignmentRequirementC,
                    const uint32_t alignmentRequirementD,
                    const StrideMap &strideA_tmp,
                    const StrideMap &strideB_tmp,
                    const StrideMap &strideC_tmp);

        /// maximum number of modes (aka dimensions) supported by this data structure.
        static constexpr uint32_t LWTENSOR_MAX_MODES = LWTENSOR_NAMESPACE::kMaxNumModes;
        /// number of modes of C
        uint32_t nmodeC_ = 0;
        /// Extents for each mode. This array has nmodeC entries.
        extent_type extent_[LWTENSOR_MAX_MODES];
        /**
         *  \brief This structure encapsulates the layout of a vectorized tensor.
         *  \ingroup runtimeDataStructurePLC3
         */ 
        /// strides; the i-th entry corresponds to the i-th entry in extent
        stride_type strideA_[LWTENSOR_MAX_MODES];
        stride_type strideB_[LWTENSOR_MAX_MODES];
        stride_type strideC_[LWTENSOR_MAX_MODES];

        uint32_t alignmentRequirementA_ = 128;
        uint32_t alignmentRequirementB_ = 128;
        uint32_t alignmentRequirementC_ = 128;
        uint32_t alignmentRequirementD_ = 128;

        bool useA_ = false;
        bool useB_ = false;
        bool useC_ = false;

        bool isStrideOne_ = false;

        /// Additional context for some activation functions
        union ActivationContext
        {
            struct {
                float alpha;
            } elu;
            struct {
                float k;
            } leakyRelu;
            struct {
                float eluAlpha;
                float outScale;
            } selu;
            struct {
                float inScale;
                float outScale;
            } scaledTanh;
            struct {
                float threshold;
            } thresholdedRelu;
            struct {
                float upper;
                float lower;
            } clip;
            struct {
                float inScale;
                float outScale;
                float approximateThreshold;
            } softPlus;
            struct {
                float slope;
                float shift;
            } hardSigmoid;
        };
        ActivationContext activationContext;

        ElementwiseTypePack typePack_;

        ElementwiseOpPack opPack_;
    };

    /**
     * \brief Callwlates the number of tiles for a given blocking.
     *
     * \details This function callwlates the total number of tiles for the provided blocking;
     * all tiles can be independently computed by different CTAs.
     * For instance, let's assume that we want to transpose a tensor A \in
     * R^{64x64x64} with a blocking of 32x32 then this function would return 64/32 * 64/32 * 64 total tiles.
     *
     * \param[in] nmodes_blocked number of entries in array 'blocking'
     * \param[in] blocking array of size 'nmodes_blocked' storing the tile size.
     * \returns total number of tiles for the given blocking w.r.t. this element-wise operation.
     * \req None
     * \pre blocking must contain at least sizeof(uint32_t) * numModesC
     * \changes_elw None
     * \throw out_of_range exceptions if ElementwiseParameters failed to initialize
     * \throw internal exception if thr total strides exceed the limit of stride_type
     * \exception-guarantee basic
     * \behavior blocking, not reentrant, and thread safe
     */
    stride_type getTotalTiles( const ElementwiseParameters & param, const uint32_t nmodes_blocked,
            const uint32_t * const blocking);

    inline std::string lwdaDataTypeToString(lwdaDataType_t typeA)
    {
        switch ( typeA )
        {
            case LWDA_R_8U:
                return std::string("k");
            case LWDA_R_8I:
                return std::string("j");
            case LWDA_R_32U:
                return std::string("u");
            case LWDA_R_32I:
                return std::string("i");
            case LWDA_R_16F:
                return std::string("h");
            case LWDA_R_32F:
                return std::string("s");
            case LWDA_R_64F:
                return std::string("d");
            case LWDA_C_32F:
                return std::string("c");
            case LWDA_C_64F:
                return std::string("z");
            default:
                return std::string("UNKNOWN type");
        }
    }

    /**
     * \brief This struct encodes the parameters used to select the proper element-wise kernel.
     *
     * \details This struct encodes the parameters used to select the proper element-wise kernel as well as most
     * of the kernel's parameters. Depending on the arguments of this struct, a certain
     * instantiation of either the tensor_elementwise_kernel() or the
     * vectorized_tensor_elementwise_kernel() kernel is launched; 
     *
     * D = binaryOp2( unaryOp1( binaryOp1( unaryOpA(A), unaryOpB(B) ) ), unaryOpC(C) ),
     *
     * This struct will be initialized and serialized as part of TRT's build time and
     * finally deserialized as part of TRT's runtime.
     * \ingroup runtimeDataStructurePLC3
     *
     * \req This struct must be serializable.
     */
    struct ElementwisePlan
    {
        std::string toString() const noexcept;
 
        /// Determines whether the user provided A and B should be swapped (before entering the kernel)
        bool swapAB_ = false;
        /// Determines whether the user provided A and C should be swapped (before entering the kernel)
        bool swapAC_ = false;
        /// This struct encodes most of the parameters that are passed into the kernel
        struct ElementwiseParameters params_; 

        int32_t containerIdx_ = -1;
        int32_t candidateIdx_ = -1;
    };

    /**
     * \ingroup runtimeDataStructurePLC3
     * \param[in] T the generic data type
     * \return the lwdaDataType_t of an generic data type.
     */
    template<typename T>
    inline lwdaDataType_t toLwdaDataType();

    template<>
    inline lwdaDataType_t toLwdaDataType<half>()
    {
        return LWDA_R_16F;
    }

    template<>
    inline lwdaDataType_t toLwdaDataType<lwtlass::half_t>()
    {
        return LWDA_R_16F;
    }

    #if LWTENSOR_LWDA_VERSION_MAJOR >= 11
    template<>
    inline lwdaDataType_t toLwdaDataType<BFloat16>()
    {
        return LWDA_R_16BF;
    }
    template<>
    inline lwdaDataType_t toLwdaDataType<lwtlass::bfloat16_t>()
    {
        return LWDA_R_16BF;
    }
    template<>
    inline lwdaDataType_t toLwdaDataType<lwtlass::tfloat32_t>()
    {
        return LWDA_R_TF32;
    }
    #endif

    template<>
    inline lwdaDataType_t toLwdaDataType<float>()
    {
        return LWDA_R_32F;
    }

    template<>
    inline lwdaDataType_t toLwdaDataType<double>()
    {
        return LWDA_R_64F;
    }

    template<>
    inline lwdaDataType_t toLwdaDataType<lwComplex>()
    {
        return LWDA_C_32F;
    }

    template<>
    inline lwdaDataType_t toLwdaDataType<lwtlass::complex<float>>()
    {
        return LWDA_C_32F;
    }

    template<>
    inline lwdaDataType_t toLwdaDataType<lwDoubleComplex>()
    {
        return LWDA_C_64F;
    }

    template<>
    inline lwdaDataType_t toLwdaDataType<lwtlass::complex<double>>()
    {
        return LWDA_C_64F;
    }

    template<>
    inline lwdaDataType_t toLwdaDataType<int32_t>()
    {
        return LWDA_R_32I;
    }

    template<>
    inline lwdaDataType_t toLwdaDataType<int8_t>()
    {
        return LWDA_R_8I;
    }

    /**
     * \ingroup runtimeDataStructurePLC3
     * \brief This struct contains all constant data types of a suported elementwise operation
     * \param kMinSM minimal SM number (e.g., 70 for gv100, 80 for ga100)
     * \param kMaxSM minimal SM number (e.g., 70 for gv100, 80 for ga100)
     */
    template<typename TypeA, typename TypeB, typename TypeC, typename TypeCompute, int kTargetCC, int kMinCC, int kMaxCC>
    struct ElementwiseTraits
    {
        using TypeA_ = TypeA;
        using TypeB_ = TypeB;
        using TypeC_ = TypeC;
        using TypeCompute_ = TypeCompute;
        static const int targetCC_ = kTargetCC;

        static lwtensorStatus_t isApplicable(const DeviceProp* deviceProp, const ElementwiseParameters &params)
        {
            const int computeCapability = deviceProp->major * 10 + deviceProp->minor;
            (void)computeCapability; // not used yet

            return((computeCapability >= kMinCC) &&
                   (computeCapability <= kMaxCC) &&
                   (params.typePack_.typeA_ == toLwdaDataType<TypeA_>()) &&
                   (params.typePack_.typeB_ == toLwdaDataType<TypeB_>()) && 
                   (params.typePack_.typeC_ == toLwdaDataType<TypeC_>()) && 
                   (params.typePack_.typeCompute_ == toLwdaDataType<TypeCompute_>()))
                ? LWTENSOR_STATUS_SUCCESS : LWTENSOR_STATUS_NOT_SUPPORTED;
        }
    };

    /**
     * \ingroup runtimeDataStructurePLC3
     * \brief This struct encodes all unary and binary operators the elementwise kernerl will perform.
     */ 
    template<
        lwtensorOperator_t opA,
        lwtensorOperator_t opB,
        lwtensorOperator_t opC,
        lwtensorOperator_t opAB,
        lwtensorOperator_t opUnaryAfterBinary,
        lwtensorOperator_t opABC>
    struct ElementwiseStaticOpPack
    {
        /// Unary operator applied to each element of tensor A
        constexpr static lwtensorOperator_t opA_ = opA;
        /// Unary operator applied to each element of tensor B
        constexpr static lwtensorOperator_t opB_ = opB;
        /// Unary operator applied to each element of tensor C 
        constexpr static lwtensorOperator_t opC_ = opC;
        /// Binary operator applied to a pair of elements from tensor A and B
        constexpr static lwtensorOperator_t opAB_ = opAB;
        /// Unary operator applied in opAB
        constexpr static lwtensorOperator_t opUnaryAfterBinary_ = opUnaryAfterBinary;
        /// Binary operator applied to a pair of elements from the result of opAB and tensor C
        constexpr static lwtensorOperator_t opABC_ = opABC;

        /*
         * \ingroup runtimeDataStructurePLC3
         * \breif Decide whether the input operator pack is the same as the constant operator pack.
         * \param[in] opPack a pack of operators to be performed in the elementwise kernel
         * \return true if the input operators are the same as the constant operators
         * \changes_elw None.
         * \behavior blocking, reentrant, thread safe
         */
        constexpr bool operator == (const ElementwiseOpPack opPack) const
        {
            return (opPack.opA_ == opA_) && 
                   (opPack.opB_ == opB_) && 
                   (opPack.opC_ == opC_) && 
                   (opPack.opAB_ == opAB_) && 
                   (opPack.opUnaryAfterBinary_ == opUnaryAfterBinary_) && 
                   (opPack.opABC_ == opABC_);
        }

        static ElementwiseOpPack toElementwiseOpPack()
        {
            ElementwiseOpPack opPack;
            opPack.opA_   = opA_;
            opPack.opB_   = opB_;
            opPack.opC_   = opC_;
            opPack.opAB_  = opAB_;
            opPack.opUnaryAfterBinary_ = opUnaryAfterBinary_;
            opPack.opABC_ = opABC_;
            return opPack;
        }
    };

    template<uint32_t NDIM_TILE_, extent_type TILE_SIZE_0, extent_type TILE_SIZE_1, extent_type TILE_SIZE_2, uint32_t NUM_THREADS_, uint32_t VEC_, bool TRANSPOSE_ = true,
        typename OpPack = ElementwiseStaticOpPack<
            lwtensorOperator_t::LWTENSOR_OP_UNKNOWN,
            lwtensorOperator_t::LWTENSOR_OP_UNKNOWN,
            lwtensorOperator_t::LWTENSOR_OP_UNKNOWN,
            lwtensorOperator_t::LWTENSOR_OP_UNKNOWN,
            lwtensorOperator_t::LWTENSOR_OP_UNKNOWN,
            lwtensorOperator_t::LWTENSOR_OP_UNKNOWN>,
        int LAUNCH_BOUNDS_MIN_CTA_ = 0 >
    struct ElementwiseConfig3D
    {
        /* This config encodes a NDIM hyper-square tile of size TILE_SIZE^NDIM */
        constexpr static uint32_t VEC = VEC_;
        /// Dimensionality of the tile (lwrrently limited to 1D and 2D)
        constexpr static uint32_t NDIM_TILE = NDIM_TILE_;
        /// Tile-size (e.g., one tile hase TILE_SIZE^(NDIM_TILE) many elements)
        //constexpr static extent_type TILE_SIZE = TILE_SIZE_;
        /// number of threads per CTA
        constexpr static uint32_t NUM_THREADS = NUM_THREADS_;
        constexpr static extent_type BLOCKING[3] = {TILE_SIZE_0, (NDIM_TILE>1) ? TILE_SIZE_1 : 1, (NDIM_TILE>2) ? TILE_SIZE_2 : 1};
        static_assert(NUM_THREADS % 32 == 0,"32");
        static_assert((BLOCKING[0] * BLOCKING[1] * BLOCKING[2]) % (VEC*NUM_THREADS) == 0,"#elements must be divisible by #threads");
        //static_assert(NDIM_TILE <= 2 ,"3D disabled");
        static_assert((TILE_SIZE_0 & (TILE_SIZE_0 - 1U)) == 0U ,"TILE_SIZE must be power of two"); // this constraint can be resolved easily (by replacing fast divide in kernel by / )
        static_assert((TILE_SIZE_1 & (TILE_SIZE_1 - 1U)) == 0U ,"TILE_SIZE must be power of two"); // this constraint can be resolved easily (by replacing fast divide in kernel by / )
        static_assert((TILE_SIZE_2 & (TILE_SIZE_2 - 1U)) == 0U ,"TILE_SIZE must be power of two"); // this constraint can be resolved easily (by replacing fast divide in kernel by / )

        using OpPack_ = OpPack;

        constexpr static bool TRANSPOSE = NDIM_TILE == 1 ? false : TRANSPOSE_;
        constexpr static int LAUNCH_BOUNDS_MIN_CTA = LAUNCH_BOUNDS_MIN_CTA_;
    };

    /**
     * \brief This data structure describes the (physical) layout of a tensor.
     *
     * \details This data structure describes the (physical) layout of a tensor, it encapsulates
     *          information such as the number of modes, the extent and stride of each mode as well
     *          as information about the vectorization of the tensor (e.g., vector index,
     *          vector-width).
     * \req None
     * \Ilwariants None
     */
    class TensorDescriptor : public Initializable<31>
    {
        public:
        /// Maximum number of modes for any tensor (public-facing)
        static constexpr uint32_t LWTENSOR_MAX_MODES_EXTERNAL = LWTENSOR_NAMESPACE::kMaxNumModesExternal;
        /// Maximum number of modes for any tensor (we need more modes internally due to vectorization)
        static constexpr uint32_t LWTENSOR_MAX_MODES = LWTENSOR_MAX_MODES_EXTERNAL + 4U;

        /**
         * \param[in] numModes Number of modes (0< numModes <= 8)
         * \param[in] extent Array of size numModes that stores the extent of each mode (excluding vectorization).
         * \param[in] stride Array of size numModes that stores the stride of each mode in elements (thus, not including the vector-width); if
         *                   NULL, then a packed data-layout is assumed (i.e., stride[0] = 1, stride[1] =
         *                   extent[0], stride[2] = stride[1] * extent[1], ...)
         * \param[in] dataType The data type of the tensor
         * \param[in] op Unary operator that will be applied to each element of the
         * respective tensor. Please note, this operator is applied in a lazy fashion at
         * the time that elements of this tensor are loaded (from within a kernel) into
         * shared-memory; thus, the original tensor elements will NOT be affected.
         * \param[in] vectorWidth The amount of elements that are store conselwtively along
         *           the vectorized mode (see vectorMode for more details
         *                        on vectorization); this is the unit of elements that extent
         *                        and stride are measured in.
         * \param[in] vectorModeIndex Stores the information about which mode is vectorized.
         *                       A vectorwidth of y along the x-th mode means that the x-th mode has a total of
         *                       extent[x] elements with (at least) y elements being stored
         *                       contiguously in memroy. The value of vectorModeIndex must be less than numModes.
         *                       For instance, a tensor A_{a,b,c} with vectorModeIndex=1 and vectorWidth=4 would
         *                       vectorize A along the b-mode, effectively turning A into a 4D tensor A_{b0,a,b1,c}
         *                       with extent(b0) = 4, extent(b1) = extent(b) / extent(b0).
         * \req None
         * \pre extent and stride (if not nullptr) must contain at least sizeof(extent_type) * numModes and
         *      sizeof(stride_type) * numModes bytes
         * \changes_elw None
         * \throw not supported exception if not all inputs satisfy the requirements
         * \exception-guarantee basic
         * \behavior blocking, not reentrant, and thread safe
         */
        TensorDescriptor(
                const uint32_t numModes,
                const extent_type * const extent,
                const stride_type * const stride = nullptr,
                const lwdaDataType_t dataType = LWDA_R_32F,
                const lwtensorOperator_t op = lwtensorOperator_t::LWTENSOR_OP_IDENTITY,
                const uint8_t vectorWidth = 1U,
                const uint32_t vectorModeIndex = 0U);

        TensorDescriptor(
                const uint32_t numModes,
                const extent_type * const extent,
                const stride_type * const stride,
                const lwdaDataType_t dataType,
                const lwtensorOperator_t op,
                const uint8_t vectorWidth,
                const uint32_t vectorModeIndex,
                const uint8_t  vectorOffset,
                const bool zeroPadding);


        TensorDescriptor() = delete;

        /**
         * \brief Copy constructor (deep copy)
         * \param[in] other the constant reference of the target
         * \req None
         * \pre the target to copy must has been successfully constructed
         * \changes_elw None
         * \exception-guarantee nothrow
         * \behavior blocking, not reentrant, and thread safe
         */
        TensorDescriptor(const TensorDescriptor &other);

        /**
         * \brief Copy assignment (deep copy)
         * \param[in] other the constant reference of the target
         * \req None
         * \pre the target to copy must have been successfully declared
         * \changes_elw None
         * \exception-guarantee nothrow
         * \behavior blocking, not reentrant, and thread safe
         */
        TensorDescriptor& operator=(const TensorDescriptor &other);   

        /**
         * \brief Init the TensorDescriptor descriptor
         * \param[in] ctx pointer to initialized lwTENSOR context
         * \param[in] numModes Number of modes (0< numModes <= 8)
         * \param[in] extent Array of size numModes that stores the extent of each mode (excluding vectorization).
         * \param[in] stride Array of size numModes that stores the stride of each mode in elements (thus, not including the vector-width); if
         *                   NULL, then a packed data-layout is assumed (i.e., stride[0] = 1, stride[1] =
         *                   extent[0], stride[2] = stride[1] * extent[1], ...)
         * \param[in] dataType The data type of the tensor
         * \param[in] op Unary operator that will be applied to each element of the
         * respective tensor. Please note, this operator is applied in a lazy fashion at
         * the time that elements of this tensor are loaded (from within a kernel) into
         * shared-memory; thus, the original tensor elements will NOT be affected.
         * \param[in] vectorWidth The amount of elements that are store conselwtively along
         *           the vectorized mode (see vectorMode for more details
         *                        on vectorization); this is the unit of elements that extent
         *                        and stride are measured in.
         * \param[in] vectorModeIndex Stores the information about which mode is vectorized.
         *                       A vectorwidth of y along the x-th mode means that the x-th mode has a total of
         *                       extent[x] elements with (at least) y elements being stored
         *                       contiguously in memroy. The value of vectorModeIndex must be less than numModes.
         *                       For instance, a tensor A_{a,b,c} with vectorModeIndex=1 and vectorWidth=4 would
         *                       vectorize A along the b-mode, effectively turning A into a 4D tensor A_{b0,a,b1,c}
         *                       with extent(b0) = 4, extent(b1) = extent(b) / extent(b0).
         * \req None
         * \pre extent and stride (if not nullptr) must contain at least sizeof(extent_type) * numModes and
         *      sizeof(stride_type) * numModes bytes
         * \returns the error code
         * \changes_elw None
         * \exception-guarantee nothrow
         * \behavior blocking, not reentrant, and thread safe
         */
        lwtensorStatus_t init(
                                const Context* ctx,
                                const uint32_t numModes,
                                const extent_type * const extent,
                                const stride_type * const stride = nullptr,
                                const lwdaDataType_t dataType = LWDA_R_32F,
                                const lwtensorOperator_t op = lwtensorOperator_t::LWTENSOR_OP_IDENTITY,
                                const uint8_t vectorWidth = 1U,
                                const uint32_t vectorModeIndex = 0U) noexcept;

        /**
         * \brief Checks if two tensors have the same vectorization properties.
         * \req None
         * \pre the target to copy must has been successfully constructed
         * \returns True iff two tensors have the same vectorization properties, false otherwise.
         * \changes_elw None
         * \exception-guarantee nothrow
         * \behavior blocking, not reentrant, and thread safe
         */
        inline bool hasSameVectorization(const TensorDescriptor& that) const noexcept
        {
            if( (this->vectorWidth_  != that.vectorWidth_) ||
                (this->vectorModeIndex_   != that.vectorModeIndex_) ||
                (this->vectorOffset_ != that.vectorOffset_) ||
                (this->zeroPadding_  != that.zeroPadding_)) {
                return false;
            }
            return true;
        }

        /**
         * \brief Checks if the tensor is vectorized.
         * \req None
         * \returns True, iff the tensor is vectorized, false otherwise.
         * \changes_elw None
         * \exception-guarantee nothrow
         * \behavior blocking, not reentrant, and thread safe
         */
        inline bool isVectorized() const noexcept
        {
            return (this->vectorWidth_ > 1U) && (vectorModeIndex_ < numModes_);
        }

        /**
         * \brief Checks if two tensors are similar (i.e., only the strides of the two tensors may be
         * different)
         * \param[in] other the constant reference of the target
         * \req None
         * \pre the constructor did not fail
         * \returns True iff two tensors are similar, false otherwise.
         * \changes_elw None
         * \exception-guarantee nothrow
         * \behavior blocking, not reentrant, and thread safe
         */
        inline bool isSimilar(const TensorDescriptor& that) const
        {
            if( (this->numModes_ != that.numModes_) ||
                (this->op_ != that.op_ ) ||
                (this->dataType_ != that.dataType_) ) {
                return false;
            }
            for( uint32_t i = 0U; i < this->numModes_; i++ )
            {
                if( (this->getExtent(i) != that.getExtent(i)) )
                {
                    return false;
                }
            }
            return true;
        }

        /**
         * \breif Checks if two tensors are identical (i.e., if the are similar and have
         * identical strides).
         * \param[in] other the constant reference of the target
         * \req None
         * \pre the constructor did not fail
         * \returns True iff two tensors are identical, false otherwise.
         * \changes_elw None
         * \exception-guarantee nothrow
         * \behavior blocking, not reentrant, and thread safe
         */
        inline bool operator == (const TensorDescriptor& that) const
        {
            if ( (!this->isSimilar(that)) || (!this->hasSameVectorization(that)) )
            {
                return false;
            }
            // for equality we want additionally that the strides are actually the
            // same
            for( uint32_t i = 0U; i < this->numModes_; i++ )
            {
                if( this->stride_[i] != that.stride_[i] )
                {
                    return false;
                }
            }
            return true;
        }

        /**
         * \returns the extent of the tensor of the n-th dimension
         * \param[in] n the mode index
         * \req None
         * \pre None
         * \throw out_of_range exception
         * \exception-guarantee basic
         * \behavior blocking, not reentrant, and thread safe
         */
        inline extent_type getExtent(const uint32_t n) const
        {
            return extent_.at(static_cast<size_t>(n));
        }
        /**
         * \returns the extent of the tensor of the n-th dimension
         * \param[in] n the mode index
         * \req None
         * \pre None
         * \throw out_of_range exception
         * \exception-guarantee basic
         * \behavior blocking, not reentrant, and thread safe
         */
        inline stride_type getStride( const uint32_t n ) const
        {
            return stride_.at( static_cast<size_t>(n) );
        }

        /**
         * \returns the lwdaDataType of the tensor element
         * \req None
         * \pre None
         * \exception-guarantee nothrow
         * \behavior blocking, not reentrant, and thread safe
         */
        inline lwdaDataType_t getDataType() const noexcept
        {
            return dataType_;
        }
        // TODO @Albert please wrap these functions into ifdef DEVELOP
#ifdef DEVELOP
        /// WARNING: this function must be used with care and is only intended for testing purposes //TODO Remove for release?
        inline void setDataType(const lwdaDataType_t dataType)
        {
            dataType_ = dataType;
        }
#endif
        /**
         * \returns the number of modes of the tensor
         * \req None
         * \pre None
         * \exception-guarantee nothrow
         * \behavior blocking, not reentrant, and thread safe
         */
        inline uint32_t getNumModes() const noexcept
        {
            return numModes_;
        }
#ifdef DEVELOP
        /// WARNING: this function must be used with care and is only intended for testing purposes //TODO Remove for release?
        inline void setNumModes(const uint32_t numModes)
        {
            numModes_ = numModes;
        }
#endif
        /**
         * \returns the vector width of the vector mode
         * \req None
         * \pre None
         * \exception-guarantee nothrow
         * \behavior blocking, not reentrant, and thread safe
         */
        inline uint8_t getVectorWidth() const noexcept
        {
            return vectorWidth_;
        }

        /**
         * \return Returns index of vectorized mode; return 999 if no mode is vectorized.
         * \req None
         * \pre None
         * \exception-guarantee nothrow
         * \behavior blocking, not reentrant, and thread safe
         */
        inline uint32_t getVectorModeIndex() const noexcept
        {
            return vectorModeIndex_;
        }
        /**
         * \return modes[vectorModeIndex_] if vectorizied, LWTENSOR_ILWALID_MODE otherwise.
         * \req None
         * \pre The constructor does not fail
         * \exception-guarantee nothrow
         * \behavior blocking, not reentrant, and thread safe
         */
        inline mode_type getVectorMode(const mode_type* const modes) const
        {
            if (this->isVectorized())
            {
                return modes[this->getVectorModeIndex()];
            }
            else
            {
                return LWTENSOR_ILWALID_MODE;
            }
        }

        /**
         * \brief Set the unary operation while reading the elements of the tensor
         * \req None
         * \pre The constructor does not fail
         * \exception-guarantee nothrow
         * \behavior blocking, not reentrant, and thread safe
         */
        inline void setOp(const lwtensorOperator_t op) noexcept
        {
            this->op_ = op;
        }

        /**
         * \brief Get the unary operation while reading the elements of the tensor
         * \req None
         * \pre The constructor does not fail
         * \exception-guarantee nothrow
         * \behavior blocking, not reentrant, and thread safe
         */
        inline lwtensorOperator_t getOp() const noexcept
        {
            return op_;
        }

        /**
         * \brief Set the vector width and mode index
         * \req According to the requirments, the supporting widths are 1, 2, 4, 8, 16, and 32.
         * \pre The constructor does not fail
         * \returns the error code
         * \exception-guarantee nothrow
         * \behavior blocking, not reentrant, and thread safe
         */
        lwtensorStatus_t setVectorization(const uint8_t vectorWidth, const uint32_t vectorModeIndex) noexcept;

        /**
         * \returns the vector mode offset
         * \req According to the requirements, the offset must >= 0 and < vector-width
         * \pre The constructor does not fail
         * \returns the error code
         * \exception-guarantee nothrow
         * \behavior blocking, not reentrant, and thread safe
         */
        lwtensorStatus_t setVectorOffset(const uint8_t offset) noexcept;

        /**
         * \returns the vector mode offset
         * \req None
         * \pre The constructor does not fail
         * \exception-guarantee nothrow
         * \behavior blocking, not reentrant, and thread safe
         */
        inline uint8_t getVectorOffset() const noexcept
        {
            return vectorOffset_;
        }

        /**
         * \brief Activate the zero-padding on the vectorized output tensor
         * \req According to the requirments, zero-padding can be actived with this function.
         * \pre The constructor does not fail
         * \returns the error code
         * \exception-guarantee nothrow
         * \behavior blocking, not reentrant, and thread safe
         */
        lwtensorStatus_t setZeroPadding(const bool padding) noexcept;

        /**
         * \returns the true is zero-padding is activated
         * \req None
         * \pre The constructor does not fail
         * \exception-guarantee nothrow
         * \behavior blocking, not reentrant, and thread safe
         */
        inline bool getZeroPadding() const noexcept
        {
            return zeroPadding_;
        }

        /**
         * \brief Produces a permutation p such that stride[p[i]] <= stride[p[i+1]], for all \f$ i \in [0,1,...,numModes-1]\f$
         * \param[out] perm an integer array of size at least numModes
         *
         * \pre perm must be able to hold at least this->numModes_ many integers.
         * \pre The constructor does not fail
         * \req none
         * \pre the constructor does not fail
         * \exception-guarantee nothrow
         * \behavior blocking, not reentrant, and thread safe
         */
        void getStridePermutationAscending(std::array<uint32_t, LWTENSOR_MAX_MODES> &perm) const
        {
            for (uint32_t i = 0U; i < numModes_; i++)
            {
                perm[i] = i;
            }

            for (uint32_t start = 0U; (start + 1U) < numModes_; start++)
            {
                for (uint32_t i = 0U; (i + start + 1U) < numModes_; i++)
                {
                    if (stride_[perm[i]] > stride_[perm[i + 1U]])
                    {
                        std::swap(perm[i], perm[i + 1U]);
                    }
                }
            }
        }

    private:
        /// Number of modes in the tensor
        uint32_t numModes_;

        /// Data type of the elements
        lwdaDataType_t dataType_;

        /// extent[i] represents the number of elements along the i-th mode (irrespective of vectorization).
        std::array<extent_type, LWTENSOR_MAX_MODES> extent_;

        /// stride[i] represents the distance **in elements** to the next element along the i-th mode (irrespective of vectorization).
        std::array<stride_type, LWTENSOR_MAX_MODES> stride_;

        /// Unary operator that will be applied to each element (see contructor for details).
        lwtensorOperator_t op_;

        /// Number of conselwtive elements along the vectorized mode.
        uint8_t vectorWidth_;

        /// Index (into extent_ and stride_) representing the vectorized mode.
        uint32_t vectorModeIndex_;

        /// The offset of the vectorized mode between [ 0, vectorWidth ) if vectorized.
        uint8_t vectorOffset_;

        /// Determines if the vectorized mode should be zero-padded (only applicable if this tensor is the output).
        bool zeroPadding_;
    };

    // Check that TensorDescriptor fits in TensorDescriptor_t
    static_assert(sizeof(TensorDescriptor) <= sizeof(lwtensorTensorDescriptor_t),
                  "Size of TensorDescriptor greater than lwtensorTensorDescriptor_t");

    /**
     * This class establishes an order for contraction algorithms and its different
     * kernels
     */
    class Contractiolwariant
    {
        private:
            lwtensorAlgo_t algo_; /// determines the algorithm that is applied
            int32_t kernel_;          /// determines the kernel --for the selected-- algorithm (aka sub-algorithm)

        public:
            static constexpr int kDefaultKernel = -1;
            lwtensorStatus_t init(const lwtensorAlgo_t algo)
            {
                if (algo < 0)
                {
                    this->algo_ = algo;
                    this->kernel_ = kDefaultKernel;
                }
                else
                {
                    this->algo_ = LWTENSOR_ALGO_GETT;
                    this->kernel_ = algo;
                }
                return LWTENSOR_STATUS_SUCCESS;
            }

            lwtensorStatus_t setKernel(const int32_t kernel)
            {
                this->kernel_ = kernel;
                return LWTENSOR_STATUS_SUCCESS;
            }

            /**
             * Determines if the corresponding plan is cacheable.
             *
             * We choose to cache only algos for which one has to ilwoke the heuristic
             * (i.e., all algos_ < 0).
             */
            bool isCacheable() const { return algo_ < 0; }

            lwtensorAlgo_t getAlgo() const { return algo_; }
            int32_t getKernelValue() const { return kernel_; }
            int32_t getKernel() const { return kernel_; }

            bool isDefaultAlgo() const { return algo_ == LWTENSOR_ALGO_DEFAULT; }

            bool isGett() const { return algo_ >= 0 || algo_ == LWTENSOR_ALGO_GETT; }
            bool isTGett() const { return algo_ == LWTENSOR_ALGO_TGETT; }
            bool isTTGT() const { return algo_ == LWTENSOR_ALGO_TTGT; }

            /**
             * This function sets algo_ and kernel_ based on the provided algo; this
             * function defines an order based on algo.
             */
            lwtensorStatus_t setFromRank(const lwtensorAlgo_t algo, const uint32_t rank);

            static bool isDefaultKernel(const int32_t kernel) { return kernel == kDefaultKernel; }
    };

    class ContractionFind : public Initializable<43>
    {
        public:
            ContractionFind(){};
            lwtensorStatus_t init(const lwtensorAlgo_t algo)
            {
                contractiolwariant_.init(algo);
                this->autotuneMode_ = LWTENSOR_AUTOTUNE_NONE;
                this->cacheMode_ = LWTENSOR_CACHE_MODE_PEDANTIC;
                incrementalCount_ = 4;
                numPartitionsK_ = -1;

                this->setInitialized(); 
                return LWTENSOR_STATUS_SUCCESS;
            }


            lwtensorStatus_t setKernel(int kernel) { return contractiolwariant_.setKernel(kernel); }

            Contractiolwariant getContractiolwariant() const { return contractiolwariant_; }

            lwtensorStatus_t setAutotuneMode(const lwtensorAutotuneMode_t &autotuneMode);
            lwtensorAutotuneMode_t getAutotuneMode() const { return autotuneMode_; }
            
            lwtensorStatus_t setCacheMode(const lwtensorCacheMode_t &cacheMode);
            lwtensorCacheMode_t getCacheMode() const { return cacheMode_; }

            lwtensorStatus_t setIncrementalCount(const uint32_t incCount);
            uint32_t getIncrementalCount() const { return incrementalCount_; }

            lwtensorStatus_t setPartitionsK(const int32_t numPartitions);
            int32_t getPartitionsK() const { return numPartitionsK_; }

        protected:
            Contractiolwariant contractiolwariant_;
            lwtensorAutotuneMode_t autotuneMode_;
            lwtensorCacheMode_t cacheMode_;
            uint32_t incrementalCount_; /// determines how many different kernels will be measured/tested _before_ relying on the cached plan
            int32_t numPartitionsK_; /// number of partitions in the k-dimension (default: -1, using heuristic)
    };

    // Check that ContractionFind fits in lwtensorContractionFind_t
    static_assert(sizeof(ContractionFind) <= sizeof(lwtensorContractionFind_t),
                  "Size of ContractionFind greater than lwtensorContractionFind_t");

    class ContractionDescriptor;

    class ContractionDynamicParams {
        public:
            ContractionDynamicParams(
                 const ContractionDescriptor* desc, bool &swapAB);

            lwdaDataType_t typeA_;
            lwdaDataType_t typeB_;
            lwdaDataType_t typeC_;
            lwtensorComputeType_t typeCompute_;

            lwtensorOperator_t opA_;
            lwtensorOperator_t opB_;
            lwtensorOperator_t opC_;

            uint32_t alignmentReqA_;
            uint32_t alignmentReqB_;
            const uint32_t alignmentReqC_;
            const uint32_t alignmentReqD_;

            bool stridedLoadsA_;
            bool stridedLoadsB_;
            bool contiguousModeIsBatchedA_;
            bool contiguousModeIsBatchedB_;

            ExtentMap extent_;
            ModeList modeA_;
            ModeList modeB_;
            ModeList modeC_;
            StrideMap strideA_;
            StrideMap strideB_;
            StrideMap strideC_;

            ModeList modeM_;
            ModeList modeN_;
            ModeList modeK_;
            ModeList modeL_;

            /**
             * Returns a hash that (with a very high probablity) "uniquely" identifies the given tensor contraction;
             * this is mostly used to identify the contraction within the plan cache.
             */
            size_t getHash(const uint64_t workspaceSize, const bool swapAB, const lwtensorAutotuneMode_t autotuneMode, uint32_t tag) const;
    };

    class ColwolutionDescriptor : public Initializable<233>
    {
        public:
        static const uint32_t kMaxColwolvedModes= 4;
        lwtensorStatus_t init(
                const Context* ctx,
                const uint32_t numModesActivation, const int32_t modeActivation[],
                const uint32_t numModesFilter, const int32_t modeFilter[],
                const uint32_t numModesOutput, const int32_t modeOutput[],
                const uint32_t numColwolvedModes, const lwtensorColwolvedMode_t colwolvedModes[],
                const uint32_t numGroups,
                const lwtensorComputeType_t typeCompute,
                const lwtensorOperator_t opOut);

        static const int MAX_MODES = TensorDescriptor::LWTENSOR_MAX_MODES_EXTERNAL;

        /**
         * Finds the positin of the mode in the colwolvedModes_ array, return -1 otherwise.
         */
        int findColwolvedMode(mode_type mode) const;

        /**
         * Determines the extent of the output tensor
         */
        void getOutputExtent( const TensorDescriptor * descActivation, const
                TensorDescriptor * descFilter, int64_t extent[]) const;

        uint32_t numModesActivation_;
        int32_t modeActivation_[MAX_MODES];
        uint32_t numModesFilter_;
        int32_t modeFilter_[MAX_MODES];
        uint32_t numModesOutput_;
        int32_t modeOutput_[MAX_MODES];

        uint32_t numColwolvedModes_;
        lwtensorColwolvedMode_t colwolvedModes_[kMaxColwolvedModes];

        uint32_t numGroups_;

        lwtensorComputeType_t typeCompute_;
        lwtensorOperator_t opOut_;
    };

}  // namespace LWTENSOR_NAMESPACE
