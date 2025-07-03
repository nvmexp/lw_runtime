/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"
#include "string.h"

/**********************************************************************/

using namespace lwn;
using namespace lwn::dt;

// lwogtest doesn't like random spew to standard output.  We just eat any
// output unless LWN_DEBUG_LOG is set to 1.
#define LWN_DEBUG_LOG 0
static void log_output(const char *fmt, ...)
{
#if LWN_DEBUG_LOG
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
#endif
}
#define LOG log_output

/**********************************************************************/

// Bit values for comparison mismatches
#define LWN_DEPTH_STENCIL_COLOR_MISMATCH   1
#define LWN_DEPTH_STENCIL_DEPTH_MISMATCH   2
#define LWN_DEPTH_STENCIL_STENCIL_MISMATCH 4

// LWNDepthStencilPixel
//   A depth/stencil pixel storage class
template <LWNformat f, typename DepthType, LWNuint size, LWNuint depthOffset, LWNuint depthSize, LWNuint stencilOffset, LWNuint stencilSize>
struct LWNDepthStencilPixel {
    uint8_t data[size];
    uint8_t stencil() const {
        uint8_t ret = 0;
        switch (stencilSize) {
            case 1: ret = data[stencilOffset];
            default: break;
        }
        return ret;
    }
    DepthType depth() const {
        uint8_t ret[sizeof(DepthType)] = { 0 };
        for (uint8_t i = 0; i < depthSize; i++) {
            ret[i] = data[depthOffset + i];
        }
        return *reinterpret_cast<const DepthType *>(&ret[0]);
    }
    typedef DepthType LWNDepthStencilPixelDepthType;
    static const LWNformat format = f;
};
typedef LWNDepthStencilPixel<LWN_FORMAT_STENCIL8,          uint16_t, 1, 0, 0, 0, 1> LWNStencil8;
typedef LWNDepthStencilPixel<LWN_FORMAT_DEPTH16,           uint16_t, 2, 0, 2, 0, 0> LWNDepth16;
typedef LWNDepthStencilPixel<LWN_FORMAT_DEPTH24,           uint32_t, 4, 1, 3, 0, 0> LWNDepth24;
typedef LWNDepthStencilPixel<LWN_FORMAT_DEPTH32F,          float,    4, 0, 4, 0, 0> LWNDepth32F;
typedef LWNDepthStencilPixel<LWN_FORMAT_DEPTH24_STENCIL8,  uint32_t, 4, 1, 3, 0, 1> LWNDepth24Stencil8;
typedef LWNDepthStencilPixel<LWN_FORMAT_DEPTH32F_STENCIL8, float,    8, 0, 4, 4, 1> LWNDepth32FStencil8;

// Fisher-Yates array shuffle
template <typename T, LWNuint size>
static void shuffle(T *p) {
    T temp;
    for (LWNuint i = (size - 1); i >= 1; i--) {
        LWNuint j = lwIntRand(0, i);
        temp = p[j];
        p[j] = p[i];
        p[i] = temp;
    }
}

// Take an integer depth, and return an NCD formatted float
template <LWNuint bits, LWNdepthMode depthMode>
static LWNfloat depthNCD(LWNuint i) {
    i += ((i >> (bits - 1)) & 1);
    LWNfloat ret;
    switch (depthMode) {
        case LWN_DEPTH_MODE_NEAR_IS_MINUS_W: ret = (LWNint(i) - LWNint(1 << (bits - 1)))/LWNfloat(1 << (bits - 1)); break;
        case LWN_DEPTH_MODE_NEAR_IS_ZERO:    ret = i/LWNfloat(1 << bits); break;
        default: ret = 0.0f; break;
    }
    return ret;
}

// LWNDepthStencilTestState
//   Depth/stencil state class with depth/stencil test
template <typename LWNDepthStencilPixelType>
struct LWNDepthStencilTestState {
    typedef typename LWNDepthStencilPixelType::LWNDepthStencilPixelDepthType DepthType;

    LWNDepthStencilTestState() {
        bits.depthPass             = 1;
        bits.stencilPass           = 1;
        bits.backFacing            = 0;
        bits.depthEnable           = 0;
        bits.depthWriteEnable      = 0;
        bits.depthFunc             = DepthFunc::ALWAYS;
        bits.stencilEnable         = 0;
        bits.stencilValue          = 0x00;
        bits.stencilFunc           = StencilFunc::ALWAYS;
        bits.stencilOpFail         = StencilOp::KEEP;
        bits.stencilOpZPass        = StencilOp::KEEP;
        bits.stencilOpZFail        = StencilOp::KEEP;
        bits.stencilFrontRef       = 0x00;
        bits.stencilFrontValueMask = 0xff;
        bits.stencilFrontMask      = 0xff;
        bits.stencilBackFunc       = StencilFunc::ALWAYS;
        bits.stencilBackOpFail     = StencilOp::KEEP;
        bits.stencilBackOpZPass    = StencilOp::KEEP;
        bits.stencilBackOpZFail    = StencilOp::KEEP;
        bits.stencilBackRef        = 0x00;
        bits.stencilBackValueMask  = 0xff;
        bits.stencilBackMask       = 0xff;
        depth[0].uiDepth           = 0;
        depth[1].uiDepth           = 0;
    }

    struct {
        LWNuint depthPass             : 1;
        LWNuint stencilPass           : 1;
        LWNuint backFacing            : 1;
        LWNuint depthEnable           : 1;
        LWNuint depthWriteEnable      : 1;
        LWNuint depthFunc             : 4;
        LWNuint stencilEnable         : 1;
        LWNuint stencilValue          : 8;
        LWNuint stencilFunc           : 4;
        LWNuint stencilOpFail         : 4;
        LWNuint stencilOpZPass        : 4;
        LWNuint stencilOpZFail        : 4;
        LWNuint stencilFrontRef       : 8;
        LWNuint stencilFrontValueMask : 8;
        LWNuint stencilFrontMask      : 8;
        LWNuint stencilBackFunc       : 4;
        LWNuint stencilBackOpFail     : 4;
        LWNuint stencilBackOpZPass    : 4;
        LWNuint stencilBackOpZFail    : 4;
        LWNuint stencilBackRef        : 8;
        LWNuint stencilBackValueMask  : 8;
        LWNuint stencilBackMask       : 8;
    } bits;
    union {
        LWNuint   uiDepth;
        LWNfloat  fDepth;
        DepthType depth;
    } depth[2];

    LWNboolean  depthPass()             const { return LWNboolean(bits.depthPass);               }
    LWNboolean  stencilPass()           const { return LWNboolean(bits.stencilPass);             }
    LWNboolean  backFacing()            const { return LWNboolean(bits.backFacing);              }
    LWNboolean  depthEnable()           const { return LWNboolean(bits.depthEnable);             }
    LWNboolean  depthWriteEnable()      const { return LWNboolean(bits.depthWriteEnable);        }
    DepthFunc   depthFunc()             const { return DepthFunc::Enum(bits.depthFunc);          }
    LWNboolean  stencilEnable()         const { return LWNboolean(bits.stencilEnable);           }
    uint8_t    stencilValue()           const { return uint8_t(bits.stencilValue);              }
    StencilFunc stencilFrontFunc()      const { return StencilFunc::Enum(bits.stencilFunc);      }
    StencilOp   stencilFrontOpFail()    const { return StencilOp::Enum(bits.stencilOpFail);      }
    StencilOp   stencilFrontOpZFail()   const { return StencilOp::Enum(bits.stencilOpZFail);     }
    StencilOp   stencilFrontOpZPass()   const { return StencilOp::Enum(bits.stencilOpZPass);     }
    uint8_t    stencilFrontRef()        const { return uint8_t(bits.stencilFrontRef);           }
    uint8_t    stencilFrontValueMask()  const { return uint8_t(bits.stencilFrontValueMask);     }
    uint8_t    stencilFrontMask()       const { return uint8_t(bits.stencilFrontMask);          }
    StencilFunc stencilBackFunc()       const { return StencilFunc::Enum(bits.stencilBackFunc);  }
    StencilOp   stencilBackOpFail()     const { return StencilOp::Enum(bits.stencilBackOpFail);  }
    StencilOp   stencilBackOpZFail()    const { return StencilOp::Enum(bits.stencilBackOpZFail); }
    StencilOp   stencilBackOpZPass()    const { return StencilOp::Enum(bits.stencilBackOpZPass); }
    uint8_t    stencilBackRef()         const { return uint8_t(bits.stencilBackRef);            }
    uint8_t    stencilBackValueMask()   const { return uint8_t(bits.stencilBackValueMask);      }
    uint8_t    stencilBackMask()        const { return uint8_t(bits.stencilBackMask);           }
    StencilFunc stencilFunc()           const { return (backFacing()) ? stencilBackFunc()      : stencilFrontFunc();      }
    StencilOp   stencilOpFail()         const { return (backFacing()) ? stencilBackOpFail()    : stencilFrontOpFail();    }
    StencilOp   stencilOpZFail()        const { return (backFacing()) ? stencilBackOpZFail()   : stencilFrontOpZFail();   }
    StencilOp   stencilOpZPass()        const { return (backFacing()) ? stencilBackOpZPass()   : stencilFrontOpZPass();   }
    uint8_t    stencilRef()             const { return (backFacing()) ? stencilBackRef()       : stencilFrontRef();       }
    uint8_t    stencilValueMask()       const { return (backFacing()) ? stencilBackValueMask() : stencilFrontValueMask(); }
    uint8_t    stencilMask()            const { return (backFacing()) ? stencilBackMask()      : stencilFrontMask();      }
    StencilOp   stencilOp()             const { return (!stencilPass()) ? stencilOpFail() : (!depthPass()) ? stencilOpZFail() : stencilOpZPass(); }

    void depthPass(const LWNboolean &i)           { bits.depthPass                      = i; }
    void stencilPass(const LWNboolean &i)         { bits.stencilPass                    = i; }
    void backFacing(const LWNboolean &i)          { bits.backFacing                     = i; }
    void depthEnable(const LWNboolean &i)         { bits.depthEnable                    = i; }
    void depthWriteEnable(const LWNboolean &i)    { bits.depthWriteEnable               = i; }
    void depthFunc(const DepthFunc &i)            { bits.depthFunc                      = i; }
    void stencilEnable(const LWNboolean &i)       { bits.stencilEnable                  = i; }
    void stencilValue(const uint8_t &i)           { bits.stencilValue                   = i; }
    void stencilFrontFunc(const StencilFunc &i)   { bits.stencilFunc                    = i; }
    void stencilFrontOpFail(const StencilOp &i)   { bits.stencilOpFail                  = i; }
    void stencilFrontOpZFail(const StencilOp &i)  { bits.stencilOpZFail                 = i; }
    void stencilFrontOpZPass(const StencilOp &i)  { bits.stencilOpZPass                 = i; }
    void stencilFrontRef(const uint8_t &i)        { bits.stencilFrontRef                = i; }
    void stencilFrontValueMask(const uint8_t &i)  { bits.stencilFrontValueMask          = i; }
    void stencilFrontMask(const uint8_t &i)       { bits.stencilFrontMask               = i; }
    void stencilBackFunc(const StencilFunc &i)    { bits.stencilBackFunc                = i; }
    void stencilBackOpFail(const StencilOp &i)    { bits.stencilBackOpFail              = i; }
    void stencilBackOpZFail(const StencilOp &i)   { bits.stencilBackOpZFail             = i; }
    void stencilBackOpZPass(const StencilOp &i)   { bits.stencilBackOpZPass             = i; }
    void stencilBackRef(const uint8_t &i)         { bits.stencilBackRef                 = i; }
    void stencilBackValueMask(const uint8_t &i)   { bits.stencilBackValueMask           = i; }
    void stencilBackMask(const uint8_t &i)        { bits.stencilBackMask                = i; }
    void stencilFunc(const StencilFunc &i)        { if (backFacing()) { stencilBackFunc(i);      } else { stencilFrontFunc(i);      } }
    void stencilOpFail(const StencilOp &i)        { if (backFacing()) { stencilBackOpFail(i);    } else { stencilFrontOpFail(i);    } }
    void stencilOpZFail(const StencilOp &i)       { if (backFacing()) { stencilBackOpZFail(i);   } else { stencilFrontOpZFail(i);   } }
    void stencilOpZPass(const StencilOp &i)       { if (backFacing()) { stencilBackOpZPass(i);   } else { stencilFrontOpZPass(i);   } }
    void stencilRef(const uint8_t &i)             { if (backFacing()) { stencilBackRef(i);       } else { stencilFrontRef(i);       } }
    void stencilValueMask(const uint8_t &i)       { if (backFacing()) { stencilBackValueMask(i); } else { stencilFrontValueMask(i); } }
    void stencilMask(const uint8_t &i)            { if (backFacing()) { stencilBackMask(i);      } else { stencilFrontMask(i);      } }
    void stencilOp(const StencilOp &i) {
        if      (!stencilPass()) { stencilOpFail(i);  }
        else if (!depthPass()  ) { stencilOpZFail(i); }
        else                     { stencilOpZPass(i); }
    }

    LWNboolean stencilTest() const {
        LWNboolean ret = true;
        if (stencilEnable()) {
            const uint8_t svm = stencilValueMask();
            const uint8_t a = (stencilRef()   & svm);
            const uint8_t b = (stencilValue() & svm);
            switch (stencilFunc()) {
                default:
                case StencilFunc::NEVER:    ret = false;    break;
                case StencilFunc::LESS:     ret = (a <  b); break;
                case StencilFunc::EQUAL:    ret = (a == b); break;
                case StencilFunc::LEQUAL:   ret = (a <= b); break;
                case StencilFunc::GREATER:  ret = (a >  b); break;
                case StencilFunc::NOTEQUAL: ret = (a != b); break;
                case StencilFunc::GEQUAL:   ret = (a >= b); break;
                case StencilFunc::ALWAYS:   ret = true;     break;
            }
        }
        return ret;
    }
    LWNboolean depthTest() const {
        LWNboolean ret = true;
        if (depthEnable()) {
            switch (depthFunc()) {
                default:
                case DepthFunc::NEVER:    ret = false;                              break;
                case DepthFunc::LESS:     ret = (depth[1].depth <  depth[0].depth); break;
                case DepthFunc::EQUAL:    ret = (depth[1].depth == depth[0].depth); break;
                case DepthFunc::LEQUAL:   ret = (depth[1].depth <= depth[0].depth); break;
                case DepthFunc::GREATER:  ret = (depth[1].depth >  depth[0].depth); break;
                case DepthFunc::NOTEQUAL: ret = (depth[1].depth != depth[0].depth); break;
                case DepthFunc::GEQUAL:   ret = (depth[1].depth >= depth[0].depth); break;
                case DepthFunc::ALWAYS:   ret = true;                               break;
            }
        }
        return ret;
    }

    LWNuint getColor() const {
        return (stencilPass() && depthPass()) ? (backFacing()) ? 0xffff0000 : 0xff00ff00 : 0xff000000;
    }
    DepthType getDepth() const {
        return depth[(depthEnable() && stencilPass() && depthPass() && depthWriteEnable())].depth;
    }
    LWNfloat getFloatDepth(const LWNuint &i, const LWNdepthMode &depthMode) const {
        LWNfloat ret;
        switch (LWNDepthStencilPixelType::format) {
            case LWN_FORMAT_DEPTH16:
                ret = (i == 0 || depthMode == LWN_DEPTH_MODE_NEAR_IS_ZERO) ?
                    depthNCD<16, LWN_DEPTH_MODE_NEAR_IS_ZERO   >(depth[i].uiDepth) :
                    depthNCD<16, LWN_DEPTH_MODE_NEAR_IS_MINUS_W>(depth[i].uiDepth);
                break;
            case LWN_FORMAT_DEPTH24:
            case LWN_FORMAT_DEPTH24_STENCIL8:
                ret = (i == 0 || depthMode == LWN_DEPTH_MODE_NEAR_IS_ZERO) ?
                    depthNCD<24, LWN_DEPTH_MODE_NEAR_IS_ZERO   >(depth[i].uiDepth) :
                    depthNCD<24, LWN_DEPTH_MODE_NEAR_IS_MINUS_W>(depth[i].uiDepth);
                break;
            case LWN_FORMAT_DEPTH32F:
            case LWN_FORMAT_DEPTH32F_STENCIL8:
                 ret = (i == 0 || depthMode == LWN_DEPTH_MODE_NEAR_IS_ZERO) ?
                    depth[i].fDepth :
                    (2.0f * depth[i].fDepth) - 1.0f;
                break;
            default: ret = 0.0f; break;
        }
        return ret;
    }
    uint8_t getStencil() const {
        uint8_t ret = stencilValue();
        const StencilOp sOp = stencilOp();
        if (stencilEnable() && sOp != StencilOp::KEEP) {
            const uint8_t sm = stencilMask();
            const uint8_t x = (ret & (~sm));
            switch (sOp) {
                default:
                case StencilOp::ZERO:      ret = 0;              break;
                case StencilOp::REPLACE:   ret = stencilRef();   break;
                case StencilOp::INCR: if (ret < 0xff) { ret++; } break;
                case StencilOp::DECR: if (ret > 0x00) { ret--; } break;
                case StencilOp::ILWERT:    ret = ~ret;           break;
                case StencilOp::INCR_WRAP: ret++;                break;
                case StencilOp::DECR_WRAP: ret--;                break;
            }
            ret = ((ret & sm) | x);
        }
        return ret;
    }

    // Randomize values to satisfy current pass and function state
    bool randomize() {

        // Randomize stencil values
        switch (LWNDepthStencilPixelType::format) {
            case LWN_FORMAT_STENCIL8:
            case LWN_FORMAT_DEPTH24_STENCIL8:
            case LWN_FORMAT_DEPTH32F_STENCIL8: {
                // Choose a random initial stencil values
                stencilValue(uint8_t(lwBitRand(8)));
                stencilFrontRef(uint8_t(lwBitRand(8)));
                stencilFrontValueMask(uint8_t(lwBitRand(8)));
                stencilFrontMask(uint8_t(lwBitRand(8)));
                stencilBackRef(uint8_t(lwBitRand(8)));
                stencilBackValueMask(uint8_t(lwBitRand(8)));
                stencilBackMask(uint8_t(lwBitRand(8)));

                // If stencil test is enabled, validate the pass state
                if (stencilEnable()) {
                    // Start with equal reference
                    stencilRef(stencilValue());

                    // Choose a random stencil result state unless the result must be equal to the initial stencil value
                    // to trigger the stencil pass condition.
                    switch (stencilFunc()) {
                        case StencilFunc::LEQUAL:   // For LEQUAL and GEQUAL, use lwBitRand(1) to a flip a coin, where 0 is the EQUAL case
                        case StencilFunc::GEQUAL:   if (!stencilPass() || lwBitRand(1)) { goto LwnDepthStencilTest_randomize_stencil; } break;
                        case StencilFunc::EQUAL:    if (!stencilPass())                 { goto LwnDepthStencilTest_randomize_stencil; } break;
                        case StencilFunc::NOTEQUAL: if ( stencilPass())                 { goto LwnDepthStencilTest_randomize_stencil; } break;
                        default: {
LwnDepthStencilTest_randomize_stencil:
                            LWNuint count = 0;
                            do {
                                stencilValue(uint8_t(lwBitRand(8)));
                                stencilRef(uint8_t(lwBitRand(8)));
                                stencilValueMask(uint8_t(lwBitRand(8)));
                                stencilMask(uint8_t(lwBitRand(8)));
                                count++;
                            } while (stencilTest() != stencilPass() && count < 1000);
                        } break;
                    }
                }

                // We absolutely have to guarantee this condition at this point.
                if (stencilTest() != stencilPass()) {
                    LOG("stencilTest is not equal to stencilPass.\n");
                    return false;
                }
            } break;
            default: break;
        }

        // Randomize depth values
        switch (LWNDepthStencilPixelType::format) {
            case LWN_FORMAT_DEPTH16:
            case LWN_FORMAT_DEPTH24:
            case LWN_FORMAT_DEPTH32F:
            case LWN_FORMAT_DEPTH24_STENCIL8:
            case LWN_FORMAT_DEPTH32F_STENCIL8: {
                // Choose a random initial depth
                switch (LWNDepthStencilPixelType::format) {
                    case LWN_FORMAT_DEPTH16:           depth[0].uiDepth = depth[1].uiDepth = lwBitRand(16);           break;
                    case LWN_FORMAT_DEPTH24:
                    case LWN_FORMAT_DEPTH24_STENCIL8:  depth[0].uiDepth = depth[1].uiDepth = lwBitRand(24);           break;
                    case LWN_FORMAT_DEPTH32F:
                    case LWN_FORMAT_DEPTH32F_STENCIL8: depth[0].fDepth  = depth[1].fDepth  = lwFloatRand(0.0f, 1.0f); break;
                    default: break;
                }

                // If the depth test is enabled, validate the pass state
                if (depthEnable()) {

                    // Choose a random result depth unless the result depth must be equal to the initial depth
                    // to trigger the depth pass condition.
                    switch (depthFunc()) {
                        case DepthFunc::LEQUAL:   // For LEQUAL and GEQUAL, use lwBitRand(1) to a flip a coin, where 0 is the EQUAL case
                        case DepthFunc::GEQUAL:   if (!depthPass() || lwBitRand(1)) { goto LwnDepthStencilTest_randomize_depth; } break;
                        case DepthFunc::EQUAL:    if (!depthPass())                 { goto LwnDepthStencilTest_randomize_depth; } break;
                        case DepthFunc::NOTEQUAL: if ( depthPass())                 { goto LwnDepthStencilTest_randomize_depth; } break;
                        default: {
LwnDepthStencilTest_randomize_depth:
                            LWNuint count = 0;
                            do { // Randomize the result depth (must be not equal to initial depth)
                                switch (LWNDepthStencilPixelType::format) {
                                    case LWN_FORMAT_DEPTH16:           depth[1].uiDepth = lwBitRand(16);           break;
                                    case LWN_FORMAT_DEPTH24:
                                    case LWN_FORMAT_DEPTH24_STENCIL8:  depth[1].uiDepth = lwBitRand(24);           break;
                                    case LWN_FORMAT_DEPTH32F:
                                    case LWN_FORMAT_DEPTH32F_STENCIL8: depth[1].fDepth  = lwFloatRand(0.0f, 1.0f); break;
                                    default: break;
                                }
                                count++;
                            } while (depth[0].depth == depth[1].depth && count < 1000);

                            // If the values don't work for the depth test, swap them
                            if (depthTest() != depthPass()) {
                                DepthType temp = depth[0].depth;
                                depth[0].depth = depth[1].depth;
                                depth[1].depth = temp;
                            }
                        } break;
                    }
                }

                // We absolutely have to guarantee this condition at this point.
                if (depthTest() != depthPass()) {
                    LOG("depthTest is not equal to depthPass.\n");
                    return false;
                }
            } break;
            default: break;
        }

        return true;
    }

    LWNuint compare(const LWNuint &cResult, const LWNDepthStencilPixelType &dsResult) const {
        LWNuint ret = 0;
        if (getColor() != cResult) {
            ret |= LWN_DEPTH_STENCIL_COLOR_MISMATCH;
        }
        switch (LWNDepthStencilPixelType::format) {
            case LWN_FORMAT_DEPTH16:
            case LWN_FORMAT_DEPTH24:
            case LWN_FORMAT_DEPTH24_STENCIL8:
            case LWN_FORMAT_DEPTH32F:
            case LWN_FORMAT_DEPTH32F_STENCIL8: {
                if (getDepth() != dsResult.depth()) {
                    ret |= LWN_DEPTH_STENCIL_DEPTH_MISMATCH;
                }
            } break;
            default: break;
        }
        switch (LWNDepthStencilPixelType::format) {
            case LWN_FORMAT_STENCIL8:
            case LWN_FORMAT_DEPTH24_STENCIL8:
            case LWN_FORMAT_DEPTH32F_STENCIL8: {
                if (getStencil() != dsResult.stencil()) {
                    ret |= LWN_DEPTH_STENCIL_STENCIL_MISMATCH;
                }
            } break;
            default: break;
        }
        return ret;
    }

    void print(const LWNdepthMode &depthMode) const {
        const char *funcs[] = { "NEVER", "LESS", "EQUAL", "LEQUAL", "GREATER", "NOTEQUAL", "GEQUAL", "ALWAYS" };
        const char *ops[]  = { "KEEP", "ZERO", "REPLACE", "INCR", "DECR", "ILWERT", "INCR_WRAP", "DECR_WRAP" };
        LOG("  depthPass             %d -> %d\n", depthPass(), depthTest());
        LOG("  stencilPass           %d -> %d\n", stencilPass(), stencilTest());
        LOG("  backFacing            %d\n",       backFacing());
        LOG("  depthEnable           %d\n",       depthEnable());
        LOG("  depthWriteEnable      %d\n",       depthWriteEnable());
        LOG("  depthFunc             LWN_DEPTH_FUNC_%s(0x%01x)\n", funcs[depthFunc()], LWNint(depthFunc()));
        LOG("  stencilEnable         %d\n",       stencilEnable());
        LOG("  stencilFrontFunc      LWN_STENCIL_FUNC_%s(0x%01x)\n", funcs[stencilFrontFunc()],  LWNint(stencilFrontFunc()));
        LOG("  stencilFrontOpFail    LWN_STENCIL_OP_%s(0x%01x)\n",   ops[stencilFrontOpFail()],  LWNint(stencilFrontOpFail()));
        LOG("  stencilFrontOpZPass   LWN_STENCIL_OP_%s(0x%01x)\n",   ops[stencilFrontOpZPass()], LWNint(stencilFrontOpZPass()));
        LOG("  stencilFrontOpZFail   LWN_STENCIL_OP_%s(0x%01x)\n",   ops[stencilFrontOpZFail()], LWNint(stencilFrontOpZFail()));
        LOG("  stencilFrontRef       0x%02x\n",   stencilFrontRef());
        LOG("  stencilFrontValueMask 0x%02x\n",   stencilFrontValueMask());
        LOG("  stencilFrontMask      0x%02x\n",   stencilFrontMask());
        LOG("  stencilBackFunc       LWN_STENCIL_FUNC_%s(0x%01x)\n", funcs[stencilBackFunc()],  LWNint(stencilBackFunc()));
        LOG("  stencilBackOpFail     LWN_STENCIL_OP_%s(0x%01x)\n",   ops[stencilBackOpFail()],  LWNint(stencilBackOpFail()));
        LOG("  stencilBackOpZPass    LWN_STENCIL_OP_%s(0x%01x)\n",   ops[stencilBackOpZPass()], LWNint(stencilBackOpZPass()));
        LOG("  stencilBackOpZFail    LWN_STENCIL_OP_%s(0x%01x)\n",   ops[stencilBackOpZFail()], LWNint(stencilBackOpZFail()));
        LOG("  stencilBackRef        0x%02x\n",   stencilBackRef());
        LOG("  stencilBackValueMask  0x%02x\n",   stencilBackValueMask());
        LOG("  stencilBackMask       0x%02x\n",   stencilBackMask());
        LOG("  stencilValue          0x%02x\n",   stencilValue());
        LOG("  stencilResult         0x%02x\n",   getStencil());
        for (uint8_t i = 0; i < 2; i++) {
        LOG("  depth[%d]              ", i);
            switch (LWNDepthStencilPixelType::format) {
                case LWN_FORMAT_DEPTH16:           LOG("0x%04x", depth[i].uiDepth); break;
                case LWN_FORMAT_DEPTH24:
                case LWN_FORMAT_DEPTH24_STENCIL8:  LOG("0x%06x", depth[i].uiDepth); break;
                default:
                case LWN_FORMAT_DEPTH32F:
                case LWN_FORMAT_DEPTH32F_STENCIL8: LOG("%f", depth[i].fDepth); break;
            }
            LOG(" - %f\n", getFloatDepth(i, depthMode));
        }
        LOG("\n");
    }
};


// The code for generating the random test data assumes that the enumerations
// are 0-indexed. This assumption is no longer valid as of LWN 48.0, so we use
// the following functions to colwert to/from the old 0-indexed values.
static DepthFunc::Enum idxToDepthFunc(LWNuint i)
{
    return DepthFunc::Enum(i+1);
}

static StencilFunc::Enum idxToStencilFunc(LWNuint i)
{
    return StencilFunc::Enum(i+1);
}

static StencilOp::Enum idxToStencilOp(LWNuint i)
{
    return StencilOp::Enum(i+1);
}

static LWNuint DepthFuncToIdx(DepthFunc::Enum e)
{
    return ((LWNuint)e)-1;
}

static LWNuint StencilFuncToIdx(StencilFunc::Enum e)
{
    return ((LWNuint)e)-1;
}

// Randomization buckets
template <typename LWNDepthStencilPixelType>
class LWNDepthStencilRandomBuckets {
public:
    LWNDepthStencilRandomBuckets(const LWNuint numCells) {
        for (LWNuint i = 0; i < 16; i++) {
            idx16[i] = i;
        }
        switch (LWNDepthStencilPixelType::format) {
            case LWN_FORMAT_DEPTH16:
            case LWN_FORMAT_DEPTH24:
            case LWN_FORMAT_DEPTH32F:
            case LWN_FORMAT_DEPTH24_STENCIL8:
            case LWN_FORMAT_DEPTH32F_STENCIL8: {
                depthEnable[0]      = (numCells/10);                          // 10% of all tests
                depthEnable[1]      = (numCells - depthEnable[0]);            // 90% of all tests
                depthWriteEnable[0] = (depthEnable[1]/10);                    // 10% of depthEnable
                depthWriteEnable[1] = (depthEnable[1] - depthWriteEnable[0]); // 90% of depthEnable
                LWNuint x = depthEnable[1]%14;
                LWNuint y = depthEnable[1]/14;
                shuffle<LWNuint, 16>(&idx16[0]);
                // Spread depth function and result evenly across pass/fail for all (valid) depth functions
                for (LWNuint i = 0; i < 16; i++) {
                    if (idx16[i] == (8 | DepthFuncToIdx(DepthFunc::NEVER)) ||  // DepthFunc::NEVER  can never pass
                        idx16[i] == (0 | DepthFuncToIdx(DepthFunc::ALWAYS))) { // DepthFunc::ALWAYS can never fail
                        depthFunc[idx16[i]] = 0;
                    }
                    else {
                        depthFunc[idx16[i]] = y;
                        if (x) {
                            x--;
                            depthFunc[idx16[i]]++;
                        }
                    }
                }
            } break;
            default: {
                depthEnable[0]      = 0;
                depthEnable[1]      = 0;
                depthWriteEnable[0] = 0;
                depthWriteEnable[1] = 0;
                for (LWNuint i = 0; i < 16; i++) {
                    depthFunc[i] = 0;
                }
            } break;
        }
        switch (LWNDepthStencilPixelType::format) {
            case LWN_FORMAT_STENCIL8:
            case LWN_FORMAT_DEPTH24_STENCIL8:
            case LWN_FORMAT_DEPTH32F_STENCIL8: {
                stencilEnable[0] = (numCells/10);                                      // 10% of all tests
                stencilEnable[1] = (numCells - stencilEnable[0])/2;                    // 45% of all tests
                stencilEnable[2] = (numCells - (stencilEnable[0] + stencilEnable[1])); // 45% of all tests
                LWNuint x = (numCells - stencilEnable[0]);
                LWNuint y = x/14;
                x %= 14;
                shuffle<LWNuint, 16>(&idx16[0]);
                // Spread stencil function and result evenly across pass/fail for all (valid) stencil functions
                for (LWNuint i = 0; i < 16; i++) {
                    if (idx16[i] == (8 | StencilFuncToIdx(StencilFunc::NEVER)) ||  // StencilFunc::NEVER  can never pass
                        idx16[i] == (0 | StencilFuncToIdx(StencilFunc::ALWAYS))) { // StencilFunc::ALWAYS can never fail
                        stencilFunc[idx16[i]] = 0;
                    }
                    else {
                        stencilFunc[idx16[i]] = y;
                        if (x) {
                            x--;
                            stencilFunc[idx16[i]]++;
                        }
                    }
                }
                x = (numCells - stencilEnable[0]);
                y = 6*(x/8);
                x %= 8;
                for (LWNuint i = 0; i < 8; i++) {
                    idx8[i] = i;
                }
                shuffle<LWNuint, 8>(&idx8[0]);
                for (LWNuint i = 0; i < 8; i++) {
                    stencilOp[idx8[i]] = y;
                    if (x) {
                        x--;
                        stencilOp[idx8[i]] += 6;
                    }
                }
            } break;
            default: {
                stencilEnable[0] = 0;
                stencilEnable[1] = 0;
                stencilEnable[2] = 0;
                for (LWNuint i = 0; i < 16; i++) {
                    stencilFunc[i] = 0;
                }
                for (LWNuint i = 0; i < 8; i++) {
                    idx8[i] = 0;
                    stencilOp[i] = 0;
                }
            } break;
        }
    }

        // Randomize using LWNDepthStencilRandomBuckets
    bool randomize(LWNDepthStencilTestState<LWNDepthStencilPixelType> &dsState) {

        // Randomize stencil enable, backfacing, and stencil function
        switch (LWNDepthStencilPixelType::format) {
            case LWN_FORMAT_STENCIL8:
            case LWN_FORMAT_DEPTH24_STENCIL8:
            case LWN_FORMAT_DEPTH32F_STENCIL8: {
                if (!pickStencilEnable(dsState)) {
                    LOG("Could not pick a random stencil enable state.\n");
                    return false;
                }
                if (dsState.stencilEnable() && !pickStencilFunc(dsState)) {
                    LOG("Could not pick a random stencil function.\n");
                    return false;
                }
                switch (dsState.stencilFunc()) {
                    case StencilFunc::NEVER: {
                        if (dsState.stencilPass()) {
                            LOG("Picked StencilFunc::NEVER with stencil pass.\n");
                            return false;
                        }
                    } break;
                    case StencilFunc::ALWAYS: {
                        if (!dsState.stencilPass()) {
                            LOG("Picked StencilFunc::ALWAYS with stencil fail.\n");
                            return false;
                        }
                    } break;
                    default: break;
                }
            } break;
            default: {
                // Without a stencil, backfacing can just be random
                dsState.backFacing(LWNboolean(lwBitRand(1)));
            } break;
        }

        // Randomize depth enable and depth function
        switch (LWNDepthStencilPixelType::format) {
            case LWN_FORMAT_DEPTH16:
            case LWN_FORMAT_DEPTH24:
            case LWN_FORMAT_DEPTH32F:
            case LWN_FORMAT_DEPTH24_STENCIL8:
            case LWN_FORMAT_DEPTH32F_STENCIL8: {
                if (!pickDepthEnable(dsState)) {
                    LOG("Could not pick a random depth enable state.\n");
                    return false;
                }
                if (dsState.depthEnable() && !pickDepthFunc(dsState)) {
                    LOG("Could not pick a random depth function.\n");
                    return false;
                }
                switch (dsState.depthFunc()) {
                    case DepthFunc::NEVER: {
                        if (dsState.depthPass()) {
                            LOG("Picked DepthFunc::NEVER with depth pass.\n");
                            return false;
                        }
                    } break;
                    case DepthFunc::ALWAYS: {
                        if (!dsState.depthPass()) {
                            LOG("Picked DepthFunc::ALWAYS with depth fail.\n");
                            return false;
                        }
                    } break;
                    default: break;
                }
            } break;
            default: break;
        }

        // Randomize stencil op state
        switch (LWNDepthStencilPixelType::format) {
            case LWN_FORMAT_STENCIL8:
            case LWN_FORMAT_DEPTH24_STENCIL8:
            case LWN_FORMAT_DEPTH32F_STENCIL8: {
                if (dsState.stencilEnable() && !pickStencilOp(dsState)) {
                    LOG("Could not pick a random stencil op.\n");
                    return false;
                }
            } break;
            default: break;
        }

        return dsState.randomize();
    }

private:

    bool pickDepthEnable(LWNDepthStencilTestState<LWNDepthStencilPixelType> &dsState) {
        if (depthEnable[0] || (depthEnable[1] && (depthWriteEnable[0] || depthWriteEnable[1]))) {
            dsState.depthEnable((depthEnable[1] && (!depthEnable[0] || lwBitRand(1))) ? LWN_TRUE : LWN_FALSE);
            if (dsState.depthEnable()) {
                depthEnable[1]--;
                dsState.depthWriteEnable((depthWriteEnable[1] && (!depthWriteEnable[0] || lwBitRand(1))) ? LWN_TRUE : LWN_FALSE);
                depthWriteEnable[dsState.depthWriteEnable()]--;
            }
            else {
                depthEnable[0]--;
                dsState.depthWriteEnable(LWN_FALSE);
            }
            return true;
        }
        return false;
    }

    bool pickStencilEnable(LWNDepthStencilTestState<LWNDepthStencilPixelType> &dsState) {
        if (stencilEnable[0] || stencilEnable[1] || stencilEnable[2]) {
            LWNuint stencilEn;
            do {
                stencilEn = lwIntRand(0, 2);
            } while (!stencilEnable[stencilEn]);
            stencilEnable[stencilEn]--;
            dsState.stencilEnable((stencilEn >= 1) ? LWN_TRUE : LWN_FALSE);
            dsState.backFacing((stencilEn == 2 || (stencilEn == 0 && lwBitRand(1))) ? LWN_TRUE : LWN_FALSE);
            return true;
        }
        return false;
    }

    bool pickDepthFunc(LWNDepthStencilTestState<LWNDepthStencilPixelType> &dsState) {
        shuffle<LWNuint, 16>(&idx16[0]);
        for (LWNuint i = 0; i < 16; i++) {
            if (depthFunc[idx16[i]]) {
                depthFunc[idx16[i]]--;
                dsState.depthPass((idx16[i] & 8) ? LWN_TRUE : LWN_FALSE);
                dsState.depthFunc(idxToDepthFunc(idx16[i] & 7));
                return true;
            }
        }
        return false;
    }

    bool pickStencilFunc(LWNDepthStencilTestState<LWNDepthStencilPixelType> &dsState) {
        shuffle<LWNuint, 16>(&idx16[0]);
        for (LWNuint i = 0; i < 16; i++) {
            if (stencilFunc[idx16[i]]) {
                stencilFunc[idx16[i]]--;
                dsState.stencilPass((idx16[i] & 8) ? LWN_TRUE : LWN_FALSE);
                dsState.stencilFunc(idxToStencilFunc(idx16[i] & 7));
                if (dsState.backFacing()) {
                    dsState.stencilFrontFunc(idxToStencilFunc(lwBitRand(3)));
                }
                else {
                    dsState.stencilBackFunc(idxToStencilFunc(lwBitRand(3)));
                }
                return true;
            }
        }
        return false;
    }

    bool pickStencilOp(LWNDepthStencilTestState<LWNDepthStencilPixelType> &dsState) {
        for (LWNuint i = 0; i < 6; i++) {
            shuffle<LWNuint, 8>(&idx8[0]);
            LWNuint j = 0;
            for (; j < 8; j++) {
                if (stencilOp[idx8[j]]) {
                    stencilOp[idx8[j]]--;
                    break;
                }
            }
            if (j >= 8) {
                break;
            }
            switch (i) {
                case 0: dsState.stencilFrontOpFail (idxToStencilOp(idx8[j])); break;
                case 1: dsState.stencilFrontOpZFail(idxToStencilOp(idx8[j])); break;
                case 2: dsState.stencilFrontOpZPass(idxToStencilOp(idx8[j])); break;
                case 3: dsState.stencilBackOpFail  (idxToStencilOp(idx8[j])); break;
                case 4: dsState.stencilBackOpZFail (idxToStencilOp(idx8[j])); break;
                case 5: dsState.stencilBackOpZPass (idxToStencilOp(idx8[j])); return true; break;
            }
        }
        return false;
    }

    LWNuint idx16[16];
    LWNuint idx8[8];
    LWNuint depthEnable[2];
    LWNuint depthWriteEnable[2];
    LWNuint depthFunc[16];
    LWNuint stencilEnable[3];
    LWNuint stencilFunc[16];
    LWNuint stencilOp[8];
};

// DepthStencil test class
template <typename LWNDepthStencilPixelType>
class LwnDepthStencilTest
{
    struct Vertex {
        vec3 pos;
    };
    static const LWNuint numCols = 256, numRows = 256;
    static const LWNuint numCells = numRows * numCols;
    static const LWNuint numPrims = numCells;
    static const LWNuint numVertices = 3 * numPrims;
    static const LWNuint sizeVBO = numVertices * sizeof(Vertex);
    static const LWNuint sizeDS = sizeof(LWNDepthStencilPixelType);

    typedef LWNDepthStencilTestState<LWNDepthStencilPixelType> LWNDepthStencilState;
    typedef LWNDepthStencilRandomBuckets<LWNDepthStencilPixelType> LWNDepthStencilBuckets;

    bool compileShader(Device *device, lwnTest::GLSLCHelper *glslcHelper, Program* &pgm, VertexShader &vs, FragmentShader &fs) const {
        char *vsString = (char *)__LWOG_MALLOC((vs.source().length() + 1)*sizeof(char));
        if (!vsString) {
            return false;
        }
        char *fsString = (char *)__LWOG_MALLOC((fs.source().length() + 1)*sizeof(char));
        if (!fsString) {
            __LWOG_FREE(vsString);
            return false;
        }

        strcpy(vsString, vs.source().c_str());
        strcpy(fsString, fs.source().c_str());

        const char *shaders[2] = {
            vsString,
            fsString
        };

        bool ret = true;
        pgm = device->CreateProgram();
        if (!glslcHelper->CompileAndSetShaders(pgm, vs, fs)) {
            LOG("Shader compile error.\lwertex Source:\n%s\n\nFragment Source:\n%s\n\nInfoLog:\n%s\n", shaders[0], shaders[1], glslcHelper->GetInfoLog());
            ret = false;
            pgm->Free();
        }

        __LWOG_FREE(vsString);
        __LWOG_FREE(fsString);

        return ret;
    }

    bool run(
        const LWNdepthMode &depthMode,
        LWNuint *cResult,
        LWNDepthStencilPixelType *dsResult,
        LWNDepthStencilState *dsState) const;
    bool verify(const LWNdepthMode &depthMode) const;

public:
    OGTEST_CppMethods();
};

template <typename LWNDepthStencilPixelType>
int LwnDepthStencilTest<LWNDepthStencilPixelType>::isSupported()
{
    return lwogCheckLWNAPIVersion(26, 1);
}

template <>
int LwnDepthStencilTest<LWNStencil8>::isSupported()
{
    return lwogCheckLWNAPIVersion(26, 1) && g_lwnDeviceCaps.supportsStencil8;
}

template <typename LWNDepthStencilPixelType>
lwString LwnDepthStencilTest<LWNDepthStencilPixelType>::getDescription()
{
    return
        "For the depth modes:\n"
        "  1. LWN_DEPTH_MODE_NEAR_IS_MINUS_W\n"
        "  2. LWN_DEPTH_MODE_NEAR_IS_ZERO\n"
        "\n"
        "The following test is performed:\n"
        "  1. Creates a 256x256 color/depth/stencil framebuffer\n"
        "  2. For each pixel in the framebuffer, a random depth/stencil state is chosen:\n"
        "    -# When the selected format has a depth, the following are true:\n"
        "      -# 10% =  6553 pixels - depth test is disabled\n"
        "      -# 90% = 58983 pixels - depth test is enabled\n"
        "      -# 10% =  5898 pixels - depth test is enabled, but depth write is disabled\n"
        "      -# 90% = 53085 pixels - depth test is enabled and depth write is enabled\n"
        "      -# Each possible depth function, and whether or not it passes or fails is equally\n"
        "         distributed over all 58983 depth test enabled pixels, for 58983/14 = 4213 pixels\n"
        "    -# When the selected format has a stencil value, the following are true:\n"
        "      -# 10% =  6553 pixels - stencil test is disabled\n"
        "      -# 45% = 29491 pixels - stencil test is enabled with front-facing triangles\n"
        "      -# 45% = 29492 pixels - stencil test is enabled with back-facing triangles\n"
        "      -# Each possible stencil function, and whether or not it passes or fails is equally\n"
        "         distributed over all 58983 stencil test enabled pixels, for 58983/14 = 4213 pixels\n"
        "  3. A random initial depth/stencil value is chosen, to which the pixel under test is cleared\n"
        "  4. Given the random depth/stencil state, a final depth/stencil value is chosen to pass or fail\n"
        "     from distributions described above.  A triangle is drawn at this depth for testing.\n"
        "  5. After all pixel's have been drawn with random depth/stencil state, the color/depth/stencil\n"
        "     buffers are copied back and compared with a buffer of results from a software depth/stencil\n"
        "     reference.\n"
        "  6. If the results from the GPU match the software reference results, clear the screen with green.\n"
        "  7. Otherwise, clear the screen with red.\n";
}

template <typename LWNDepthStencilPixelType>
bool LwnDepthStencilTest<LWNDepthStencilPixelType>::run(
    const LWNdepthMode &depthMode,
    LWNuint *cResult,
    LWNDepthStencilPixelType *dsResult,
    LWNDepthStencilState *dsState) const
{
    bool ret = false;
    Vertex *vertexData = NULL;
    Program *pgm;

    DeviceState *testDevice =
        new DeviceState(LWNdeviceFlagBits(0), LWN_WINDOW_ORIGIN_MODE_LOWER_LEFT,
                                 depthMode);
    if (!testDevice || !testDevice->isValid()) {
        delete testDevice;
        DeviceState::SetDefaultActive();
        return false;
    }

    testDevice->SetActive();
    Device *device = testDevice->getDevice();
    Queue *queue = testDevice->getQueue();
    QueueCommandBuffer &cmd = testDevice->getQueueCB();
    lwnTest::GLSLCHelper *glslcHelper = testDevice->getGLSLCHelper();

    // Shaders and Program
    VertexShader vs(440);
    vs <<
        "layout(location = 0) in vec3 position;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0f);\n"
        "}\n";
    FragmentShader fs(440);
    fs <<
        "layout(location = 0) out vec4 color;\n"
        "void main() {\n"
        "  color = (gl_FrontFacing) ? vec4(0.0f, 1.0f, 0.0f, 1.0f) : vec4(0.0f, 0.0f, 1.0f, 1.0f);\n"
        "}\n";
    if (!compileShader(device, glslcHelper, pgm, vs, fs)) {
        goto LwnDepthStencilTest_FailTest;
    }

    // Vertex data
    vertexData = (Vertex *)__LWOG_MALLOC(sizeVBO);
    if (!vertexData) {
        goto LwnDepthStencilTest_free_Program;
    }
    else {
        // Populate random depth/stencil state and vertex data
        LWNDepthStencilBuckets buckets(numCells);
        Vertex *vPtr = vertexData;
        const LWNfloat verts[2] = { -1.0f, 3.0f };
        //LOG("\n");
        for (LWNuint i = 0; i < numCells; i++) {
            //LOG("%05d:\n", i);
            if (!buckets.randomize(dsState[i])) {
                goto LwnDepthStencilTest_free_vertexData;
            }
            LWNuint idx1 = 1, idx2 = 0;
            if (dsState[i].backFacing()) {
                idx1 = 0;
                idx2 = 1;
            }
            LWNfloat fDepth = dsState[i].getFloatDepth(1, depthMode);
            (vPtr++)->pos = vec3(verts[   0], verts[   0], fDepth);
            (vPtr++)->pos = vec3(verts[idx1], verts[idx2], fDepth);
            (vPtr++)->pos = vec3(verts[idx2], verts[idx1], fDepth);
        }
    }

    {   // Create framebuffer
        Framebuffer fb(numCols, numRows);
        fb.setFlags(TextureFlags::COMPRESSIBLE);
        fb.setColorFormat(Format::RGBA8);
        fb.setDepthStencilFormat(LWNDepthStencilPixelType::format);
        fb.alloc(device);
        fb.bind(cmd);
        cmd.SetViewport(0, 0, numCols, numRows);
        cmd.SetScissor(0, 0, numCols, numRows);
        cmd.SetDepthRange(0.0f, 1.0f);
        cmd.ClearColor(0, 0.0f, 0.0f, 0.0f, 1.0f);
        cmd.ClearDepthStencil(1.0f, LWN_TRUE, 0x00, 0xff);

        // Bind the program
        cmd.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);

        // Vertex stream, state, and buffer
        VertexStream stream = VertexStream(sizeof(Vertex));
        LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, pos);
        VertexArrayState vertexState = stream.CreateVertexArrayState();
        MemoryPoolAllocator vboAllocator(device, NULL, sizeVBO, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
        Buffer *vertexBuffer = stream.AllocateVertexBuffer(device, numVertices, vboAllocator, vertexData);
        cmd.BindVertexArrayState(vertexState);
        cmd.BindVertexBuffer(0, vertexBuffer->GetAddress(), sizeVBO);

        // Create polygon state to disable backface lwlling
        PolygonState polygonState;
        polygonState.SetDefaults();
        polygonState.SetLwllFace(Face::NONE);
        cmd.BindPolygonState(&polygonState);

        // Create a depth/stencil state
        DepthStencilState depthStencilState;
        depthStencilState.SetDefaults();
        switch (LWNDepthStencilPixelType::format) {
            case LWN_FORMAT_STENCIL8: {
                depthStencilState.SetDepthTestEnable(LWN_FALSE);
                depthStencilState.SetDepthWriteEnable(LWN_FALSE);
                depthStencilState.SetDepthFunc(DepthFunc::ALWAYS);
            } break;
            case LWN_FORMAT_DEPTH16:
            case LWN_FORMAT_DEPTH24: {
                depthStencilState.SetStencilTestEnable(LWN_FALSE);
                depthStencilState.SetStencilFunc(Face::FRONT_AND_BACK, StencilFunc::ALWAYS);
                depthStencilState.SetStencilOp(Face::FRONT_AND_BACK, StencilOp::KEEP, StencilOp::KEEP, StencilOp::KEEP);
                cmd.SetStencilRef(Face::FRONT_AND_BACK, 0x00);
                cmd.SetStencilValueMask(Face::FRONT_AND_BACK, 0xff);
                cmd.SetStencilMask(Face::FRONT_AND_BACK, 0xff);
            } break;
            default: break;
        }

        int submitCount = 0;
        // Loop through random DepthStencilState options
        for (LWNuint j = 0; j < numRows; j++) {
            for (LWNuint i = 0; i < numCols; i++) {
                LWNuint k = (j * numCols) + i;
                // Only testing one pixel
                cmd.SetScissor(i, j, 1, 1);

                // Clear the depth and stencil to the initial values
                cmd.ClearDepthStencil(dsState[k].getFloatDepth(0, depthMode), LWN_TRUE, dsState[k].stencilValue(), 0xff);

                // Populate and bind depth/stencil state
                switch (LWNDepthStencilPixelType::format) {
                    case LWN_FORMAT_STENCIL8: break;
                    default: {
                        if (dsState[k].depthEnable()) {
                            depthStencilState.SetDepthTestEnable(LWN_TRUE);
                            depthStencilState.SetDepthWriteEnable(dsState[k].depthWriteEnable());
                            depthStencilState.SetDepthFunc(dsState[k].depthFunc());
                        }
                        else {
                            depthStencilState.SetDepthTestEnable(LWN_FALSE);
                        }
                    } break;
                }
                switch (LWNDepthStencilPixelType::format) {
                    case LWN_FORMAT_DEPTH16:
                    case LWN_FORMAT_DEPTH24: break;
                    default: {
                        if (dsState[k].stencilEnable()) {
                            depthStencilState.SetStencilTestEnable(LWN_TRUE);
                            depthStencilState.SetStencilFunc(Face::FRONT, dsState[k].stencilFrontFunc());
                            depthStencilState.SetStencilOp(
                                Face::FRONT,
                                dsState[k].stencilFrontOpFail(),
                                dsState[k].stencilFrontOpZFail(),
                                dsState[k].stencilFrontOpZPass());
                            depthStencilState.SetStencilFunc(Face::BACK, dsState[k].stencilBackFunc());
                            depthStencilState.SetStencilOp(
                                Face::BACK,
                                dsState[k].stencilBackOpFail(),
                                dsState[k].stencilBackOpZFail(),
                                dsState[k].stencilBackOpZPass());
                            cmd.SetStencilRef(Face::FRONT, dsState[k].stencilFrontRef());
                            cmd.SetStencilRef(Face::BACK,  dsState[k].stencilBackRef());
                            cmd.SetStencilValueMask(Face::FRONT, dsState[k].stencilFrontValueMask());
                            cmd.SetStencilValueMask(Face::BACK,  dsState[k].stencilBackValueMask());
                            cmd.SetStencilMask(Face::FRONT, dsState[k].stencilFrontMask());
                            cmd.SetStencilMask(Face::BACK,  dsState[k].stencilBackMask());
                        }
                        else {
                            depthStencilState.SetStencilTestEnable(LWN_FALSE);
                        }
                    } break;
                }
                cmd.BindDepthStencilState(&depthStencilState);

                // Draw a single triangle
                cmd.DrawArrays(DrawPrimitive::TRIANGLES, 3*k, 3);

                // We seem to run out of control memory in this test if we run with the debug
                // layer enabled with draw-time validation enabled.  So submit+fence on every
                // 64th iteration of this test to avoid running out of space in the command
                // buffer.
                submitCount++;
                if ((submitCount & 0x3f) == 0) {
                    cmd.submit();
                    testDevice->insertFence();
                }
            }
        }

        cmd.submit();
        queue->Finish();

        // Copy back color and depth/stencil buffers
        ReadTextureData(device, queue, cmd, fb.getColorTexture(0), numCols, numRows, 1,      4,  cResult);
        ReadTextureData(device, queue, cmd, fb.getDepthTexture( ), numCols, numRows, 1, sizeDS, dsResult);

        // Clean up framebuffer
        fb.destroy();

        // Success
        ret = true;
    }

LwnDepthStencilTest_free_vertexData:
    __LWOG_FREE(vertexData);
LwnDepthStencilTest_free_Program:
    pgm->Free();
LwnDepthStencilTest_FailTest:
    delete testDevice;
    DeviceState::SetDefaultActive();

    return ret;
}

template <typename LWNDepthStencilPixelType>
bool LwnDepthStencilTest<LWNDepthStencilPixelType>::verify(const LWNdepthMode &depthMode) const
{
    bool ret = false;
    LWNuint *cResult = NULL;
    LWNDepthStencilPixelType *dsResult = NULL;
    LWNDepthStencilState *dsState = NULL;

    // Result color buffer
    cResult = new LWNuint [numCells];
    if (!cResult) {
        goto LwnDepthStencilTest_FailTest;
    }

    // Result depth/stencil buffer
    dsResult = new LWNDepthStencilPixelType [numCells];
    if (!dsResult) {
        goto LwnDepthStencilTest_free_cResult;
    }

    // Random depth/stencil states
    dsState = new LWNDepthStencilState [numCells];
    if (!dsState) {
        goto LwnDepthStencilTest_free_dsResult;
    }

    if (run(depthMode, cResult, dsResult, dsState)) {
        // Compare the results with the reference
        LWNuint colorFail = 0, depthFail = 0, stencilFail = 0;
        for (LWNuint j = 0; j < numRows; j++) {
            for (LWNuint i = 0; i < numCols; i++) {
                LWNuint k = (j * numCols) + i;
                LWNuint compare = dsState[k].compare(cResult[k], dsResult[k]);
                if (compare) {
                    if (compare & LWN_DEPTH_STENCIL_COLOR_MISMATCH) {
                        colorFail++;
                        LOG("ERROR: color   mismatch at (%d, %d) - (0x%08x vs 0x%08x)\n", i, j, dsState[k].getColor(), cResult[k]);
                    }
                    if (compare & LWN_DEPTH_STENCIL_DEPTH_MISMATCH) {
                        depthFail++;
                        LOG("ERROR: depth   mismatch at (%d, %d) - (", i, j);
                        switch (LWNDepthStencilPixelType::format) {
                            case LWN_FORMAT_DEPTH16:           LOG("0x%04x vs 0x%04x)\n", dsState[k].getDepth(), dsResult[k].depth()); break;
                            case LWN_FORMAT_DEPTH24:
                            case LWN_FORMAT_DEPTH24_STENCIL8:  LOG("0x%06x vs 0x%06x)\n", dsState[k].getDepth(), dsResult[k].depth()); break;
                            case LWN_FORMAT_DEPTH32F:
                            case LWN_FORMAT_DEPTH32F_STENCIL8: LOG("%f vs %f)\n", dsState[k].getDepth(), dsResult[k].depth()); break;
                            default: break;
                        }
                    }
                    if (compare & LWN_DEPTH_STENCIL_STENCIL_MISMATCH) {
                        stencilFail++;
                        LOG("ERROR: stencil mismatch at (%d, %d) - (0x%02x vs 0x%02x)\n", i, j, dsState[k].getStencil(), dsResult[k].stencil());
                    }
                    dsState[k].print(depthMode);
                }
            }
        }
        if (colorFail || depthFail || stencilFail) {
            LOG("ERROR: color(p:%d f:%d) depth(p:%d f:%d) stencil(p:%d f:%d)\n",
                numCells - colorFail, colorFail, numCells - depthFail, depthFail, numCells - stencilFail, stencilFail);
        }
        else {
            ret = true;
        }
    }

    delete [] dsState;
LwnDepthStencilTest_free_dsResult:
    delete [] dsResult;
LwnDepthStencilTest_free_cResult:
    delete [] cResult;
LwnDepthStencilTest_FailTest:
    return ret;
}

template <typename LWNDepthStencilPixelType>
void LwnDepthStencilTest<LWNDepthStencilPixelType>::initGraphics()
{
    lwnDefaultInitGraphics();
    DisableLWNObjectTracking();
}

template <typename LWNDepthStencilPixelType>
void LwnDepthStencilTest<LWNDepthStencilPixelType>::doGraphics()
{
    DeviceState *deviceState = DeviceState::GetActive();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Clear to green if pass, red if fail
    float clearColor[4] = { 1, 0, 0, 1 };
    if (verify(LWN_DEPTH_MODE_NEAR_IS_MINUS_W) && verify(LWN_DEPTH_MODE_NEAR_IS_ZERO)) {
        clearColor[0] = 0;
        clearColor[1] = 1;
    }
    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    queueCB.ClearColor(0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
    queueCB.submit();
    queue->Finish();
}

template <typename LWNDepthStencilPixelType>
void LwnDepthStencilTest<LWNDepthStencilPixelType>::exitGraphics()
{
    EnableLWNObjectTracking();
    lwnDefaultExitGraphics();
}

typedef LwnDepthStencilTest<LWNStencil8>         LwnDepthStencilTest_Stencil8;
typedef LwnDepthStencilTest<LWNDepth16>          LwnDepthStencilTest_Depth16;
typedef LwnDepthStencilTest<LWNDepth24>          LwnDepthStencilTest_Depth24;
typedef LwnDepthStencilTest<LWNDepth32F>         LwnDepthStencilTest_Depth32F;
typedef LwnDepthStencilTest<LWNDepth24Stencil8>  LwnDepthStencilTest_Depth24Stencil8;
typedef LwnDepthStencilTest<LWNDepth32FStencil8> LwnDepthStencilTest_Depth32FStencil8;

OGTEST_CppTest(LwnDepthStencilTest_Stencil8,         lwn_depth_stencil_s8,     );
OGTEST_CppTest(LwnDepthStencilTest_Depth16,          lwn_depth_stencil_d16,    );
OGTEST_CppTest(LwnDepthStencilTest_Depth24,          lwn_depth_stencil_d24,    );
OGTEST_CppTest(LwnDepthStencilTest_Depth32F,         lwn_depth_stencil_d32f,   );
OGTEST_CppTest(LwnDepthStencilTest_Depth24Stencil8,  lwn_depth_stencil_d24s8,  );
OGTEST_CppTest(LwnDepthStencilTest_Depth32FStencil8, lwn_depth_stencil_d32fs8, );
