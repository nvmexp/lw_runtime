#define IEEE_DOUBLE_EXPO_BIAS  (1023)

LW_DOUBLE
LwDblMul (LW_DOUBLE a, LW_DOUBLE b) 
{
    struct {
        unsigned int lo;
        unsigned int hi;
    } xx, yy;
    unsigned int expo_x, expo_y, t, prod0, prod1, prod2, prod3, u, s;
    // we'd like a union here with xx.i.lo, x.i.hi, etc!
    unsigned short xlolo, xlohi, xhilo, xhihi;
    unsigned short ylolo, ylohi, yhilo, yhihi;

    xx.lo = floatToRawIntBits (a.x);
    xx.hi = floatToRawIntBits (a.y);
    yy.lo = floatToRawIntBits (b.x);
    yy.hi = floatToRawIntBits (b.y);

    expo_y = 0x7FF;
    t = xx.hi >> 20;
    expo_x = expo_y & t;
    expo_x = expo_x - 1;
    t = yy.hi >> 20;
    expo_y = expo_y & t;
    expo_y = expo_y - 1;

    if ((expo_x <= 0x7FD) && 
        (expo_y <= 0x7FD)) {

//multiply:
        expo_x = expo_x + expo_y;
        expo_y = xx.hi ^ yy.hi;

        t = xx.lo >> 21;
        xx.lo = xx.lo << 11;
        xx.hi = xx.hi << 11;
        xx.hi = xx.hi | t;
        yy.hi = yy.hi & 0x001fffff;
        xx.hi = xx.hi | 0x80000000;
        yy.hi = yy.hi | 0x00100000;

        /* multiply mantissas: 16 multiplies of 16x16->32 bits */
        xlolo = (unsigned short)(xx.lo & 0xffff);
        xlohi = (unsigned short)(xx.lo >> 16);
        xhilo = (unsigned short)(xx.hi & 0xffff);
        xhihi = (unsigned short)(xx.hi >> 16);
        ylolo = (unsigned short)(yy.lo & 0xffff);
        ylohi = (unsigned short)(yy.lo >> 16);
        yhilo = (unsigned short)(yy.hi & 0xffff);
        yhihi = (unsigned short)(yy.hi >> 16);
        
        prod0 = ((unsigned int)xlolo) * ylolo;
        prod1 = ((unsigned int)xlohi) * ylolo;
        prod2 = ((unsigned int)xlolo) * ylohi;
        s = prod0 >> 16;
        s = s + (prod1 & 0xffff);
        s = s + (prod2 & 0xffff);
        prod0 = prod0 & 0xffff;               /* bits <15:0> */
        prod0 = prod0 + (s << 16);            /* bits <31:0> */
        s = s >> 16;
        s = s + (prod1 >> 16);
        s = s + (prod2 >> 16);
        prod1 = ((unsigned int)xhilo) * ylolo;
        prod2 = ((unsigned int)xlohi) * ylohi;
        prod3 = ((unsigned int)xlolo) * yhilo;
        s = s + (prod1 & 0xffff);
        s = s + (prod2 & 0xffff);
        s = s + (prod3 & 0xffff);
        t = s & 0xffff;                       /* bits <47:32> */
        s = s >> 16;
        s = s + (prod1 >> 16);
        s = s + (prod2 >> 16);
        s = s + (prod3 >> 16);
        prod1 = ((unsigned int)xhihi) * ylolo;
        prod2 = ((unsigned int)xhilo) * ylohi;
        prod3 = ((unsigned int)xlohi) * yhilo;
        u     = ((unsigned int)xlolo) * yhihi;
        s = s + (prod1 & 0xffff);
        s = s + (prod2 & 0xffff);
        s = s + (prod3 & 0xffff);
        s = s + (u     & 0xffff);             
        t = t + (s << 16);                    /* bits <63:32> */
        s = s >> 16; 
        s = s + (prod1 >> 16);
        s = s + (prod2 >> 16);
        s = s + (prod3 >> 16);
        s = s + (u     >> 16);
        prod1 = t;                            /* bits <63:32> */
        prod2 = ((unsigned int)xhihi) * ylohi;
        prod3 = ((unsigned int)xhilo) * yhilo;
        u     = ((unsigned int)xlohi) * yhihi;
        s = s + (prod2 & 0xffff);
        s = s + (prod3 & 0xffff);
        s = s + (u     & 0xffff);
        t = s & 0xffff;                       /* bits <79:64> */
        s = s >> 16;
        s = s + (prod2 >> 16);
        s = s + (prod3 >> 16);
        s = s + (u     >> 16);
        prod3 = ((unsigned int)xhihi) * yhilo;
        u     = ((unsigned int)xhilo) * yhihi;
        s = s + (prod3 & 0xffff);
        s = s + (u & 0xffff);
        prod2 = t + (s << 16);                /* bits <95:80> */
        s = s >> 16;
        s = s + (prod3 >> 16);
        s = s + (u     >> 16);
        prod3 = ((unsigned int)xhihi) * yhihi;       
        prod3 = prod3 + s;                    /* bits <127:96> */
        
        yy.lo = prod0;
        yy.hi = prod1;
        xx.lo = prod2;
        xx.hi = prod3;

        expo_x = expo_x - (IEEE_DOUBLE_EXPO_BIAS - 2);
        expo_y = expo_y & 0x80000000;

        /* normalize mantissa */
        t = xx.hi < 0x00100000;
        s = xx.lo >> 31;
        s = (xx.hi << 1) + s;
        xx.hi = t ? s : xx.hi;
        s = yy.hi >> 31;
        s = (xx.lo << 1) + s;
        xx.lo = t ? s : xx.lo;
        s = yy.lo >> 31;
        s = (yy.hi << 1) + s;
        yy.hi = t ? s : yy.hi;
        s = yy.lo << 1;
        yy.lo = t ? s : yy.lo;
        expo_x  = t ? (expo_x - 1) : expo_x;

        if (expo_x <= 0x7FD) {
            xx.hi = xx.hi & ~0x00100000;  /* lop off integer bit */
            xx.hi = xx.hi | expo_y;       /* OR in sign bit */
            t = expo_x << 20;
            xx.hi = xx.hi + t;
            xx.hi = xx.hi + 0x00100000;   /* add in expo*/
            /* round result to nearest-even */
            t = yy.lo ? 1 : 0;
            yy.hi = yy.hi | t;            /* implement sticky bit */
            s = xx.lo & 1;
            u = yy.hi >> 31;
            t = yy.hi == 0x80000000;
            s = t ? s : u;    
            t = xx.lo;
            xx.lo = xx.lo + s;
            t = (t > xx.lo) ? 1 : 0;
            xx.hi = xx.hi + t;            /* propagate carry */
        } else if ((int)expo_x >= 2046) {
            /* overflow: return infinity */
            xx.hi = expo_y | 0x7FF00000;
            xx.lo = 0;
        } else {
            /* zero, denormal, or smallest normal */
            expo_x = ((unsigned int)-((int)expo_x));
            if (expo_x > 54) {
                /* massive underflow: return 0 */
                xx.hi = expo_y;
                xx.lo = 0;
            } else {
                /* underflow: denormalize and round */
                t = yy.lo ? 1 : 0;
                yy.hi = yy.hi | t;        /*implement sticky bit*/
                if (expo_x >= 32) {
                    t = yy.hi ? 1 : 0;
                    yy.hi = xx.lo | t;
                    xx.lo = xx.hi;
                    xx.hi = 0;
                    expo_x -= 32;
                }
                if (expo_x) {
                    s = 32 - expo_x;
                    t = yy.hi ? 1 : 0;
                    yy.hi = (xx.lo << s) | t;
                    t = xx.lo >> expo_x;
                    xx.lo = (xx.hi << s) | t;
                    xx.hi = xx.hi >> expo_x;
                }
                /* round result to nearest-even */
                expo_x = xx.lo;
                xx.lo = xx.lo + ((yy.hi == 0x80000000) ? 
                                     (xx.lo & 1) : (yy.hi >> 31));
                if (expo_x > xx.lo) xx.hi++;  /* propagate carry bit */
                xx.hi = xx.hi | expo_y;       /* OR in sign bit */
            }
        }
    } else {
        t = xx.hi ^ yy.hi;
        t = t & 0x80000000;
        s = xx.hi + xx.hi;
        u = yy.hi + yy.hi;
        if (!(s | xx.lo)) {
            if (expo_y != 2046) {
                /* x == 0, y != NaN, Inf. Return 0 */
                xx.hi = t;
                xx.lo = 0;
            } else {
                expo_y = (yy.lo ? 1 : 0);
                expo_y = expo_y | u;
                if (expo_y == 0xFFE00000) {
                    /* x == 0, y == Inf; return INDEFINITE */
                    xx.hi = expo_y | 0x00180000;
                    xx.lo = yy.lo;
                } else {
                    /* x == 0, y == NaN; return NaN colwerted to QNaN */
                    xx.hi = yy.hi | 0x00080000;
                    xx.lo = yy.lo;
                }
            }
        } else if (!(u | yy.lo)) {
            if (expo_x != 2046) {
                /* y == 0, x != NaN, Inf. Return 0 */
                xx.hi = t;
                xx.lo = 0;
            } else {
                expo_x = (xx.lo ? 1 : 0);
                expo_x = expo_x | s;
                if (expo_x == 0xFFE00000) {
                    /* y == 0, x == Inf; return INDEFINITE */
                    xx.hi = expo_x | 0x00180000;
                    xx.lo = 0;
                } else {
                    /* y == 0, x == NaN; return NaN colwerted to QNaN */
                    xx.hi = xx.hi | 0x00080000;                    
                }
            }
        } else if ((expo_y != 2046) && (expo_x != 2046)) {
            expo_y++;
            expo_x++;
            /*
             * If both operands are denormals, we only need to normalize 
             * one of them as the result will be either a denormal or zero.
             */
            if (expo_x == 0) {
                t = xx.hi & 0x80000000;
                s = xx.lo >> 21;
                xx.lo = xx.lo << 11;
                xx.hi = xx.hi << 11;
                xx.hi = xx.hi | s;
                if (!xx.hi) {
                    xx.hi = xx.lo;
                    xx.lo = 0;
                    expo_x -= 32;
                }
                while (!(xx.hi & 0x80000000)) {
                    s = xx.lo >> 31;
                    xx.lo = xx.lo + xx.lo;
                    xx.hi = xx.hi + xx.hi;
                    xx.hi = xx.hi | s;
                    expo_x--;
                }
                
                xx.lo = (xx.lo >> 11) | (xx.hi << 21);
                xx.hi = (xx.hi >> 11) | t;
                expo_y--;
//                goto multiply;
            } else if (expo_y == 0) {
                t = yy.hi & 0x80000000;
                yy.hi = (yy.hi << 11) | (yy.lo >> 21);
                yy.lo = yy.lo << 11;
                if (!yy.hi) {
                    yy.hi = yy.lo;
                    yy.lo = 0;
                    expo_y -= 32;
                }
                while (!(yy.hi & 0x80000000)) {
                    yy.hi = (yy.hi << 1) | (yy.lo >> 31);
                    yy.lo = yy.lo << 1;
                    expo_y--;
                }
                yy.lo = (yy.lo >> 11) | (yy.hi << 21);
                yy.hi = (yy.hi >> 11) | t;
                expo_x--;
//                goto multiply;
            }
            /* we don't have goto in Cg, and we don't want to slow down
             * the fastpath, so copy the whole block starting at label
             * multiply to here.
             */

//multiply:
            expo_x = expo_x + expo_y;
            expo_y = xx.hi ^ yy.hi;
            
            t = xx.lo >> 21;
            xx.lo = xx.lo << 11;
            xx.hi = xx.hi << 11;
            xx.hi = xx.hi | t;
            yy.hi = yy.hi & 0x001fffff;
            xx.hi = xx.hi | 0x80000000;
            yy.hi = yy.hi | 0x00100000;
            
            /* multiply mantissas: 16 multiplies of 16x16->32 bits */
            xlolo = (unsigned short)(xx.lo & 0xffff);
            xlohi = (unsigned short)(xx.lo >> 16);
            xhilo = (unsigned short)(xx.hi & 0xffff);
            xhihi = (unsigned short)(xx.hi >> 16);
            ylolo = (unsigned short)(yy.lo & 0xffff);
            ylohi = (unsigned short)(yy.lo >> 16);
            yhilo = (unsigned short)(yy.hi & 0xffff);
            yhihi = (unsigned short)(yy.hi >> 16);
            
            prod0 = ((unsigned int)xlolo) * ylolo;
            prod1 = ((unsigned int)xlohi) * ylolo;
            prod2 = ((unsigned int)xlolo) * ylohi;
            s = prod0 >> 16;
            s = s + (prod1 & 0xffff);
            s = s + (prod2 & 0xffff);
            prod0 = prod0 & 0xffff;               /* bits <15:0> */
            prod0 = prod0 + (s << 16);            /* bits <31:0> */
            s = s >> 16;
            s = s + (prod1 >> 16);
            s = s + (prod2 >> 16);
            prod1 = ((unsigned int)xhilo) * ylolo;
            prod2 = ((unsigned int)xlohi) * ylohi;
            prod3 = ((unsigned int)xlolo) * yhilo;
            s = s + (prod1 & 0xffff);
            s = s + (prod2 & 0xffff);
            s = s + (prod3 & 0xffff);
            t = s & 0xffff;                       /* bits <47:32> */
            s = s >> 16;
            s = s + (prod1 >> 16);
            s = s + (prod2 >> 16);
            s = s + (prod3 >> 16);
            prod1 = ((unsigned int)xhihi) * ylolo;
            prod2 = ((unsigned int)xhilo) * ylohi;
            prod3 = ((unsigned int)xlohi) * yhilo;
            u     = ((unsigned int)xlolo) * yhihi;
            s = s + (prod1 & 0xffff);
            s = s + (prod2 & 0xffff);
            s = s + (prod3 & 0xffff);
            s = s + (u     & 0xffff);             
            t = t + (s << 16);                    /* bits <63:32> */
            s = s >> 16; 
            s = s + (prod1 >> 16);
            s = s + (prod2 >> 16);
            s = s + (prod3 >> 16);
            s = s + (u     >> 16);
            prod1 = t;                            /* bits <63:32> */
            prod2 = ((unsigned int)xhihi) * ylohi;
            prod3 = ((unsigned int)xhilo) * yhilo;
            u     = ((unsigned int)xlohi) * yhihi;
            s = s + (prod2 & 0xffff);
            s = s + (prod3 & 0xffff);
            s = s + (u     & 0xffff);
            t = s & 0xffff;                       /* bits <79:64> */
            s = s >> 16;
            s = s + (prod2 >> 16);
            s = s + (prod3 >> 16);
            s = s + (u     >> 16);
            prod3 = ((unsigned int)xhihi) * yhilo;
            u     = ((unsigned int)xhilo) * yhihi;
            s = s + (prod3 & 0xffff);
            s = s + (u & 0xffff);
            prod2 = t + (s << 16);                /* bits <95:80> */
            s = s >> 16;
            s = s + (prod3 >> 16);
            s = s + (u     >> 16);
            prod3 = ((unsigned int)xhihi) * yhihi;       
            prod3 = prod3 + s;                    /* bits <127:96> */
            
            yy.lo = prod0;
            yy.hi = prod1;
            xx.lo = prod2;
            xx.hi = prod3;
            
            expo_x = expo_x - (IEEE_DOUBLE_EXPO_BIAS - 2);
            expo_y = expo_y & 0x80000000;
            
            /* normalize mantissa */
            t = xx.hi < 0x00100000;
            s = xx.lo >> 31;
            s = (xx.hi << 1) + s;
            xx.hi = t ? s : xx.hi;
            s = yy.hi >> 31;
            s = (xx.lo << 1) + s;
            xx.lo = t ? s : xx.lo;
            s = yy.lo >> 31;
            s = (yy.hi << 1) + s;
            yy.hi = t ? s : yy.hi;
            s = yy.lo << 1;
            yy.lo = t ? s : yy.lo;
            expo_x  = t ? (expo_x - 1) : expo_x;
            
            if (expo_x <= 0x7FD) {
                xx.hi = xx.hi & ~0x00100000;  /* lop off integer bit */
                xx.hi = xx.hi | expo_y;       /* OR in sign bit */
                t = expo_x << 20;
                xx.hi = xx.hi + t;
                xx.hi = xx.hi + 0x00100000;   /* add in expo*/
                /* round result to nearest-even */
                t = yy.lo ? 1 : 0;
                yy.hi = yy.hi | t;            /* implement sticky bit*/
                s = xx.lo & 1;
                u = yy.hi >> 31;
                t = yy.hi == 0x80000000;
                s = t ? s : u;    
                t = xx.lo;
                xx.lo = xx.lo + s;
                t = (t > xx.lo) ? 1 : 0;
                xx.hi = xx.hi + t;            /* propagate carry */
            } else if ((int)expo_x >= 2046) {
                /* overflow: return infinity */
                xx.hi = expo_y | 0x7FF00000;
                xx.lo = 0;
            } else {
                /* zero, denormal, or smallest normal */
                expo_x = ((unsigned int)-((int)expo_x));
                if (expo_x > 54) {
                    /* massive underflow: return 0 */
                    xx.hi = expo_y;
                    xx.lo = 0;
                } else {
                    /* underflow: denormalize and round */
                    t = yy.lo ? 1 : 0;
                    yy.hi = yy.hi | t;        /*implement sticky bit*/
                    if (expo_x >= 32) {
                        t = yy.hi ? 1 : 0;
                        yy.hi = xx.lo | t;
                        xx.lo = xx.hi;
                        xx.hi = 0;
                        expo_x -= 32;
                    }
                    if (expo_x) {
                        s = 32 - expo_x;
                        t = yy.hi ? 1 : 0;
                        yy.hi = (xx.lo << s) | t;
                        t = xx.lo >> expo_x;
                        xx.lo = (xx.hi << s) | t;
                        xx.hi = xx.hi >> expo_x;
                    }
                    /* round result to nearest-even */
                    expo_x = xx.lo;
                    xx.lo = xx.lo + ((yy.hi == 0x80000000) ? 
                                     (xx.lo & 1) : (yy.hi >> 31));
                    if (expo_x > xx.lo) xx.hi++;  /* propagate CY bit*/
                    xx.hi = xx.hi | expo_y;       /* OR in sign bit */
                }
            }
            /* end of replicated block */
        } else {
            expo_x = (xx.lo ? 1 : 0);
            expo_x = expo_x | s;
            expo_y = (yy.lo ? 1 : 0);
            expo_y = expo_y | u;
            /* if x is NaN, return x */
            if (expo_x > 0xFFE00000) {
                /* cvt any SNaNs to QNaNs */
                xx.hi = xx.hi | 0x00080000;
            }
            /* if y is NaN, return y */
            else if (expo_y > 0xFFE00000) {
                /* cvt any SNaNs to QNaNs */
                xx.hi = yy.hi | 0x00080000;
                xx.lo = yy.lo;
            } 
            /* x * infinity, infinity * y ==> return Inf */
            else {
                xx.hi = t | 0x7ff00000;
                xx.lo = 0;
            }
        }
    }
    a.x = intBitsToFloat (xx.lo);
    a.y = intBitsToFloat (xx.hi);
    return a;
}
