!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     3
.MAX_IBUF    4
.MAX_OBUF    11
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v1024-lw40.s -o allprogs-new32//v1024-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[0].C[0]
#semantic C[93].C[93]
#semantic C[92].C[92]
#semantic C[91].C[91]
#semantic C[90].C[90]
#var float4 o[TEX3] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 C[93] :  : c[93] : -1 : 0
#var float4 C[92] :  : c[92] : -1 : 0
#var float4 C[91] :  : c[91] : -1 : 0
#var float4 C[90] :  : c[90] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[WGT].x
#ibuf 4 = v[WGT].y
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[TEX0].x
#obuf 5 = o[TEX0].y
#obuf 6 = o[TEX1].x
#obuf 7 = o[TEX1].y
#obuf 8 = o[TEX2].x
#obuf 9 = o[TEX2].y
#obuf 10 = o[TEX3].x
#obuf 11 = o[TEX3].y
BB0:
FADD     o[10], v[3], c[372];
FADD     o[11], v[4], c[373];
FADD     o[8], v[3], c[368];
FADD     o[9], v[4], c[369];
FADD     o[6], v[3], c[364];
FADD     o[7], v[4], c[365];
FADD     o[4], v[3], c[360];
FADD     o[5], v[4], c[361];
MOV      o[0], v[0];
MOV      o[1], v[1];
MOV      o[2], v[2];
MOV32    R0, c[1];
MOV32    o[3], R0;
END
# 13 instructions, 4 R-regs
# 13 inst, (5 mov, 0 mvi, 0 tex, 0 complex, 8 math)
#    11 64-bit, 2 32-bit, 0 32-bit-const
