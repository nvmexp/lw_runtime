!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     3
.MAX_IBUF    4
.MAX_OBUF    29
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v1025-lw40.s -o allprogs-new32//v1025-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[0].C[0]
#semantic C[95].C[95]
#semantic C[94].C[94]
#semantic C[93].C[93]
#semantic C[92].C[92]
#semantic C[91].C[91]
#semantic C[90].C[90]
#var float4 o[TEX7] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX6] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX5] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX4] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX3] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 C[95] :  : c[95] : -1 : 0
#var float4 C[94] :  : c[94] : -1 : 0
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
#obuf 12 = o[TEX3].z
#obuf 13 = o[TEX3].w
#obuf 14 = o[TEX4].x
#obuf 15 = o[TEX4].y
#obuf 16 = o[TEX4].z
#obuf 17 = o[TEX4].w
#obuf 18 = o[TEX5].x
#obuf 19 = o[TEX5].y
#obuf 20 = o[TEX5].z
#obuf 21 = o[TEX5].w
#obuf 22 = o[TEX6].x
#obuf 23 = o[TEX6].y
#obuf 24 = o[TEX6].z
#obuf 25 = o[TEX6].w
#obuf 26 = o[TEX7].x
#obuf 27 = o[TEX7].y
#obuf 28 = o[TEX7].z
#obuf 29 = o[TEX7].w
BB0:
FADD     o[26], v[3], c[380];
FADD     o[27], v[4], c[381];
FADD     o[28], v[4], -c[381];
FADD     o[29], v[3], -c[380];
FADD     o[22], v[3], c[376];
FADD     o[23], v[4], c[377];
FADD     o[24], v[4], -c[377];
FADD     o[25], v[3], -c[376];
FADD     o[18], v[3], c[372];
FADD     o[19], v[4], c[373];
FADD     o[20], v[4], -c[373];
FADD     o[21], v[3], -c[372];
FADD     o[14], v[3], c[368];
FADD     o[15], v[4], c[369];
FADD     o[16], v[4], -c[369];
FADD     o[17], v[3], -c[368];
FADD     o[10], v[3], c[364];
FADD     o[11], v[4], c[365];
FADD     o[12], v[4], -c[365];
FADD     o[13], v[3], -c[364];
FADD     o[8], v[3], -c[360];
FADD     o[9], v[4], -c[361];
FADD     o[6], v[3], c[360];
FADD     o[7], v[4], c[361];
MOV      o[4], v[3];
MOV      o[5], v[4];
MOV      o[0], v[0];
MOV      o[1], v[1];
MOV      o[2], v[2];
MOV32    R0, c[1];
MOV32    o[3], R0;
END
# 31 instructions, 4 R-regs
# 31 inst, (7 mov, 0 mvi, 0 tex, 0 complex, 24 math)
#    29 64-bit, 2 32-bit, 0 32-bit-const
