!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     3
.MAX_IBUF    11
.MAX_OBUF    15
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v873-lw40.s -o allprogs-new32//v873-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[93].C[93]
#semantic C[92].C[92]
#semantic C[91].C[91]
#semantic C[90].C[90]
#semantic C[16].C[16]
#semantic C[7].C[7]
#semantic C[6].C[6]
#semantic C[5].C[5]
#semantic C[4].C[4]
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[FOGC] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 v[TEX0] : $vin.F : F[0] : -1 : 0
#var float4 C[93] :  : c[93] : -1 : 0
#var float4 C[92] :  : c[92] : -1 : 0
#var float4 C[91] :  : c[91] : -1 : 0
#var float4 C[90] :  : c[90] : -1 : 0
#var float4 v[TEX9] : $vin.F : F[0] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[7] :  : c[7] : -1 : 0
#var float4 C[6] :  : c[6] : -1 : 0
#var float4 C[5] :  : c[5] : -1 : 0
#var float4 C[4] :  : c[4] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[UNUSED1].x
#ibuf 5 = v[UNUSED1].y
#ibuf 6 = v[UNUSED1].z
#ibuf 7 = v[UNUSED1].w
#ibuf 8 = v[TEX0].x
#ibuf 9 = v[TEX0].y
#ibuf 10 = v[TEX0].z
#ibuf 11 = v[TEX0].w
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
#obuf 10 = o[TEX2].z
#obuf 11 = o[TEX2].w
#obuf 12 = o[FOGC].x
#obuf 13 = o[FOGC].y
#obuf 14 = o[FOGC].z
#obuf 15 = o[FOGC].w
BB0:
FMUL     R0, v[1], c[25];
MOV32    R1, c[67];
FMAD     R0, v[0], c[24], R0;
MOV      o[8], v[8];
MOV      o[9], v[9];
FMAD     R0, v[2], c[26], R0;
MOV      o[10], v[10];
MOV      o[11], v[11];
FMAD     R2, v[3], c[27], R0;
FMUL     R0, v[5], c[369];
FMAD     R3, -R2, R1, c[64];
FMAD     R0, v[4], c[368], R0;
FMUL     R1, v[5], c[373];
MOV32    o[12], R3;
FMAD     R0, v[6], c[370], R0;
FMAD     R1, v[4], c[372], R1;
MOV32    o[13], R3;
FMAD     o[6], v[7], c[371], R0;
FMAD     R0, v[6], c[374], R1;
MOV32    o[14], R3;
MOV32    o[15], R3;
FMAD     o[7], v[7], c[375], R0;
FMUL     R0, v[5], c[361];
FMUL     R1, v[5], c[365];
FMUL     R3, v[1], c[17];
FMAD     R0, v[4], c[360], R0;
FMAD     R1, v[4], c[364], R1;
FMAD     R3, v[0], c[16], R3;
FMAD     R0, v[6], c[362], R0;
FMAD     R1, v[6], c[366], R1;
FMAD     R3, v[2], c[18], R3;
FMAD     o[4], v[7], c[363], R0;
FMAD     o[5], v[7], c[367], R1;
FMAD     o[0], v[3], c[19], R3;
MOV32    o[2], R2;
FMUL     R0, v[1], c[21];
FMUL     R1, v[1], c[29];
FMAD     R0, v[0], c[20], R0;
FMAD     R1, v[0], c[28], R1;
FMAD     R0, v[2], c[22], R0;
FMAD     R1, v[2], c[30], R1;
FMAD     o[1], v[3], c[23], R0;
FMAD     o[3], v[3], c[31], R1;
END
# 43 instructions, 4 R-regs
# 43 inst, (10 mov, 0 mvi, 0 tex, 0 complex, 33 math)
#    37 64-bit, 6 32-bit, 0 32-bit-const
