!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     7
.MAX_IBUF    6
.MAX_OBUF    36
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v1142-lw40.s -o allprogs-new32//v1142-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[91].C[91]
#semantic C[90].C[90]
#semantic C[16].C[16]
#semantic C[2].C[2]
#semantic C[10].C[10]
#semantic C[11].C[11]
#semantic C[9].C[9]
#semantic C[8].C[8]
#semantic C[0].C[0]
#semantic c.c
#semantic C[1].C[1]
#var float4 o[TEX7] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX6] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX5] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX4] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX3] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[FOGC] : $vout.O : O[0] : -1 : 0
#var float4 o[BCOL1] : $vout.O : O[0] : -1 : 0
#var float4 o[BCOL0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 C[91] :  : c[91] : -1 : 0
#var float4 C[90] :  : c[90] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[WGT].z
#ibuf 5 = v[NOR].x
#ibuf 6 = v[NOR].y
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[BCOL0].x
#obuf 5 = o[BCOL0].y
#obuf 6 = o[BCOL0].z
#obuf 7 = o[BCOL0].w
#obuf 8 = o[BCOL1].x
#obuf 9 = o[BCOL1].y
#obuf 10 = o[BCOL1].z
#obuf 11 = o[BCOL1].w
#obuf 12 = o[TEX0].x
#obuf 13 = o[TEX0].y
#obuf 14 = o[TEX1].x
#obuf 15 = o[TEX1].y
#obuf 16 = o[TEX2].x
#obuf 17 = o[TEX2].y
#obuf 18 = o[TEX3].x
#obuf 19 = o[TEX3].y
#obuf 20 = o[TEX3].z
#obuf 21 = o[TEX4].x
#obuf 22 = o[TEX4].y
#obuf 23 = o[TEX4].z
#obuf 24 = o[TEX5].x
#obuf 25 = o[TEX5].y
#obuf 26 = o[TEX5].z
#obuf 27 = o[TEX6].x
#obuf 28 = o[TEX6].y
#obuf 29 = o[TEX6].z
#obuf 30 = o[TEX7].x
#obuf 31 = o[TEX7].y
#obuf 32 = o[TEX7].z
#obuf 33 = o[FOGC].x
#obuf 34 = o[FOGC].y
#obuf 35 = o[FOGC].z
#obuf 36 = o[FOGC].w
BB0:
FMUL     R0, v[4], c[4];
R2A      A0, R0;
FMUL     R0, v[1], c[A0 + 169];
FMUL     R1, v[1], c[A0 + 173];
FMAD     R0, v[0], c[A0 + 168], R0;
FMAD     R1, v[0], c[A0 + 172], R1;
FMUL     R2, v[1], c[A0 + 177];
FMAD     R0, v[2], c[A0 + 170], R0;
FMAD     R1, v[2], c[A0 + 174], R1;
FMAD     R2, v[0], c[A0 + 176], R2;
FMAD     R0, v[3], c[A0 + 171], R0;
FMAD     R1, v[3], c[A0 + 175], R1;
FMAD     R2, v[2], c[A0 + 178], R2;
MOV32    R3, c[43];
FMUL32   R4, R1, c[41];
FMAD     R2, v[3], c[A0 + 179], R2;
FMAD     R5, R0, c[40], R4;
MOV32    R4, c[67];
FMUL32   R6, R1, c[33];
FMAD     R5, R2, c[42], R5;
FMAD     R6, R0, c[32], R6;
FMAD     R3, R3, c[1], R5;
MOV32    R5, c[35];
FMAD     R6, R2, c[34], R6;
FMAD     R4, -R3, R4, c[64];
FMUL32   R7, R1, c[37];
MOV32    o[33], R4;
FMAD     o[0], R5, c[1], R6;
FMAD     R5, R0, c[36], R7;
MOV32    o[2], R3;
MOV32    o[34], R4;
FMAD     R3, R2, c[38], R5;
MOV32    o[35], R4;
MOV32    o[36], R4;
MOV32    R4, c[39];
FMUL32   R6, R1, c[45];
MOV32    R5, c[47];
FMAD     R6, R0, c[44], R6;
FMAD     o[1], R4, c[1], R3;
FMAD     R4, R2, c[46], R6;
MOV32    o[30], c[0];
MOV32    R3, c[0];
FMAD     o[3], R5, c[1], R4;
MOV32    R4, c[0];
MOV32    o[31], R3;
MOV32    o[27], c[0];
MOV32    o[32], R4;
MOV32    R3, c[0];
MOV32    R4, c[0];
MOV32    o[24], c[0];
MOV32    o[28], R3;
MOV32    o[29], R4;
MOV32    R3, c[0];
MOV32    R4, c[0];
FADD32   o[18], -R0, c[8];
MOV32    o[25], R3;
MOV32    o[26], R4;
FADD32   o[19], -R1, c[9];
FADD32   o[20], -R2, c[10];
MOV32    o[21], c[0];
MOV32    R0, c[0];
MOV32    R1, c[0];
FMUL     R2, v[5], c[360];
MOV32    o[22], R0;
MOV32    o[23], R1;
FMAD     R0, v[6], c[361], R2;
FMUL     R1, v[5], c[364];
MOV32    o[8], c[0];
MOV32    o[16], R0;
FMAD     R1, v[6], c[365], R1;
MOV32    o[14], R0;
MOV32    o[12], R0;
MOV32    o[17], R1;
MOV32    o[15], R1;
MOV32    o[13], R1;
MOV32    R0, c[0];
MOV32    R1, c[0];
MOV32    R2, c[0];
MOV32    o[9], R0;
MOV32    o[10], R1;
MOV32    o[11], R2;
MOV32    o[4], c[0];
MOV32    R0, c[0];
MOV32    R1, c[0];
MOV32    R2, c[0];
MOV32    o[5], R0;
MOV32    o[6], R1;
MOV32    o[7], R2;
END
# 88 instructions, 8 R-regs
# 88 inst, (51 mov, 0 mvi, 0 tex, 0 complex, 37 math)
#    31 64-bit, 57 32-bit, 0 32-bit-const
