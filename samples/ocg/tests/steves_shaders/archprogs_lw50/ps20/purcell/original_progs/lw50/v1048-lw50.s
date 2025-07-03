!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     19
.MAX_IBUF    16
.MAX_OBUF    36
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v1048-lw40.s -o allprogs-new32//v1048-lw50.s
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
#var float4 v[COL1] : $vin.F : F[0] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 v[TEX8] : $vin.F : F[0] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[WGT].x
#ibuf 5 = v[WGT].y
#ibuf 6 = v[NOR].y
#ibuf 7 = v[NOR].z
#ibuf 8 = v[COL0].x
#ibuf 9 = v[COL0].y
#ibuf 10 = v[COL0].z
#ibuf 11 = v[COL1].x
#ibuf 12 = v[COL1].y
#ibuf 13 = v[UNUSED0].x
#ibuf 14 = v[UNUSED0].y
#ibuf 15 = v[UNUSED0].z
#ibuf 16 = v[UNUSED0].w
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
FMUL     R0, v[7], c[4];
FMUL     R1, v[6], c[4];
R2A      A0, R0;
R2A      A1, R1;
FMUL     R2, v[5], c[A1 + 172];
FMUL     R1, v[5], c[A1 + 173];
FMUL     R0, v[5], c[A1 + 174];
FMAD     R4, v[4], c[A0 + 172], R2;
FMAD     R5, v[4], c[A0 + 173], R1;
FMAD     R3, v[4], c[A0 + 174], R0;
FMUL     R2, v[5], c[A1 + 176];
FMUL     R1, v[14], R5;
FMUL     R0, v[5], c[A1 + 177];
FMAD     R8, v[4], c[A0 + 176], R2;
FMAD     R1, v[13], R4, R1;
FMAD     R9, v[4], c[A0 + 177], R0;
FMUL     R0, v[5], c[A1 + 178];
FMAD     R12, v[15], R3, R1;
FMUL     R1, v[9], R9;
FMAD     R7, v[4], c[A0 + 178], R0;
FMUL     R0, v[9], R5;
FMAD     R2, v[8], R8, R1;
FMUL     R1, v[14], R9;
FMAD     R0, v[8], R4, R0;
FMAD     R10, v[10], R7, R2;
FMAD     R1, v[13], R8, R1;
FMAD     R6, v[10], R3, R0;
FMUL32   R0, R12, R10;
FMAD     R11, v[15], R7, R1;
FMUL     R1, v[5], c[A1 + 168];
FMUL     R2, v[5], c[A1 + 169];
FMAD     R0, R6, R11, -R0;
FMAD     R1, v[4], c[A0 + 168], R1;
FMAD     R14, v[4], c[A0 + 169], R2;
FMUL     o[24], v[16], R0;
FMUL     R0, v[5], c[A1 + 170];
FMUL     R2, v[9], R14;
FMUL     R13, v[14], R14;
FMAD     R0, v[4], c[A0 + 170], R0;
FMAD     R2, v[8], R1, R2;
FMAD     R13, v[13], R1, R13;
FMAD     R2, v[10], R0, R2;
FMAD     R13, v[15], R0, R13;
FMUL     R16, v[1], R14;
FMUL32   R14, R11, R2;
FMUL32   R15, R13, R6;
FMAD     R1, v[0], R1, R16;
FMAD     R14, R10, R13, -R14;
FMAD     R15, R2, R12, -R15;
FMAD     R0, v[2], R0, R1;
FMUL     o[25], v[16], R14;
FMUL     o[26], v[16], R15;
FMUL     R14, v[1], R5;
FMUL     R1, v[5], c[A1 + 171];
FMUL     R5, v[5], c[A1 + 175];
FMAD     R4, v[0], R4, R14;
FMAD     R1, v[4], c[A0 + 171], R1;
FMAD     R5, v[4], c[A0 + 175], R5;
FMAD     R3, v[2], R3, R4;
FMAD     R0, v[3], R1, R0;
FMUL     R4, v[1], R9;
FMAD     R1, v[3], R5, R3;
FMUL     R3, v[5], c[A1 + 179];
FMAD     R4, v[0], R8, R4;
FMUL32   R5, R1, c[41];
FMAD     R3, v[4], c[A0 + 179], R3;
FMAD     R8, v[2], R7, R4;
FMAD     R7, R0, c[40], R5;
MOV32    R4, c[43];
FMAD     R3, v[3], R3, R8;
MOV32    R5, c[67];
FMAD     R9, R3, c[42], R7;
FMUL32   R8, R2, R2;
FSET     R7, R2, c[0], LT;
FMAD     R4, R4, c[1], R9;
FMUL32   R9, R6, R6;
R2A      A1, R7;
FMAD     R7, -R4, R5, c[64];
FSET     R14, R6, c[0], LT;
FMUL32   R5, R10, R10;
MOV32    o[33], R7;
R2A      A0, R14;
MOV32    o[34], R7;
MOV32    o[35], R7;
MOV32    o[36], R7;
FMUL32   R14, R9, c[A0 + 92];
FSET     R7, R10, c[0], LT;
FMUL32   R15, R9, c[A0 + 93];
FMUL32   R9, R9, c[A0 + 94];
R2A      A0, R7;
FMAD     R7, R8, c[A1 + 84], R14;
FMAD     R14, R8, c[A1 + 85], R15;
FMAD     R8, R8, c[A1 + 86], R9;
FMAD     o[4], R5, c[A0 + 100], R7;
FMAD     o[5], R5, c[A0 + 101], R14;
FMAD     o[6], R5, c[A0 + 102], R8;
FMUL32   R7, R1, c[33];
MOV32    R5, c[35];
FMUL32   R8, R1, c[37];
FMAD     R7, R0, c[32], R7;
FMAD     R8, R0, c[36], R8;
FMAD     R7, R3, c[34], R7;
MOV32    o[2], R4;
FMAD     R4, R3, c[38], R8;
FMAD     o[0], R5, c[1], R7;
MOV32    R5, c[39];
FMUL32   R7, R1, c[45];
MOV32    o[27], R2;
MOV32    R2, R5;
FMAD     R5, R0, c[44], R7;
MOV32    o[28], R6;
FMAD     o[1], R2, c[1], R4;
FMAD     R2, R3, c[46], R5;
MOV32    o[29], R10;
MOV32    o[21], R13;
MOV32    o[22], R12;
MOV32    o[23], R11;
FADD32   o[18], -R0, c[8];
FADD32   o[19], -R1, c[9];
FADD32   o[20], -R3, c[10];
MOV32    R1, c[47];
MOV32    o[30], c[0];
MOV32    R0, c[0];
MOV32    R3, R1;
MOV32    R1, c[0];
MOV32    o[31], R0;
FMAD     o[3], R3, c[1], R2;
MOV32    o[32], R1;
FMUL     R0, v[11], c[360];
FMUL     R1, v[11], c[364];
FMAD     R0, v[12], c[361], R0;
FMAD     R1, v[12], c[365], R1;
MOV32    o[8], c[0];
MOV32    o[16], R0;
MOV32    o[17], R1;
MOV32    o[14], R0;
MOV32    o[12], R0;
MOV32    o[15], R1;
MOV32    o[13], R1;
MOV32    R0, c[0];
MOV32    R1, c[0];
MOV32    R2, c[0];
MOV32    o[9], R0;
MOV32    o[10], R1;
MOV32    o[11], R2;
MOV32    R0, c[0];
MOV32    o[7], R0;
END
# 147 instructions, 20 R-regs
# 147 inst, (43 mov, 0 mvi, 0 tex, 0 complex, 104 math)
#    93 64-bit, 54 32-bit, 0 32-bit-const
