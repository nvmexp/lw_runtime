!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     15
.MAX_IBUF    12
.MAX_OBUF    36
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v1114-lw40.s -o allprogs-new32//v1114-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[91].C[91]
#semantic C[90].C[90]
#semantic C[32].C[32]
#semantic C[33].C[33]
#semantic c.c
#semantic C[27].C[27]
#semantic C[30].C[30]
#semantic C[28].C[28]
#semantic C[31].C[31]
#semantic C[29].C[29]
#semantic C[2].C[2]
#semantic C[16].C[16]
#semantic C[10].C[10]
#semantic C[11].C[11]
#semantic C[9].C[9]
#semantic C[8].C[8]
#semantic C[0].C[0]
#semantic C[44].C[44]
#semantic C[43].C[43]
#semantic C[42].C[42]
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
#var float4 C[32] :  : c[32] : -1 : 0
#var float4 C[33] :  : c[33] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 C[27] :  : c[27] : -1 : 0
#var float4 C[30] :  : c[30] : -1 : 0
#var float4 C[28] :  : c[28] : -1 : 0
#var float4 C[31] :  : c[31] : -1 : 0
#var float4 C[29] :  : c[29] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 C[44] :  : c[44] : -1 : 0
#var float4 C[43] :  : c[43] : -1 : 0
#var float4 C[42] :  : c[42] : -1 : 0
#var float4 v[COL1] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[WGT].x
#ibuf 5 = v[WGT].y
#ibuf 6 = v[WGT].z
#ibuf 7 = v[NOR].x
#ibuf 8 = v[NOR].y
#ibuf 9 = v[COL1].x
#ibuf 10 = v[COL1].y
#ibuf 11 = v[COL1].z
#ibuf 12 = v[COL1].w
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
FMUL     R1, v[1], c[169];
FMUL     R0, v[1], c[173];
FMAD     R1, v[0], c[168], R1;
FMAD     R0, v[0], c[172], R0;
FMAD     R2, v[2], c[170], R1;
FMAD     R1, v[2], c[174], R0;
FMUL     R0, v[1], c[177];
FMAD     R3, v[3], c[171], R2;
FMAD     R4, v[3], c[175], R1;
FMAD     R0, v[0], c[176], R0;
FADD32   R8, -R3, c[116];
FADD32   R6, -R4, c[117];
FMAD     R1, v[2], c[178], R0;
FMUL32   R0, R6, R6;
FMAD     R5, v[3], c[179], R1;
MOV32    R2, c[1];
FMAD     R1, R8, R8, R0;
FADD32   R0, -R5, c[118];
FMUL32   R7, R2, c[1];
FMAD     R9, R0, R0, R1;
MOV32    R1, c[1];
RSQ      R2, |R9|;
MOV32    R11, R1;
FMUL32   R1, R9, c[1];
FMUL32   R10, R9, R2;
FMAD     R11, R9, -c[127], R11;
FMUL32   R9, R10, c[125];
FMAX     R11, R11, c[0];
FMUL32   R10, R8, R2;
FMAD     R7, R7, c[124], R9;
FMIN     R9, R11, c[1];
FMUL32   R8, R6, R2;
FMAD     R1, R1, c[126], R7;
FMUL32   R6, R0, R2;
FMUL32   R0, -R8, c[113];
RCP      R1, R1;
FMAD     R0, -R10, c[112], R0;
FMUL32   R7, R1, R9;
FMAD     R0, -R6, c[114], R0;
FMUL32   R9, R7, c[108];
FADD32   R0, R0, -c[122];
FMUL32   R1, R0, c[123];
FMUL     R0, v[5], c[173];
FMAX     R1, R1, c[0];
FMAD     R0, v[4], c[172], R0;
LG2      R2, |R1|;
FMAD     R1, v[6], c[174], R0;
FMUL     R0, v[5], c[169];
FMUL32   R2, R2, c[120];
FMUL32   R11, R1, R8;
FMAD     R0, v[4], c[168], R0;
RRO      R8, R2, 1;
FMUL     R2, v[5], c[177];
FMAD     R0, v[6], c[170], R0;
EX2      R8, R8;
FMAD     R2, v[4], c[176], R2;
FMAD     R10, R0, R10, R11;
FMIN     R8, R8, c[1];
FMAD     R2, v[6], c[178], R2;
FMUL32   R12, R0, R0;
FMUL32   R9, R9, R8;
FMAD     R10, R2, R6, R10;
FSET     R6, R0, c[0], LT;
FMUL32   R13, R1, R1;
FMAX     R10, R10, c[0];
R2A      A1, R6;
FSET     R6, R1, c[0], LT;
FMUL32   R11, R2, R2;
FSET     R14, R2, c[0], LT;
R2A      A2, R6;
FMUL32   R6, R1, -c[133];
R2A      A0, R14;
FMUL32   R14, R13, c[A2 + 92];
FMAD     R6, R0, -c[132], R6;
FMUL32   R15, R7, c[109];
FMAD     R14, R12, c[A1 + 84], R14;
FMAD     R6, R2, -c[134], R6;
FMUL32   R15, R15, R8;
FMAD     R14, R11, c[A0 + 100], R14;
FMAX     R6, R6, c[0];
FMUL32   R7, R7, c[110];
FMAD     R9, R9, R10, R14;
FMUL32   R14, R13, c[A2 + 93];
FMUL32   R7, R7, R8;
FMUL32   R8, R13, c[A2 + 94];
FMAD     o[4], R6, c[128], R9;
FMAD     R9, R12, c[A1 + 85], R14;
FMAD     R12, R12, c[A1 + 86], R8;
FMUL     R8, v[10], c[173];
FMAD     R9, R11, c[A0 + 101], R9;
FMAD     R11, R11, c[A0 + 102], R12;
FMAD     R8, v[9], c[172], R8;
FMAD     R9, R15, R10, R9;
FMAD     R10, R7, R10, R11;
FMAD     R7, v[11], c[174], R8;
FMAD     o[5], R6, c[129], R9;
FMAD     o[6], R6, c[130], R10;
FMUL32   R9, R7, R2;
FMUL     R6, v[10], c[177];
FMUL     R8, v[10], c[169];
FMAD     R6, v[9], c[176], R6;
FMAD     R8, v[9], c[168], R8;
FMAD     R6, v[11], c[178], R6;
FMAD     R8, v[11], c[170], R8;
FMAD     R9, R1, R6, -R9;
FMUL32   R10, R6, R0;
FMUL32   R11, R8, R1;
FMUL     o[24], v[12], R9;
FMAD     R9, R2, R8, -R10;
FMAD     R10, R0, R7, -R11;
FMUL32   R11, R4, c[41];
FMUL     o[25], v[12], R9;
FMUL     o[26], v[12], R10;
FMAD     R10, R3, c[40], R11;
MOV32    R9, c[43];
MOV32    R11, c[67];
FMAD     R10, R5, c[42], R10;
FMUL32   R12, R4, c[33];
FMAD     R9, R9, c[1], R10;
FMAD     R12, R3, c[32], R12;
MOV32    R10, c[35];
FMAD     R11, -R9, R11, c[64];
FMAD     R12, R5, c[34], R12;
MOV32    o[33], R11;
MOV32    o[34], R11;
FMAD     o[0], R10, c[1], R12;
MOV32    o[35], R11;
MOV32    o[36], R11;
MOV32    o[2], R9;
FMUL32   R10, R4, c[37];
MOV32    R9, c[39];
FMUL32   R11, R4, c[45];
FMAD     R10, R3, c[36], R10;
FMAD     R12, R3, c[44], R11;
FMAD     R10, R5, c[38], R10;
MOV32    R11, c[47];
FMAD     R12, R5, c[46], R12;
FMAD     o[1], R9, c[1], R10;
MOV32    R9, R11;
MOV32    o[27], R0;
MOV32    o[28], R1;
FMAD     o[3], R9, c[1], R12;
MOV32    o[29], R2;
MOV32    o[21], R8;
MOV32    o[22], R7;
MOV32    o[23], R6;
FADD32   o[18], -R3, c[8];
FADD32   o[19], -R4, c[9];
FADD32   o[20], -R5, c[10];
MOV32    o[30], c[0];
MOV32    R0, c[0];
MOV32    R1, c[0];
FMUL     R2, v[7], c[360];
MOV32    o[31], R0;
MOV32    o[32], R1;
FMAD     R0, v[8], c[361], R2;
FMUL     R1, v[7], c[364];
MOV32    o[8], c[0];
MOV32    o[16], R0;
FMAD     R1, v[8], c[365], R1;
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
MOV32    R0, c[0];
MOV32    o[7], R0;
END
# 173 instructions, 16 R-regs
# 173 inst, (43 mov, 0 mvi, 0 tex, 4 complex, 126 math)
#    93 64-bit, 80 32-bit, 0 32-bit-const
