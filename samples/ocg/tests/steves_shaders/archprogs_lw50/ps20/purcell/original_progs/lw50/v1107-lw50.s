!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     15
.MAX_IBUF    12
.MAX_OBUF    36
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v1107-lw40.s -o allprogs-new32//v1107-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[91].C[91]
#semantic C[90].C[90]
#semantic C[27].C[27]
#semantic C[28].C[28]
#semantic c.c
#semantic C[16].C[16]
#semantic C[2].C[2]
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
#var float4 C[27] :  : c[27] : -1 : 0
#var float4 C[28] :  : c[28] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
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
FMUL     R2, v[10], c[173];
FMUL     R1, v[5], c[177];
FMUL     R0, v[5], c[173];
FMAD     R2, v[9], c[172], R2;
FMAD     R1, v[4], c[176], R1;
FMAD     R0, v[4], c[172], R0;
FMAD     R4, v[11], c[174], R2;
FMAD     R2, v[6], c[178], R1;
FMAD     R1, v[6], c[174], R0;
FMUL     R3, v[10], c[177];
FMUL32   R6, R4, R2;
FMUL     R0, v[5], c[169];
FMAD     R3, v[9], c[176], R3;
FMUL     R5, v[10], c[169];
FMAD     R0, v[4], c[168], R0;
FMAD     R3, v[11], c[178], R3;
FMAD     R5, v[9], c[168], R5;
FMAD     R0, v[6], c[170], R0;
FMAD     R7, R1, R3, -R6;
FMAD     R5, v[11], c[170], R5;
FMUL32   R6, R3, R0;
FMUL     o[24], v[12], R7;
FMUL32   R7, R5, R1;
FMAD     R6, R2, R5, -R6;
FMUL32   R8, R1, -c[113];
FMAD     R7, R0, R4, -R7;
FMUL     o[25], v[12], R6;
FMAD     R6, R0, -c[112], R8;
FMUL     o[26], v[12], R7;
FMUL32   R9, R0, R0;
FMAD     R6, R2, -c[114], R6;
FSET     R7, R0, c[0], LT;
FMUL32   R10, R1, R1;
FMAX     R6, R6, c[0];
R2A      A1, R7;
FSET     R11, R1, c[0], LT;
FMUL32   R7, R2, R2;
FSET     R8, R2, c[0], LT;
R2A      A2, R11;
R2A      A0, R8;
FMUL32   R8, R10, c[A2 + 92];
FMUL32   R11, R10, c[A2 + 93];
FMUL32   R10, R10, c[A2 + 94];
FMAD     R8, R9, c[A1 + 84], R8;
FMAD     R11, R9, c[A1 + 85], R11;
FMAD     R9, R9, c[A1 + 86], R10;
FMAD     R8, R7, c[A0 + 100], R8;
FMAD     R10, R7, c[A0 + 101], R11;
FMAD     R7, R7, c[A0 + 102], R9;
FMAD     o[4], R6, c[108], R8;
FMAD     o[5], R6, c[109], R10;
FMAD     o[6], R6, c[110], R7;
FMUL     R6, v[1], c[169];
FMUL     R7, v[1], c[173];
FMAD     R6, v[0], c[168], R6;
FMAD     R7, v[0], c[172], R7;
FMUL     R8, v[1], c[177];
FMAD     R6, v[2], c[170], R6;
FMAD     R7, v[2], c[174], R7;
FMAD     R8, v[0], c[176], R8;
FMAD     R6, v[3], c[171], R6;
FMAD     R7, v[3], c[175], R7;
FMAD     R8, v[2], c[178], R8;
MOV32    R9, c[43];
FMUL32   R10, R7, c[41];
FMAD     R8, v[3], c[179], R8;
FMAD     R11, R6, c[40], R10;
MOV32    R10, c[67];
FMUL32   R12, R7, c[33];
FMAD     R11, R8, c[42], R11;
FMAD     R12, R6, c[32], R12;
FMAD     R9, R9, c[1], R11;
MOV32    R11, c[35];
FMAD     R12, R8, c[34], R12;
FMAD     R10, -R9, R10, c[64];
FMUL32   R13, R7, c[37];
MOV32    o[33], R10;
FMAD     o[0], R11, c[1], R12;
FMAD     R11, R6, c[36], R13;
MOV32    o[2], R9;
MOV32    o[34], R10;
FMAD     R9, R8, c[38], R11;
MOV32    o[35], R10;
MOV32    o[36], R10;
MOV32    R10, c[39];
FMUL32   R12, R7, c[45];
MOV32    R11, c[47];
FMAD     R12, R6, c[44], R12;
FMAD     o[1], R10, c[1], R9;
FMAD     R9, R8, c[46], R12;
MOV32    o[27], R0;
MOV32    o[28], R1;
FMAD     o[3], R11, c[1], R9;
MOV32    o[29], R2;
MOV32    o[21], R5;
MOV32    o[22], R4;
MOV32    o[23], R3;
FADD32   o[18], -R6, c[8];
FADD32   o[19], -R7, c[9];
FADD32   o[20], -R8, c[10];
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
# 124 instructions, 16 R-regs
# 124 inst, (39 mov, 0 mvi, 0 tex, 0 complex, 85 math)
#    71 64-bit, 53 32-bit, 0 32-bit-const
