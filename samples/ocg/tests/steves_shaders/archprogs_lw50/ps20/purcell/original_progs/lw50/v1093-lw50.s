!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     35
.MAX_IBUF    16
.MAX_OBUF    36
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v1093-lw40.s -o allprogs-new32//v1093-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[91].C[91]
#semantic C[90].C[90]
#semantic C[32].C[32]
#semantic C[36].C[36]
#semantic C[3].C[3]
#semantic C[27].C[27]
#semantic C[31].C[31]
#semantic C[34].C[34]
#semantic C[29].C[29]
#semantic C[2].C[2]
#semantic C[16].C[16]
#semantic C[11].C[11]
#semantic C[10].C[10]
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
#var float4 C[32] :  : c[32] : -1 : 0
#var float4 C[36] :  : c[36] : -1 : 0
#var float4 C[3] :  : c[3] : -1 : 0
#var float4 C[27] :  : c[27] : -1 : 0
#var float4 C[31] :  : c[31] : -1 : 0
#var float4 C[34] :  : c[34] : -1 : 0
#var float4 v[TEX8] : $vin.F : F[0] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 C[29] :  : c[29] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
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
FMUL     R2, v[5], c[A1 + 176];
FMUL     R1, v[5], c[A1 + 177];
FMUL     R0, v[5], c[A1 + 178];
FMAD     R2, v[4], c[A0 + 176], R2;
FMAD     R3, v[4], c[A0 + 177], R1;
FMAD     R0, v[4], c[A0 + 178], R0;
FMUL     R5, v[5], c[A1 + 172];
FMUL     R4, v[9], R3;
FMUL     R1, v[5], c[A1 + 173];
FMAD     R15, v[4], c[A0 + 172], R5;
FMAD     R4, v[8], R2, R4;
FMAD     R16, v[4], c[A0 + 173], R1;
FMUL     R1, v[5], c[A1 + 174];
FMAD     R4, v[10], R0, R4;
FMUL     R5, v[14], R16;
FMAD     R11, v[4], c[A0 + 174], R1;
FMUL     R1, v[9], R16;
FMAD     R6, v[13], R15, R5;
FMUL     R5, v[14], R3;
FMAD     R1, v[8], R15, R1;
FMAD     R10, v[15], R11, R6;
FMAD     R5, v[13], R2, R5;
FMAD     R6, v[10], R11, R1;
FMUL32   R1, R4, R10;
FMAD     R8, v[15], R0, R5;
FMUL     R7, v[5], c[A1 + 168];
FMUL     R5, v[5], c[A1 + 169];
FMAD     R1, R6, R8, -R1;
FMAD     R7, v[4], c[A0 + 168], R7;
FMAD     R17, v[4], c[A0 + 169], R5;
FMUL     R13, v[16], R1;
FMUL     R1, v[5], c[A1 + 170];
FMUL     R5, v[9], R17;
FMUL     R9, v[14], R17;
FMAD     R1, v[4], c[A0 + 170], R1;
FMAD     R5, v[8], R7, R5;
FMAD     R9, v[13], R7, R9;
FMAD     R5, v[10], R1, R5;
FMAD     R9, v[15], R1, R9;
FMUL32   R14, R5, R8;
FMUL32   R12, R6, R9;
FMUL     R17, v[1], R17;
FMAD     R14, R4, R9, -R14;
FMAD     R12, R5, R10, -R12;
FMAD     R7, v[0], R7, R17;
FMUL     R14, v[16], R14;
FMUL     R12, v[16], R12;
FMAD     R1, v[2], R1, R7;
FMUL     R17, v[1], R16;
FMUL32   R16, R14, c[7];
FMUL     R7, v[5], c[A1 + 171];
FMAD     R17, v[0], R15, R17;
FMAD     R16, R13, c[7], R16;
FMAD     R15, v[4], c[A0 + 171], R7;
FMAD     R7, v[2], R11, R17;
FMAD     R27, R12, c[7], R16;
FMAD     R1, v[3], R15, R1;
FMUL     R15, v[1], R3;
FMUL     R3, v[5], c[A1 + 175];
FADD32   R11, -R1, c[116];
FMAD     R15, v[0], R2, R15;
FMAD     R2, v[4], c[A0 + 175], R3;
FMUL     R3, v[5], c[A1 + 179];
FMAD     R0, v[2], R0, R15;
FMAD     R2, v[3], R2, R7;
FMAD     R3, v[4], c[A0 + 179], R3;
FADD32   R15, -R2, c[117];
FMAD     R0, v[3], R3, R0;
FMUL32   R16, R10, c[7];
FMUL32   R7, R15, R15;
FADD32   R3, -R0, c[118];
FMAD     R17, R9, c[7], R16;
FMAD     R7, R11, R11, R7;
FMUL32   R16, R6, c[7];
FMAD     R30, R8, c[7], R17;
FMAD     R7, R3, R3, R7;
FMAD     R16, R5, c[7], R16;
RSQ      R20, |R7|;
FMAD     R29, R4, c[7], R16;
FMUL32   R25, R27, R27;
FMUL32   R19, R15, R20;
FMUL32   R18, R11, R20;
FMUL32   R17, R3, R20;
FMUL32   R3, R27, R19;
FSET     R11, R27, c[0], LT;
FMUL32   R23, R30, R30;
FMAD     R3, R30, R18, R3;
R2A      A2, R11;
FSET     R15, R30, c[0], LT;
FMAD     R3, R29, R17, R3;
FMUL32   R11, R25, c[A2 + 92];
R2A      A1, R15;
FMAX     R22, R3, c[0];
FMUL32   R3, R29, R29;
FMAD     R15, R23, c[A1 + 84], R11;
FSET     R11, R29, c[0], LT;
FMUL32   R21, R7, R20;
MOV32    R16, c[125];
R2A      A0, R11;
FADD32   R11, -R1, c[136];
FMAD     R16, R16, R21, c[124];
FMAD     R24, R3, c[A0 + 100], R15;
FADD32   R15, -R2, c[137];
FMAD     R21, R7, c[126], R16;
FADD32   R7, -R0, c[138];
FMUL32   R16, R15, R15;
FMAD     R20, R20, c[127], R21;
FMAD     R16, R11, R11, R16;
RCP      R20, R20;
FMAD     R21, R7, R7, R16;
FMUL32   R16, R20, c[108];
RSQ      R26, |R21|;
FMAD     R24, R16, R22, R24;
FMUL32   R15, R15, R26;
FMUL32   R11, R11, R26;
FMUL32   R7, R7, R26;
FMUL32   R31, R27, R15;
FMUL32   R28, R21, R26;
MOV32    R27, c[145];
FMAD     R30, R30, R11, R31;
FMAD     R27, R27, R28, c[144];
FMAD     R29, R29, R7, R30;
FMUL32   R28, R25, c[A2 + 93];
FMAD     R21, R21, c[146], R27;
FMAX     R27, R29, c[0];
FMAD     R28, R23, c[A1 + 85], R28;
FMAD     R26, R26, c[147], R21;
FMUL32   R21, R20, c[109];
FMUL32   R29, R25, c[A2 + 94];
RCP      R25, R26;
FMAD     R26, R3, c[A0 + 101], R28;
FMAD     R28, R23, c[A1 + 86], R29;
FMUL32   R23, R20, c[110];
FMAD     R26, R21, R22, R26;
FMAD     R28, R3, c[A0 + 102], R28;
FMUL32   R3, R25, c[128];
FMUL32   R20, R25, c[129];
FMAD     R28, R23, R22, R28;
FMAD     o[30], R3, R27, R24;
FMAD     o[31], R20, R27, R26;
FMUL32   R22, R25, c[130];
FMUL32   R25, R14, c[12];
FMUL32   R24, R10, c[12];
FMAD     o[32], R22, R27, R28;
FMAD     R27, R12, c[13], R25;
FMAD     R25, R8, c[13], R24;
FMUL32   R24, R6, c[12];
FMUL32   R26, R27, R19;
FMUL32   R30, R27, R27;
FMAD     R24, R4, c[13], R24;
FMAD     R26, R25, R18, R26;
FSET     R28, R27, c[0], LT;
FMUL32   R29, R25, R25;
FMAD     R26, R24, R17, R26;
R2A      A2, R28;
FSET     R28, R25, c[0], LT;
FMAX     R26, R26, c[0];
FMUL32   R31, R30, c[A2 + 92];
R2A      A1, R28;
FMUL32   R28, R27, R15;
FMUL32   R27, R24, R24;
FMAD     R31, R29, c[A1 + 84], R31;
FMAD     R25, R25, R11, R28;
FSET     R28, R24, c[0], LT;
FMUL32   R32, R30, c[A2 + 93];
FMAD     R24, R24, R7, R25;
R2A      A0, R28;
FMAD     R25, R29, c[A1 + 85], R32;
FMUL32   R28, R30, c[A2 + 94];
FMAD     R30, R27, c[A0 + 100], R31;
FMAX     R24, R24, c[0];
FMAD     R28, R29, c[A1 + 86], R28;
FMAD     R29, R16, R26, R30;
FMAD     R25, R27, c[A0 + 101], R25;
FMAD     R27, R27, c[A0 + 102], R28;
FMAD     o[8], R3, R24, R29;
FMAD     R25, R21, R26, R25;
FMAD     R26, R23, R26, R27;
FMUL32   R27, R14, c[6];
FMAD     o[9], R20, R24, R25;
FMAD     o[10], R22, R24, R26;
FMAD     R24, R13, c[5], R27;
FMUL32   R25, R10, c[6];
FMUL32   R26, R6, c[6];
FMAD     R24, R12, c[6], R24;
FMAD     R25, R9, c[5], R25;
FMAD     R26, R5, c[5], R26;
FMUL32   R27, R24, R19;
FMAD     R25, R8, c[6], R25;
FMAD     R19, R4, c[6], R26;
FMUL32   R26, R24, R24;
FMAD     R18, R25, R18, R27;
FSET     R28, R24, c[0], LT;
FMUL32   R27, R25, R25;
FMAD     R17, R19, R17, R18;
R2A      A2, R28;
FSET     R18, R25, c[0], LT;
FMAX     R17, R17, c[0];
FMUL32   R28, R26, c[A2 + 92];
FMUL32   R15, R24, R15;
R2A      A1, R18;
FMUL32   R18, R19, R19;
FMAD     R11, R25, R11, R15;
FMAD     R24, R27, c[A1 + 84], R28;
FSET     R15, R19, c[0], LT;
FMAD     R7, R19, R7, R11;
FMUL32   R11, R26, c[A2 + 93];
R2A      A0, R15;
FMAX     R7, R7, c[0];
FMUL32   R15, R26, c[A2 + 94];
FMAD     R19, R18, c[A0 + 100], R24;
FMAD     R11, R27, c[A1 + 85], R11;
FMAD     R15, R27, c[A1 + 86], R15;
FMAD     R16, R16, R17, R19;
FMAD     R11, R18, c[A0 + 101], R11;
FMAD     R15, R18, c[A0 + 102], R15;
FMAD     o[4], R3, R7, R16;
FMAD     R3, R21, R17, R11;
FMAD     R11, R23, R17, R15;
MOV32    o[28], R12;
FMAD     o[5], R20, R7, R3;
FMAD     o[6], R22, R7, R11;
MOV32    o[25], R14;
MOV32    o[22], R13;
FMUL32   R11, R2, c[41];
MOV32    R3, c[43];
MOV32    R7, c[67];
FMAD     R11, R1, c[40], R11;
FMUL32   R12, R2, c[33];
FMAD     R11, R0, c[42], R11;
FMAD     R12, R1, c[32], R12;
FMAD     R3, R3, c[1], R11;
MOV32    R11, c[35];
FMAD     R12, R0, c[34], R12;
FMAD     R7, -R3, R7, c[64];
FMUL32   R13, R2, c[37];
MOV32    o[33], R7;
FMAD     o[0], R11, c[1], R12;
FMAD     R11, R1, c[36], R13;
MOV32    o[2], R3;
MOV32    o[34], R7;
FMAD     R3, R0, c[38], R11;
MOV32    o[35], R7;
MOV32    o[36], R7;
MOV32    R7, c[39];
FMUL32   R11, R2, c[45];
MOV32    o[27], R8;
FMAD     R8, R1, c[44], R11;
MOV32    o[29], R4;
FMAD     o[1], R7, c[1], R3;
FMAD     R3, R0, c[46], R8;
MOV32    o[24], R10;
MOV32    o[26], R6;
MOV32    o[21], R9;
MOV32    o[23], R5;
FADD32   o[18], -R1, c[8];
FADD32   o[19], -R2, c[9];
FADD32   o[20], -R0, c[10];
MOV32    R2, c[47];
FMUL     R0, v[11], c[360];
FMUL     R1, v[11], c[364];
FMAD     R0, v[12], c[361], R0;
FMAD     R1, v[12], c[365], R1;
FMAD     o[3], R2, c[1], R3;
MOV32    o[16], R0;
MOV32    o[17], R1;
MOV32    o[14], R0;
MOV32    o[12], R0;
MOV32    o[15], R1;
MOV32    o[13], R1;
MOV32    R0, c[0];
MOV32    R1, c[0];
MOV32    o[11], R0;
MOV32    o[7], R1;
END
# 278 instructions, 36 R-regs
# 278 inst, (42 mov, 0 mvi, 0 tex, 4 complex, 232 math)
#    182 64-bit, 96 32-bit, 0 32-bit-const
