!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     19
.MAX_IBUF    14
.MAX_OBUF    27
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v1061-lw40.s -o allprogs-new32//v1061-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[92].C[92]
#semantic C[91].C[91]
#semantic C[16].C[16]
#semantic C[2].C[2]
#semantic C[11].C[11]
#semantic C[10].C[10]
#semantic C[9].C[9]
#semantic C[0].C[0]
#semantic C[8].C[8]
#semantic C[44].C[44]
#semantic C[43].C[43]
#semantic C[42].C[42]
#var float4 o[TEX6] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX5] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX4] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX3] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[FOGC] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 C[92] :  : c[92] : -1 : 0
#var float4 C[91] :  : c[91] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 C[44] :  : c[44] : -1 : 0
#var float4 C[43] :  : c[43] : -1 : 0
#var float4 C[42] :  : c[42] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[WGT].x
#ibuf 5 = v[WGT].y
#ibuf 6 = v[WGT].z
#ibuf 7 = v[NOR].x
#ibuf 8 = v[NOR].y
#ibuf 9 = v[NOR].z
#ibuf 10 = v[NOR].w
#ibuf 11 = v[COL0].x
#ibuf 12 = v[COL0].y
#ibuf 13 = v[COL0].z
#ibuf 14 = v[COL0].w
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[TEX0].x
#obuf 5 = o[TEX0].y
#obuf 6 = o[TEX1].x
#obuf 7 = o[TEX1].y
#obuf 8 = o[TEX1].z
#obuf 9 = o[TEX2].x
#obuf 10 = o[TEX2].y
#obuf 11 = o[TEX2].z
#obuf 12 = o[TEX3].x
#obuf 13 = o[TEX3].y
#obuf 14 = o[TEX3].z
#obuf 15 = o[TEX4].x
#obuf 16 = o[TEX4].y
#obuf 17 = o[TEX4].z
#obuf 18 = o[TEX5].x
#obuf 19 = o[TEX5].y
#obuf 20 = o[TEX5].z
#obuf 21 = o[TEX6].x
#obuf 22 = o[TEX6].y
#obuf 23 = o[TEX6].z
#obuf 24 = o[FOGC].x
#obuf 25 = o[FOGC].y
#obuf 26 = o[FOGC].z
#obuf 27 = o[FOGC].w
BB0:
FMUL     R0, v[5], c[177];
FMUL     R2, v[12], c[173];
FMUL     R1, v[5], c[173];
FMAD     R0, v[4], c[176], R0;
FMAD     R2, v[11], c[172], R2;
FMAD     R1, v[4], c[172], R1;
FMAD     R0, v[6], c[178], R0;
FMAD     R12, v[13], c[174], R2;
FMAD     R2, v[6], c[174], R1;
FMUL     R4, v[12], c[177];
FMUL32   R3, R0, R12;
FMUL     R1, v[5], c[169];
FMAD     R5, v[11], c[176], R4;
FMUL     R4, v[12], c[169];
FMAD     R1, v[4], c[168], R1;
FMAD     R11, v[13], c[178], R5;
FMAD     R4, v[11], c[168], R4;
FMAD     R1, v[6], c[170], R1;
FMAD     R3, R2, R11, -R3;
FMAD     R13, v[13], c[170], R4;
FMUL32   R4, R1, R11;
FMUL     R16, v[14], R3;
FMUL     R3, v[1], c[173];
FMAD     R5, R0, R13, -R4;
FMUL     R4, v[1], c[169];
FMAD     R3, v[0], c[172], R3;
FMUL     R15, v[14], R5;
FMAD     R4, v[0], c[168], R4;
FMAD     R5, v[2], c[174], R3;
FMUL32   R3, R2, R13;
FMAD     R4, v[2], c[170], R4;
FMAD     R5, v[3], c[175], R5;
FMAD     R3, R1, R12, -R3;
FMAD     R4, v[3], c[171], R4;
FADD32   R19, -R5, c[9];
FMUL     R14, v[14], R3;
FADD32   R18, -R4, c[8];
FMUL32   R7, R19, R15;
FMUL     R3, v[1], c[177];
FMUL32   R6, R5, c[33];
FMAD     R8, R18, R16, R7;
FMAD     R3, v[0], c[176], R3;
FMAD     R10, R4, c[32], R6;
MOV32    R6, c[35];
FMAD     R3, v[2], c[178], R3;
FMUL32   R7, R5, c[45];
FMAD     R3, v[3], c[179], R3;
FMAD     R9, R4, c[44], R7;
MOV32    R7, c[47];
FMAD     R10, R3, c[34], R10;
FMAD     R9, R3, c[46], R9;
FMAD     R6, R6, c[1], R10;
FADD32   R17, -R3, c[10];
FMAD     R7, R7, c[1], R9;
FMUL32   R9, R5, c[37];
FMAD     o[7], R17, R14, R8;
FADD32   R8, R6, R7;
FMAD     R10, R4, c[36], R9;
MOV32    R9, c[39];
FMUL32   o[18], R8, c[3];
FMAD     R8, R3, c[38], R10;
MOV32    o[16], R14;
MOV32    o[13], R15;
FMAD     R8, R9, c[1], R8;
MOV32    o[10], R16;
FMUL32   R9, R19, R12;
FMUL32   R14, R19, R2;
F2F      R10, -R8;
FMAD     R9, R18, R13, R9;
FMAD     R14, R18, R1, R14;
FADD32   R10, R10, R7;
FMAD     o[6], R17, R11, R9;
FMAD     o[8], R17, R0, R14;
FMUL32   o[19], R10, c[3];
FMUL32   R9, R2, c[33];
FMUL32   R10, R2, c[37];
FMUL32   R14, R2, c[41];
FMAD     R9, R1, c[32], R9;
FMAD     R10, R1, c[36], R10;
FMAD     R14, R1, c[40], R14;
FMAD     o[21], R0, c[34], R9;
FMAD     o[22], R0, c[38], R10;
FMAD     o[23], R0, c[42], R14;
FMUL32   R9, R5, c[41];
MOV32    o[20], R7;
MOV32    R5, c[43];
FMAD     R4, R4, c[40], R9;
MOV32    R9, c[67];
FMAD     R3, R3, c[42], R4;
MOV32    o[0], R6;
MOV32    o[1], R8;
FMAD     R3, R5, c[1], R3;
MOV32    o[3], R7;
MOV32    o[15], R11;
FMAD     R4, -R3, R9, c[64];
MOV32    o[2], R3;
MOV32    o[17], R0;
MOV32    o[12], R12;
MOV32    o[14], R2;
MOV32    o[9], R13;
MOV32    o[11], R1;
MOV32    o[24], R4;
MOV32    o[25], R4;
MOV32    o[26], R4;
MOV32    o[27], R4;
FMUL     R0, v[8], c[365];
FMUL     R1, v[8], c[369];
FMAD     R0, v[7], c[364], R0;
FMAD     R1, v[7], c[368], R1;
FMAD     R0, v[9], c[366], R0;
FMAD     R1, v[9], c[370], R1;
FMAD     o[4], v[10], c[367], R0;
FMAD     o[5], v[10], c[371], R1;
END
# 113 instructions, 20 R-regs
# 113 inst, (23 mov, 0 mvi, 0 tex, 0 complex, 90 math)
#    70 64-bit, 43 32-bit, 0 32-bit-const
