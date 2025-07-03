!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     19
.MAX_IBUF    15
.MAX_OBUF    15
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v1139-lw40.s -o allprogs-new32//v1139-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[95].C[95]
#semantic C[92].C[92]
#semantic C[91].C[91]
#semantic C[90].C[90]
#semantic C[16].C[16]
#semantic C[11].C[11]
#semantic C[10].C[10]
#semantic C[9].C[9]
#semantic C[8].C[8]
#semantic C[0].C[0]
#semantic c.c
#semantic C[3].C[3]
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[FOGC] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 v[TEX9] : $vin.F : F[0] : -1 : 0
#var float4 C[95] :  : c[95] : -1 : 0
#var float4 C[92] :  : c[92] : -1 : 0
#var float4 C[91] :  : c[91] : -1 : 0
#var float4 C[90] :  : c[90] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 C[3] :  : c[3] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[WGT].x
#ibuf 5 = v[WGT].y
#ibuf 6 = v[NOR].x
#ibuf 7 = v[NOR].y
#ibuf 8 = v[NOR].z
#ibuf 9 = v[COL0].x
#ibuf 10 = v[COL0].y
#ibuf 11 = v[COL0].z
#ibuf 12 = v[UNUSED1].x
#ibuf 13 = v[UNUSED1].y
#ibuf 14 = v[UNUSED1].z
#ibuf 15 = v[UNUSED1].w
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[TEX0].x
#obuf 5 = o[TEX0].y
#obuf 6 = o[TEX0].z
#obuf 7 = o[TEX0].w
#obuf 8 = o[TEX1].x
#obuf 9 = o[TEX1].y
#obuf 10 = o[TEX1].z
#obuf 11 = o[TEX1].w
#obuf 12 = o[FOGC].x
#obuf 13 = o[FOGC].y
#obuf 14 = o[FOGC].z
#obuf 15 = o[FOGC].w
BB0:
MOV      R0, v[5];
MOV32    R1, c[14];
MOV32    R2, c[14];
FADD     R0, v[4], R0;
FMAD     R1, v[6], R1, c[15];
FMAD     R2, v[7], R2, c[15];
FADD32   R0, -R0, c[1];
F2I.FLOOR R1, R1;
F2I.FLOOR R2, R2;
MOV32    R3, c[14];
I2I.M4   R1, R1;
I2I.M4   R2, R2;
FMAD     R3, v[8], R3, c[15];
R2A      A0, R1;
R2A      A1, R2;
F2I.FLOOR R1, R3;
I2I.M4   R1, R1;
R2A      A2, R1;
FMUL     R3, v[4], c[A2];
FMUL     R2, v[4], c[A2 + 1];
FMUL     R1, v[4], c[A2 + 2];
FMAD     R3, v[5], c[A1], R3;
FMAD     R2, v[5], c[A1 + 1], R2;
FMAD     R1, v[5], c[A1 + 2], R1;
FMAD     R8, R0, c[A0], R3;
FMAD     R9, R0, c[A0 + 1], R2;
FMAD     R7, R0, c[A0 + 2], R1;
FMUL     R1, v[4], c[A2 + 3];
FMUL     R3, v[1], R9;
FMUL     R2, v[4], c[A2 + 4];
FMAD     R1, v[5], c[A1 + 3], R1;
FMAD     R3, v[0], R8, R3;
FMAD     R2, v[5], c[A1 + 4], R2;
FMAD     R1, R0, c[A0 + 3], R1;
FMAD     R3, v[2], R7, R3;
FMAD     R12, R0, c[A0 + 4], R2;
FMUL     R2, v[4], c[A2 + 5];
FMAD     R10, v[3], R1, R3;
FMUL     R1, v[4], c[A2 + 6];
FMAD     R3, v[5], c[A1 + 5], R2;
FMUL     R2, v[4], c[A2 + 7];
FMAD     R1, v[5], c[A1 + 6], R1;
FMAD     R13, R0, c[A0 + 5], R3;
FMAD     R2, v[5], c[A1 + 7], R2;
FMAD     R11, R0, c[A0 + 6], R1;
FMUL     R1, v[1], R13;
FMAD     R2, R0, c[A0 + 7], R2;
FMUL     R4, v[4], c[A2 + 8];
FMAD     R3, v[0], R12, R1;
FMUL     R1, v[4], c[A2 + 9];
FMAD     R4, v[5], c[A1 + 8], R4;
FMAD     R3, v[2], R11, R3;
FMAD     R1, v[5], c[A1 + 9], R1;
FMAD     R4, R0, c[A0 + 8], R4;
FMAD     R14, v[3], R2, R3;
FMAD     R5, R0, c[A0 + 9], R1;
FMUL     R2, v[4], c[A2 + 10];
FMUL     R1, v[4], c[A2 + 11];
FMUL32   R3, R14, c[361];
FMAD     R2, v[5], c[A1 + 10], R2;
FMAD     R1, v[5], c[A1 + 11], R1;
FMAD     R15, R10, c[360], R3;
FMAD     R3, R0, c[A0 + 10], R2;
FMAD     R2, R0, c[A0 + 11], R1;
FMUL     R0, v[1], R5;
MOV32    R1, c[363];
MVI      R6, 0.0;
FMAD     R0, v[0], R4, R0;
F2F      R16, R6;
FMAD     R6, v[2], R3, R0;
MVI      R0, 0.0;
FMUL32   R17, R16, R16;
FMAD     R2, v[3], R2, R6;
F2F      R0, R0;
FMUL     R6, v[10], R9;
FMAD     R9, R2, c[362], R15;
FMAD     R6, v[9], R8, R6;
FMAD     R1, R1, c[1], R9;
FMUL     R8, v[10], R13;
FMAD     R6, v[11], R7, R6;
F2F      R1, -R1;
FMAD     R7, v[9], R12, R8;
FMUL     R5, v[10], R5;
FMAD     R8, R1, R1, R17;
FMAD     R7, v[11], R11, R7;
FMAD     R4, v[9], R4, R5;
FMAD     R5, R0, R0, R8;
FMUL32   R8, R7, c[361];
FMAD     R3, v[11], R3, R4;
RSQ      R4, |R5|;
FMAD     R5, R6, c[360], R8;
FMUL32   R8, R7, c[365];
FMUL32   R1, R1, R4;
FMAD     R5, R3, c[362], R5;
FMAD     R8, R6, c[364], R8;
FMUL32   R7, R7, c[369];
FMAD     R8, R3, c[366], R8;
FMAD     R6, R6, c[368], R7;
FMUL32   R9, R16, R4;
FMUL32   R7, R8, R8;
FMAD     R3, R3, c[370], R6;
FMAD     R6, R5, R5, R7;
FMUL32   R0, R0, R4;
FMAD     R4, R3, R3, R6;
RSQ      R4, |R4|;
FMUL32   R5, R5, R4;
FMUL32   R6, R8, R4;
FMUL32   R3, R3, R4;
FMUL32   o[7], R4, R4;
FMUL32   R4, R9, R6;
MOV32    o[5], R6;
MOV32    o[6], R3;
FMAD     R1, R1, R5, R4;
FMUL32   R5, R14, c[41];
MOV32    R4, c[43];
FMAD     R0, R0, R3, R1;
FMAD     R1, R10, c[40], R5;
MOV32    R3, R4;
FADD32   o[4], R0, c[380];
FMAD     R0, R2, c[42], R1;
MOV32    R1, c[67];
FMUL32   R4, R14, c[33];
FMAD     R0, R3, c[1], R0;
MOV32    R3, c[35];
FMAD     R4, R10, c[32], R4;
FMAD     R1, -R0, R1, c[64];
FMAD     R4, R2, c[34], R4;
MOV32    o[12], R1;
MOV32    o[13], R1;
FMAD     o[0], R3, c[1], R4;
MOV32    o[14], R1;
MOV32    o[15], R1;
MOV32    o[2], R0;
FMUL32   R0, R14, c[37];
FMUL32   R3, R14, c[45];
MOV32    R1, c[39];
FMAD     R0, R10, c[36], R0;
FMAD     R3, R10, c[44], R3;
FMAD     R0, R2, c[38], R0;
FMAD     R3, R2, c[46], R3;
MOV32    R2, c[47];
FMAD     o[1], R1, c[1], R0;
MOV      o[8], v[12];
MOV32    R0, R2;
MOV      o[9], v[13];
MOV      o[10], v[14];
FMAD     o[3], R0, c[1], R3;
MOV      o[11], v[15];
END
# 148 instructions, 20 R-regs
# 148 inst, (23 mov, 2 mvi, 0 tex, 2 complex, 121 math)
#    110 64-bit, 38 32-bit, 0 32-bit-const
