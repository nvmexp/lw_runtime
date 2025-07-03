!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     7
.MAX_IBUF    11
.MAX_OBUF    22
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v329-lw40.s -o allprogs-new32//v329-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[13].C[13]
#semantic C[12].C[12]
#semantic C[7].C[7]
#semantic C[6].C[6]
#semantic C[5].C[5]
#semantic C[4].C[4]
#semantic C[11].C[11]
#semantic C[3].C[3]
#semantic C[2].C[2]
#semantic C[1].C[1]
#semantic C[0].C[0]
#semantic C[9].C[9]
#semantic c.c
#semantic C[8].C[8]
#semantic C[10].C[10]
#var float4 o[TEX4] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX3] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 C[13] :  : c[13] : -1 : 0
#var float4 C[12] :  : c[12] : -1 : 0
#var float4 C[7] :  : c[7] : -1 : 0
#var float4 C[6] :  : c[6] : -1 : 0
#var float4 C[5] :  : c[5] : -1 : 0
#var float4 C[4] :  : c[4] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 C[3] :  : c[3] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[WGT].x
#ibuf 5 = v[WGT].y
#ibuf 6 = v[WGT].z
#ibuf 7 = v[WGT].w
#ibuf 8 = v[NOR].x
#ibuf 9 = v[NOR].y
#ibuf 10 = v[NOR].z
#ibuf 11 = v[NOR].w
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
#obuf 12 = o[TEX2].x
#obuf 13 = o[TEX2].y
#obuf 14 = o[TEX2].z
#obuf 15 = o[TEX3].x
#obuf 16 = o[TEX3].y
#obuf 17 = o[TEX3].z
#obuf 18 = o[TEX3].w
#obuf 19 = o[TEX4].x
#obuf 20 = o[TEX4].y
#obuf 21 = o[TEX4].z
#obuf 22 = o[TEX4].w
BB0:
FADD     R0, v[0], c[40];
MOV32    R3, -c[38];
FRC      R2, R0;
FADD     R1, v[3], R0;
FMAD     R4, R2, R3, c[39];
FMUL32   R3, R1, c[32];
FADD     R1, v[3], c[41];
FMUL32   R5, R4, R2;
FRC      R4, R3;
FADD     R3, v[3], R1;
FMUL32   R2, R5, R2;
FMUL32   R4, R4, c[33];
F2I.FLOOR R4, R4;
I2I.M4   R4, R4;
R2A      A0, R4;
FMUL32   R4, R0, c[32];
FADD32   R3, c[A0 + 67], R3;
FRC      R4, R4;
FMUL32   R3, R3, c[32];
FMUL32   R4, R4, c[33];
FRC      R3, R3;
F2I.FLOOR R4, R4;
FADD     R5, v[3], R1;
FMUL32   R3, R3, c[33];
I2I.M4   R4, R4;
F2I.FLOOR R3, R3;
R2A      A1, R4;
I2I.M4   R3, R3;
FADD32   R4, c[A1 + 67], R5;
R2A      A1, R3;
FMUL32   R3, R4, c[32];
FRC      R4, R3;
FADD32   R3, c[A0 + 67], R1;
FMUL32   R4, R4, c[33];
FMUL32   R3, R3, c[32];
F2I.FLOOR R4, R4;
FRC      R3, R3;
I2I.M4   R4, R4;
FMUL32   R5, R3, c[33];
FMUL32   R3, R0, c[32];
R2A      A0, R4;
F2I.FLOOR R4, R5;
FRC      R5, R3;
MOV32    R3, c[A0 + 64];
I2I.M4   R4, R4;
FMUL32   R5, R5, c[33];
FADD32   R3, c[A1 + 64], -R3;
R2A      A1, R4;
F2I.FLOOR R4, R5;
FMAD     R3, R3, R2, c[A0 + 64];
I2I.M4   R4, R4;
R2A      A0, R4;
FADD32   R1, c[A0 + 67], R1;
FMUL32   R1, R1, c[32];
FRC      R1, R1;
FMUL32   R1, R1, c[33];
FRC      R0, R0;
MOV32    R4, -c[38];
F2I.FLOOR R1, R1;
FMAD     R4, R0, R4, c[39];
I2I.M4   R1, R1;
FMUL32   R4, R4, R0;
R2A      A0, R1;
MOV      R6, v[1];
FMUL32   R5, R4, R0;
MOV32    R1, c[A0 + 64];
FADD     R0, -v[0], c[48];
FADD32   R4, c[A1 + 64], -R1;
FADD     R1, -v[2], c[50];
FMAD     R2, R4, R2, c[A0 + 64];
FADD32   R3, R3, -R2;
FMAD     R2, R3, R5, R2;
FMAD     R2, R2, c[43], R6;
FADD32   R3, -R2, c[49];
FMUL32   R5, R2, c[17];
FMUL32   R4, R3, R3;
FMAD     R5, v[0], c[16], R5;
FMAD     R4, R0, R0, R4;
FMAD     R5, v[2], c[18], R5;
FMUL32   R6, R2, c[21];
FMAD     R4, R1, R1, R4;
FMAD     o[8], v[3], c[19], R5;
FMAD     R5, v[0], c[20], R6;
RSQ      R4, |R4|;
FMUL32   R6, R2, c[25];
FMAD     R5, v[2], c[22], R5;
FMUL32   o[13], R3, R4;
FMUL32   o[12], R0, R4;
FMUL32   o[14], R1, R4;
FMAD     o[9], v[3], c[23], R5;
FMAD     R0, v[0], c[24], R6;
FMUL32   R1, R2, c[29];
FMUL32   R3, R2, c[1];
FMAD     R0, v[2], c[26], R0;
FMAD     R1, v[0], c[28], R1;
FMAD     R3, v[0], c[0], R3;
FMAD     o[10], v[3], c[27], R0;
FMAD     R0, v[2], c[30], R1;
FMAD     R1, v[2], c[2], R3;
FMUL32   R3, R2, c[5];
FMAD     o[11], v[3], c[31], R0;
FMAD     o[0], v[3], c[3], R1;
FMAD     R0, v[0], c[4], R3;
FMUL32   R1, R2, c[9];
FMUL32   R2, R2, c[13];
FMAD     R0, v[2], c[6], R0;
FMAD     R1, v[0], c[8], R1;
FMAD     R2, v[0], c[12], R2;
FMAD     o[1], v[3], c[7], R0;
FMAD     R0, v[2], c[10], R1;
FMAD     R1, v[2], c[14], R2;
MOV      o[19], v[8];
FMAD     o[2], v[3], c[11], R0;
FMAD     o[3], v[3], c[15], R1;
MOV      o[20], v[9];
MOV      o[21], v[10];
MOV      o[22], v[11];
FADD     o[15], v[4], c[52];
FADD     o[16], v[5], c[53];
FADD     o[17], v[6], c[54];
FADD     o[18], v[7], c[55];
FADD     o[4], v[4], c[44];
FADD     o[5], v[5], c[45];
FADD     o[6], v[6], c[46];
FADD     o[7], v[7], c[47];
END
# 125 instructions, 8 R-regs
# 125 inst, (9 mov, 0 mvi, 0 tex, 1 complex, 115 math)
#    83 64-bit, 42 32-bit, 0 32-bit-const
