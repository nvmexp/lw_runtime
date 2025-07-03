!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     11
.MAX_IBUF    12
.MAX_OBUF    9
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v315-lw40.s -o allprogs-new32//v315-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[6].C[6]
#semantic C[5].C[5]
#semantic C[4].C[4]
#semantic C[3].C[3]
#semantic C[2].C[2]
#semantic C[1].C[1]
#semantic C[0].C[0]
#semantic c.c
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[BCOL0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 C[6] :  : c[6] : -1 : 0
#var float4 C[5] :  : c[5] : -1 : 0
#var float4 C[4] :  : c[4] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 C[3] :  : c[3] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[WGT].x
#ibuf 5 = v[WGT].y
#ibuf 6 = v[WGT].z
#ibuf 7 = v[NOR].x
#ibuf 8 = v[NOR].y
#ibuf 9 = v[COL0].x
#ibuf 10 = v[COL0].y
#ibuf 11 = v[COL0].z
#ibuf 12 = v[COL0].w
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[BCOL0].x
#obuf 5 = o[BCOL0].y
#obuf 6 = o[BCOL0].z
#obuf 7 = o[BCOL0].w
#obuf 8 = o[TEX0].x
#obuf 9 = o[TEX0].y
BB0:
F2I.FLOOR R0, v[12];
F2I.FLOOR R1, v[10];
I2I.M4   R0, R0;
I2I.M4   R1, R1;
R2A      A0, R0;
R2A      A1, R1;
FMUL     R1, v[1], c[A0 + 1];
FMUL     R0, v[1], c[A1 + 1];
FMAD     R1, v[0], c[A0], R1;
FMAD     R2, v[0], c[A1], R0;
FMUL     R0, v[1], c[A0 + 5];
FMAD     R1, v[2], c[A0 + 2], R1;
FMAD     R2, v[2], c[A1 + 2], R2;
FMAD     R0, v[0], c[A0 + 4], R0;
FMAD     R1, v[3], c[A0 + 3], R1;
FMAD     R2, v[3], c[A1 + 3], R2;
FMAD     R3, v[2], c[A0 + 6], R0;
FMUL     R0, v[1], c[A1 + 5];
FMUL     R2, v[9], R2;
FMAD     R4, v[3], c[A0 + 7], R3;
FMAD     R0, v[0], c[A1 + 4], R0;
FMAD     R2, v[11], R1, R2;
FMAD     R1, v[2], c[A1 + 6], R0;
FADD32   R3, -R2, c[16];
FMUL     R0, v[1], c[A0 + 9];
FMAD     R1, v[3], c[A1 + 7], R1;
FMAD     R0, v[0], c[A0 + 8], R0;
FMUL     R5, v[9], R1;
FMUL     R1, v[1], c[A1 + 9];
FMAD     R0, v[2], c[A0 + 10], R0;
FMAD     R4, v[11], R4, R5;
FMAD     R1, v[0], c[A1 + 8], R1;
FMAD     R0, v[3], c[A0 + 11], R0;
FADD32   R6, -R4, c[17];
FMAD     R1, v[2], c[A1 + 10], R1;
FMUL32   R5, R6, R6;
FMAD     R7, v[3], c[A1 + 11], R1;
FMUL     R1, v[5], c[A0 + 1];
FMAD     R5, R3, R3, R5;
FMUL     R8, v[9], R7;
FMAD     R7, v[4], c[A0], R1;
FMUL     R1, v[5], c[A1 + 1];
FMAD     R0, v[11], R0, R8;
FMAD     R7, v[6], c[A0 + 2], R7;
FMAD     R8, v[4], c[A1], R1;
FADD32   R1, -R0, c[18];
FMUL     R9, v[5], c[A0 + 5];
FMAD     R8, v[6], c[A1 + 2], R8;
FMAD     R5, R1, R1, R5;
FMAD     R9, v[4], c[A0 + 4], R9;
FMUL     R8, v[9], R8;
RSQ      R5, |R5|;
FMAD     R9, v[6], c[A0 + 6], R9;
FMAD     R8, v[11], R7, R8;
FMUL32   R11, R6, R5;
FMUL     R7, v[5], c[A1 + 5];
FMUL     R6, v[5], c[A0 + 9];
FMAD     R10, v[4], c[A1 + 4], R7;
FMAD     R6, v[4], c[A0 + 8], R6;
FMUL     R7, v[5], c[A1 + 9];
FMAD     R10, v[6], c[A1 + 6], R10;
FMAD     R6, v[6], c[A0 + 10], R6;
FMAD     R7, v[4], c[A1 + 8], R7;
FMUL     R10, v[9], R10;
FMAD     R7, v[6], c[A1 + 10], R7;
FMAD     R9, v[11], R9, R10;
FMUL     R7, v[9], R7;
FMUL32   R10, R3, R5;
FMUL32   R3, R9, R9;
FMAD     R6, v[11], R6, R7;
FMUL32   R1, R1, R5;
FMAD     R3, R8, R8, R3;
FMAD     R3, R6, R6, R3;
RSQ      R3, |R3|;
MOV32    R5, c[24];
FMUL32   R7, R9, R3;
FMUL32   R8, R8, R3;
FMUL32   R3, R6, R3;
FMUL32   R6, R11, R7;
MOV32    R7, c[25];
FMAD     R6, R10, R8, R6;
MOV32    R8, c[26];
FMAD     R1, R1, R3, R6;
MOV32    R3, R8;
MOV32    R6, c[27];
FMAX     R1, R1, c[19];
FMUL32   R8, R4, c[1];
FMAD     o[4], R1, c[20], R5;
FMAD     o[5], R1, c[21], R7;
FMAD     o[6], R1, c[22], R3;
FMAD     o[7], R1, c[23], R6;
FMAD     R1, R2, c[0], R8;
FMUL32   R3, R4, c[5];
FMUL32   R5, R4, c[9];
FMAD     R1, R0, c[2], R1;
FMAD     R3, R2, c[4], R3;
FMAD     R5, R2, c[8], R5;
FMAD     o[0], v[3], c[3], R1;
FMAD     R1, R0, c[6], R3;
FMAD     R3, R0, c[10], R5;
FMUL32   R4, R4, c[13];
FMAD     o[1], v[3], c[7], R1;
FMAD     o[2], v[3], c[11], R3;
FMAD     R1, R2, c[12], R4;
MOV      o[8], v[7];
MOV      o[9], v[8];
FMAD     R0, R0, c[14], R1;
FMAD     o[3], v[3], c[15], R0;
END
# 108 instructions, 12 R-regs
# 108 inst, (7 mov, 0 mvi, 0 tex, 2 complex, 99 math)
#    87 64-bit, 21 32-bit, 0 32-bit-const
