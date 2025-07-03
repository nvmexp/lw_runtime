!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     3
.MAX_IBUF    8
.MAX_OBUF    35
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v972-lw40.s -o allprogs-new32//v972-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[38].C[38]
#semantic C[0].C[0]
#semantic C[16].C[16]
#semantic C[6].C[6]
#semantic C[93].C[93]
#semantic C[92].C[92]
#semantic C[91].C[91]
#semantic C[90].C[90]
#semantic C[2].C[2]
#semantic C[44].C[44]
#semantic C[43].C[43]
#semantic C[42].C[42]
#semantic C[7].C[7]
#semantic C[5].C[5]
#semantic C[4].C[4]
#var float4 o[TEX7] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX6] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX5] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX4] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX3] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[FOGC] : $vout.O : O[0] : -1 : 0
#var float4 o[BCOL0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 C[38] :  : c[38] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[6] :  : c[6] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 C[93] :  : c[93] : -1 : 0
#var float4 C[92] :  : c[92] : -1 : 0
#var float4 C[91] :  : c[91] : -1 : 0
#var float4 C[90] :  : c[90] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[44] :  : c[44] : -1 : 0
#var float4 C[43] :  : c[43] : -1 : 0
#var float4 C[42] :  : c[42] : -1 : 0
#var float4 C[7] :  : c[7] : -1 : 0
#var float4 C[5] :  : c[5] : -1 : 0
#var float4 C[4] :  : c[4] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[WGT].x
#ibuf 4 = v[WGT].y
#ibuf 5 = v[NOR].x
#ibuf 6 = v[NOR].y
#ibuf 7 = v[COL0].x
#ibuf 8 = v[COL0].y
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
#obuf 10 = o[TEX1].x
#obuf 11 = o[TEX1].y
#obuf 12 = o[TEX2].x
#obuf 13 = o[TEX2].y
#obuf 14 = o[TEX2].z
#obuf 15 = o[TEX2].w
#obuf 16 = o[TEX3].x
#obuf 17 = o[TEX3].y
#obuf 18 = o[TEX3].z
#obuf 19 = o[TEX3].w
#obuf 20 = o[TEX4].x
#obuf 21 = o[TEX4].y
#obuf 22 = o[TEX4].z
#obuf 23 = o[TEX5].x
#obuf 24 = o[TEX5].y
#obuf 25 = o[TEX5].z
#obuf 26 = o[TEX6].x
#obuf 27 = o[TEX6].y
#obuf 28 = o[TEX6].z
#obuf 29 = o[TEX7].x
#obuf 30 = o[TEX7].y
#obuf 31 = o[TEX7].z
#obuf 32 = o[FOGC].x
#obuf 33 = o[FOGC].y
#obuf 34 = o[FOGC].z
#obuf 35 = o[FOGC].w
BB0:
FMUL     R0, v[1], c[25];
MOV32    R1, c[67];
FMAD     R0, v[0], c[24], R0;
MOV32    o[29], c[0];
MOV32    R2, c[0];
FMAD     R3, v[2], c[26], R0;
MOV32    R0, c[0];
MOV32    o[30], R2;
FADD32   R3, R3, c[27];
MOV32    o[31], R0;
MOV32    o[26], c[0];
FMAD     R2, -R3, R1, c[64];
MOV32    R0, c[0];
MOV32    R1, c[0];
MOV32    o[32], R2;
MOV32    o[27], R0;
MOV32    o[28], R1;
MOV32    o[33], R2;
MOV32    o[34], R2;
MOV32    o[35], R2;
MOV32    o[23], c[0];
MOV32    R0, c[0];
MOV32    R1, c[0];
FMUL     R2, v[1], c[169];
MOV32    o[24], R0;
MOV32    o[25], R1;
FMAD     R0, v[0], c[168], R2;
FMUL     R1, v[1], c[173];
FMUL     R2, v[1], c[177];
FMAD     R0, v[2], c[170], R0;
FMAD     R1, v[0], c[172], R1;
FMAD     R2, v[0], c[176], R2;
FADD32   R0, R0, c[171];
FMAD     R1, v[2], c[174], R1;
FMAD     R2, v[2], c[178], R2;
FADD32   o[20], -R0, c[8];
FADD32   R0, R1, c[175];
FADD32   R1, R2, c[179];
MOV32    o[16], c[0];
FADD32   o[21], -R0, c[9];
FADD32   o[22], -R1, c[10];
MOV32    R0, c[0];
MOV      R1, v[5];
MOV      R2, v[6];
MOV32    o[17], R0;
FADD     R0, v[7], R1;
FADD     R2, v[8], R2;
FADD     R1, v[7], R0;
MOV32    o[12], R0;
FADD     R0, v[8], R2;
MOV32    o[13], R2;
FADD     o[18], v[7], R1;
MOV32    o[14], R1;
FADD     o[19], v[8], R0;
MOV32    o[15], R0;
FMUL     R0, v[3], c[368];
FMUL     R1, v[3], c[372];
FMUL     R2, v[3], c[360];
FMAD     o[10], v[4], c[369], R0;
FMAD     o[11], v[4], c[373], R1;
FMAD     o[8], v[4], c[361], R2;
FMUL     R0, v[3], c[364];
MOV32    o[4], c[152];
MOV32    o[5], c[153];
FMAD     o[9], v[4], c[365], R0;
MOV32    o[6], c[154];
MOV32    o[7], c[155];
FMUL     R0, v[1], c[17];
MOV32    o[2], R3;
FMUL     R1, v[1], c[21];
FMAD     R0, v[0], c[16], R0;
FMUL     R2, v[1], c[29];
FMAD     R1, v[0], c[20], R1;
FMAD     R0, v[2], c[18], R0;
FMAD     R2, v[0], c[28], R2;
FMAD     R1, v[2], c[22], R1;
FADD32   o[0], R0, c[19];
FMAD     R0, v[2], c[30], R2;
FADD32   o[1], R1, c[23];
FADD32   o[3], R0, c[31];
END
# 80 instructions, 4 R-regs
# 80 inst, (34 mov, 0 mvi, 0 tex, 0 complex, 46 math)
#    38 64-bit, 42 32-bit, 0 32-bit-const
