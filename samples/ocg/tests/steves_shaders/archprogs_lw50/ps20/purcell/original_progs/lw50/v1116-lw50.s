!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     7
.MAX_IBUF    15
.MAX_OBUF    39
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v1116-lw40.s -o allprogs-new32//v1116-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[38].C[38]
#semantic C[16].C[16]
#semantic C[95].C[95]
#semantic C[94].C[94]
#semantic C[93].C[93]
#semantic C[92].C[92]
#semantic C[97].C[97]
#semantic C[96].C[96]
#semantic C[91].C[91]
#semantic C[90].C[90]
#semantic C[2].C[2]
#semantic C[6].C[6]
#semantic C[44].C[44]
#semantic C[43].C[43]
#semantic C[42].C[42]
#semantic C[7].C[7]
#semantic C[5].C[5]
#semantic C[4].C[4]
#semantic C[0].C[0]
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
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[95] :  : c[95] : -1 : 0
#var float4 C[94] :  : c[94] : -1 : 0
#var float4 C[93] :  : c[93] : -1 : 0
#var float4 C[92] :  : c[92] : -1 : 0
#var float4 C[97] :  : c[97] : -1 : 0
#var float4 C[96] :  : c[96] : -1 : 0
#var float4 C[91] :  : c[91] : -1 : 0
#var float4 C[90] :  : c[90] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[6] :  : c[6] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 v[FOGC] : $vin.F : F[0] : -1 : 0
#var float4 C[44] :  : c[44] : -1 : 0
#var float4 C[43] :  : c[43] : -1 : 0
#var float4 C[42] :  : c[42] : -1 : 0
#var float4 v[COL1] : $vin.F : F[0] : -1 : 0
#var float4 C[7] :  : c[7] : -1 : 0
#var float4 C[5] :  : c[5] : -1 : 0
#var float4 C[4] :  : c[4] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[WGT].x
#ibuf 4 = v[WGT].y
#ibuf 5 = v[WGT].z
#ibuf 6 = v[NOR].x
#ibuf 7 = v[NOR].y
#ibuf 8 = v[COL0].x
#ibuf 9 = v[COL0].y
#ibuf 10 = v[COL1].x
#ibuf 11 = v[COL1].y
#ibuf 12 = v[COL1].z
#ibuf 13 = v[FOG].x
#ibuf 14 = v[FOG].y
#ibuf 15 = v[FOG].z
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
#obuf 10 = o[TEX0].z
#obuf 11 = o[TEX0].w
#obuf 12 = o[TEX1].x
#obuf 13 = o[TEX1].y
#obuf 14 = o[TEX1].z
#obuf 15 = o[TEX1].w
#obuf 16 = o[TEX2].x
#obuf 17 = o[TEX2].y
#obuf 18 = o[TEX2].z
#obuf 19 = o[TEX2].w
#obuf 20 = o[TEX3].x
#obuf 21 = o[TEX3].y
#obuf 22 = o[TEX3].z
#obuf 23 = o[TEX3].w
#obuf 24 = o[TEX4].x
#obuf 25 = o[TEX4].y
#obuf 26 = o[TEX4].z
#obuf 27 = o[TEX5].x
#obuf 28 = o[TEX5].y
#obuf 29 = o[TEX5].z
#obuf 30 = o[TEX6].x
#obuf 31 = o[TEX6].y
#obuf 32 = o[TEX6].z
#obuf 33 = o[TEX7].x
#obuf 34 = o[TEX7].y
#obuf 35 = o[TEX7].z
#obuf 36 = o[FOGC].x
#obuf 37 = o[FOGC].y
#obuf 38 = o[FOGC].z
#obuf 39 = o[FOGC].w
BB0:
MOV32    R0, c[171];
FMUL     R1, v[1], c[169];
MOV32    R2, c[175];
FMAD     R1, v[0], c[168], R1;
FMUL     R3, v[1], c[173];
FMAD     R1, v[2], c[170], R1;
MOV32    R4, c[179];
FMAD     R3, v[0], c[172], R3;
FMAD     R0, R0, c[1], R1;
MOV32    R1, R4;
FMAD     R3, v[2], c[174], R3;
FADD32   o[24], -R0, c[8];
FMUL     R0, v[1], c[177];
FMAD     R2, R2, c[1], R3;
MOV32    R3, c[27];
FMAD     R0, v[0], c[176], R0;
FADD32   o[25], -R2, c[9];
MOV32    R2, R3;
FMAD     R0, v[2], c[178], R0;
FMUL     R3, v[1], c[25];
MOV32    R4, c[67];
FMAD     R0, R1, c[1], R0;
FMAD     R1, v[0], c[24], R3;
FADD32   o[26], -R0, c[10];
FMAD     R0, v[2], c[26], R1;
MOV32    R1, c[19];
FMUL     R3, v[1], c[17];
FMAD     R0, R2, c[1], R0;
FMAD     R2, v[0], c[16], R3;
FMAD     R3, -R0, R4, c[64];
MOV32    R4, c[23];
FMAD     R2, v[2], c[18], R2;
MOV32    o[36], R3;
FMAD     o[0], R1, c[1], R2;
MOV32    o[2], R0;
MOV32    o[37], R3;
MOV32    o[38], R3;
MOV32    o[39], R3;
FMUL     R0, v[1], c[21];
MOV32    R2, c[31];
FMUL     R1, v[1], c[29];
FMAD     R0, v[0], c[20], R0;
FMAD     R3, v[0], c[28], R1;
FMAD     R1, v[2], c[22], R0;
FMUL     R0, v[4], c[169];
FMAD     R3, v[2], c[30], R3;
FMAD     o[1], R4, c[1], R1;
FMAD     R0, v[3], c[168], R0;
FMAD     o[3], R2, c[1], R3;
FMUL     R1, v[4], c[173];
FMAD     o[33], v[5], c[170], R0;
FMUL     R0, v[4], c[177];
FMAD     R1, v[3], c[172], R1;
FMUL     R2, v[14], c[169];
FMAD     R0, v[3], c[176], R0;
FMAD     o[34], v[5], c[174], R1;
FMAD     R1, v[13], c[168], R2;
FMAD     o[35], v[5], c[178], R0;
FMUL     R0, v[14], c[173];
FMAD     o[30], v[15], c[170], R1;
FMUL     R1, v[14], c[177];
FMAD     R0, v[13], c[172], R0;
FMUL     R2, v[11], c[169];
FMAD     R1, v[13], c[176], R1;
FMAD     o[31], v[15], c[174], R0;
FMAD     R0, v[10], c[168], R2;
FMAD     o[32], v[15], c[178], R1;
FMUL     R1, v[11], c[173];
FMAD     o[27], v[12], c[170], R0;
FMUL     R0, v[11], c[177];
FMAD     R1, v[10], c[172], R1;
MOV      o[20], v[8];
FMAD     R0, v[10], c[176], R0;
FMAD     o[28], v[12], c[174], R1;
MOV      o[21], v[9];
FMAD     o[29], v[12], c[178], R0;
MOV32    R0, c[0];
MOV32    R1, c[0];
MOV32    o[16], c[0];
MOV32    o[22], R0;
MOV32    o[23], R1;
MOV32    R0, c[0];
MOV32    R1, c[0];
MOV32    R2, c[0];
MOV32    o[17], R0;
MOV32    o[18], R1;
MOV32    o[19], R2;
FMUL     R0, v[6], c[368];
FMUL     R1, v[6], c[372];
FMUL     R2, v[6], c[380];
FMAD     o[12], v[7], c[369], R0;
FMAD     o[13], v[7], c[373], R1;
FMAD     o[14], v[7], c[381], R2;
FMUL     R0, v[6], c[376];
FMUL     R1, v[6], c[360];
FMUL     R2, v[6], c[364];
FMAD     o[15], v[7], c[377], R0;
FMAD     o[8], v[7], c[361], R1;
FMAD     o[9], v[7], c[365], R2;
FMUL     R0, v[6], c[384];
FMUL     R1, v[6], c[388];
MOV32    o[4], c[152];
FMAD     o[10], v[7], c[385], R0;
FMAD     o[11], v[7], c[389], R1;
MOV32    o[5], c[153];
MOV32    o[6], c[154];
MOV32    o[7], c[155];
END
# 107 instructions, 8 R-regs
# 107 inst, (32 mov, 0 mvi, 0 tex, 0 complex, 75 math)
#    74 64-bit, 33 32-bit, 0 32-bit-const
