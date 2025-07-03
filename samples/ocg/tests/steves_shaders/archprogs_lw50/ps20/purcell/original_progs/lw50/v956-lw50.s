!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     19
.MAX_IBUF    22
.MAX_OBUF    22
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v956-lw40.s -o allprogs-new32//v956-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[235].C[235]
#semantic C[234].C[234]
#semantic C[233].C[233]
#semantic C[236].C[236]
#semantic C[242].C[242]
#semantic C[241].C[241]
#semantic C[239].C[239]
#semantic C[238].C[238]
#semantic C[237].C[237]
#semantic C[240].C[240]
#semantic C[228].C[228]
#semantic C[227].C[227]
#semantic C[226].C[226]
#semantic C[225].C[225]
#semantic C[232].C[232]
#semantic C[231].C[231]
#semantic C[230].C[230]
#semantic C[229].C[229]
#semantic c.c
#semantic C[243].C[243]
#var float4 o[TEX5] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX4] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX3] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 C[235] :  : c[235] : -1 : 0
#var float4 C[234] :  : c[234] : -1 : 0
#var float4 C[233] :  : c[233] : -1 : 0
#var float4 C[236] :  : c[236] : -1 : 0
#var float4 C[242] :  : c[242] : -1 : 0
#var float4 C[241] :  : c[241] : -1 : 0
#var float4 v[COL1] : $vin.F : F[0] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 C[239] :  : c[239] : -1 : 0
#var float4 C[238] :  : c[238] : -1 : 0
#var float4 C[237] :  : c[237] : -1 : 0
#var float4 C[240] :  : c[240] : -1 : 0
#var float4 C[228] :  : c[228] : -1 : 0
#var float4 C[227] :  : c[227] : -1 : 0
#var float4 C[226] :  : c[226] : -1 : 0
#var float4 C[225] :  : c[225] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 C[232] :  : c[232] : -1 : 0
#var float4 C[231] :  : c[231] : -1 : 0
#var float4 C[230] :  : c[230] : -1 : 0
#var float4 C[229] :  : c[229] : -1 : 0
#var float4 v[TEX8] : $vin.F : F[0] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 C[243] :  : c[243] : -1 : 0
#var float4 v[FOGC] : $vin.F : F[0] : -1 : 0
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
#ibuf 12 = v[COL1].x
#ibuf 13 = v[COL1].y
#ibuf 14 = v[COL1].z
#ibuf 15 = v[FOG].x
#ibuf 16 = v[FOG].y
#ibuf 17 = v[FOG].z
#ibuf 18 = v[FOG].w
#ibuf 19 = v[UNUSED0].x
#ibuf 20 = v[UNUSED0].y
#ibuf 21 = v[UNUSED0].z
#ibuf 22 = v[UNUSED0].w
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[TEX0].x
#obuf 5 = o[TEX0].y
#obuf 6 = o[TEX1].x
#obuf 7 = o[TEX1].y
#obuf 8 = o[TEX1].z
#obuf 9 = o[TEX1].w
#obuf 10 = o[TEX2].x
#obuf 11 = o[TEX2].y
#obuf 12 = o[TEX2].z
#obuf 13 = o[TEX2].w
#obuf 14 = o[TEX3].x
#obuf 15 = o[TEX3].y
#obuf 16 = o[TEX3].z
#obuf 17 = o[TEX4].x
#obuf 18 = o[TEX4].y
#obuf 19 = o[TEX4].z
#obuf 20 = o[TEX5].x
#obuf 21 = o[TEX5].y
#obuf 22 = o[TEX5].z
BB0:
FMUL     R0, v[16], c[972];
FMUL     R1, v[15], c[972];
R2A      A2, R0;
R2A      A3, R1;
FMUL     R1, v[1], c[A2 + 1];
FMUL     R0, v[1], c[A3 + 1];
FMAD     R2, v[0], c[A2], R1;
FMAD     R0, v[0], c[A3], R0;
FMUL     R1, v[17], c[972];
FMAD     R2, v[2], c[A2 + 2], R2;
FMAD     R0, v[2], c[A3 + 2], R0;
R2A      A1, R1;
FMAD     R2, v[3], c[A2 + 3], R2;
FMAD     R3, v[3], c[A3 + 3], R0;
FMUL     R0, v[1], c[A1 + 1];
FMUL     R1, v[18], c[972];
FMUL     R3, v[19], R3;
FMAD     R0, v[0], c[A1], R0;
R2A      A0, R1;
FMAD     R3, v[20], R2, R3;
FMAD     R2, v[2], c[A1 + 2], R0;
FMUL     R1, v[1], c[A0 + 1];
FMUL     R0, v[1], c[A2 + 5];
FMAD     R2, v[3], c[A1 + 3], R2;
FMAD     R1, v[0], c[A0], R1;
FMAD     R0, v[0], c[A2 + 4], R0;
FMAD     R3, v[21], R2, R3;
FMAD     R1, v[2], c[A0 + 2], R1;
FMAD     R2, v[2], c[A2 + 6], R0;
FMUL     R0, v[1], c[A3 + 5];
FMAD     R1, v[3], c[A0 + 3], R1;
FMAD     R2, v[3], c[A2 + 7], R2;
FMAD     R0, v[0], c[A3 + 4], R0;
FMAD     R15, v[22], R1, R3;
FMUL     R1, v[1], c[A1 + 5];
FMAD     R3, v[2], c[A3 + 6], R0;
FMUL     R0, v[1], c[A0 + 5];
FMAD     R1, v[0], c[A1 + 4], R1;
FMAD     R3, v[3], c[A3 + 7], R3;
FMAD     R0, v[0], c[A0 + 4], R0;
FMAD     R1, v[2], c[A1 + 6], R1;
FMUL     R3, v[19], R3;
FMAD     R0, v[2], c[A0 + 6], R0;
FMAD     R1, v[3], c[A1 + 7], R1;
FMAD     R3, v[20], R2, R3;
FMAD     R2, v[3], c[A0 + 7], R0;
FMUL     R0, v[1], c[A2 + 9];
FMAD     R3, v[21], R1, R3;
FMUL     R1, v[1], c[A3 + 9];
FMAD     R0, v[0], c[A2 + 8], R0;
FMAD     R16, v[22], R2, R3;
FMAD     R1, v[0], c[A3 + 8], R1;
FMAD     R2, v[2], c[A2 + 10], R0;
FMUL32   R0, R16, c[905];
FMAD     R1, v[2], c[A3 + 10], R1;
FMAD     R3, v[3], c[A2 + 11], R2;
FMAD     R0, R15, c[901], R0;
FMAD     R4, v[3], c[A3 + 11], R1;
FMUL     R2, v[1], c[A1 + 9];
FMUL     R1, v[1], c[A0 + 9];
FMUL     R4, v[19], R4;
FMAD     R2, v[0], c[A1 + 8], R2;
FMAD     R1, v[0], c[A0 + 8], R1;
FMAD     R4, v[20], R3, R4;
FMAD     R3, v[2], c[A1 + 10], R2;
FMAD     R2, v[2], c[A0 + 10], R1;
FMUL32   R1, R16, c[904];
FMAD     R3, v[3], c[A1 + 11], R3;
FMAD     R2, v[3], c[A0 + 11], R2;
FMAD     R1, R15, c[900], R1;
FMAD     R3, v[21], R3, R4;
FMAD     R14, v[22], R2, R3;
FMAD     R2, R14, c[909], R0;
FMAD     R1, R14, c[908], R1;
FMUL32   R0, R16, c[906];
FADD32   R18, R2, c[913];
FADD32   R19, R1, c[912];
FMAD     R0, R15, c[902], R0;
FADD32   R9, -R18, c[965];
FADD32   R10, -R19, c[964];
FMAD     R0, R14, c[910], R0;
FMUL32   R1, R10, c[948];
FADD32   R17, R0, c[914];
FMUL32   R0, R10, c[949];
FMAD     R2, R9, c[952], R1;
FADD32   R1, -R17, c[966];
FMAD     R3, R9, c[953], R0;
FMUL     R0, v[7], c[A2 + 5];
FMAD     R2, R1, c[956], R2;
FMAD     R4, R1, c[957], R3;
FMAD     R5, v[6], c[A2 + 4], R0;
FMUL     R0, v[7], c[A3 + 5];
FMUL     R3, v[7], c[A1 + 5];
FMAD     R6, v[8], c[A2 + 6], R5;
FMAD     R0, v[6], c[A3 + 4], R0;
FMAD     R5, v[6], c[A1 + 4], R3;
FMUL     R3, v[7], c[A0 + 5];
FMAD     R0, v[8], c[A3 + 6], R0;
FMAD     R5, v[8], c[A1 + 6], R5;
FMAD     R3, v[6], c[A0 + 4], R3;
FMUL     R7, v[19], R0;
FMUL     R0, v[7], c[A2 + 1];
FMAD     R3, v[8], c[A0 + 6], R3;
FMAD     R7, v[20], R6, R7;
FMAD     R6, v[6], c[A2], R0;
FMUL     R0, v[7], c[A3 + 1];
FMAD     R5, v[21], R5, R7;
FMAD     R6, v[8], c[A2 + 2], R6;
FMAD     R0, v[6], c[A3], R0;
FMAD     R7, v[22], R3, R5;
FMUL     R5, v[7], c[A1 + 1];
FMAD     R3, v[8], c[A3 + 2], R0;
FMUL32   R0, R4, R7;
FMAD     R5, v[6], c[A1], R5;
FMUL     R8, v[19], R3;
FMUL     R3, v[7], c[A0 + 1];
FMAD     R5, v[8], c[A1 + 2], R5;
FMAD     R6, v[20], R6, R8;
FMAD     R3, v[6], c[A0], R3;
FMUL32   R8, R10, c[950];
FMAD     R5, v[21], R5, R6;
FMAD     R3, v[8], c[A0 + 2], R3;
FMAD     R8, R9, c[954], R8;
FMUL     R6, v[7], c[A2 + 9];
FMAD     R3, v[22], R3, R5;
FMAD     R1, R1, c[958], R8;
FMAD     R6, v[6], c[A2 + 8], R6;
FMAD     R5, R2, R3, R0;
FMUL     R0, v[7], c[A3 + 9];
FMAD     R8, v[8], c[A2 + 10], R6;
FMUL     R6, v[7], c[A1 + 9];
FMAD     R9, v[6], c[A3 + 8], R0;
FMUL     R0, v[7], c[A0 + 9];
FMAD     R6, v[6], c[A1 + 8], R6;
FMAD     R9, v[8], c[A3 + 10], R9;
FMAD     R0, v[6], c[A0 + 8], R0;
FMAD     R6, v[8], c[A1 + 10], R6;
FMUL     R9, v[19], R9;
FMAD     R0, v[8], c[A0 + 10], R0;
FMUL     R10, v[10], c[A2 + 5];
FMAD     R8, v[20], R8, R9;
FMUL     R9, v[10], c[A3 + 5];
FMAD     R10, v[9], c[A2 + 4], R10;
FMAD     R6, v[21], R6, R8;
FMAD     R8, v[9], c[A3 + 4], R9;
FMAD     R9, v[11], c[A2 + 6], R10;
FMAD     R0, v[22], R0, R6;
FMAD     R8, v[11], c[A3 + 6], R8;
FMUL     R6, v[10], c[A1 + 5];
FMAD     o[17], R1, R0, R5;
FMUL     R8, v[19], R8;
FMAD     R6, v[9], c[A1 + 4], R6;
FMUL     R5, v[10], c[A0 + 5];
FMAD     R9, v[20], R9, R8;
FMAD     R8, v[11], c[A1 + 6], R6;
FMAD     R6, v[9], c[A0 + 4], R5;
FMUL     R5, v[10], c[A2 + 1];
FMAD     R9, v[21], R8, R9;
FMAD     R8, v[11], c[A0 + 6], R6;
FMAD     R6, v[9], c[A2], R5;
FMUL     R5, v[10], c[A3 + 1];
FMAD     R10, v[22], R8, R9;
FMAD     R9, v[11], c[A2 + 2], R6;
FMAD     R8, v[9], c[A3], R5;
FMUL32   R6, R4, R10;
FMUL     R5, v[10], c[A1 + 1];
FMAD     R11, v[11], c[A3 + 2], R8;
FMUL     R8, v[10], c[A0 + 1];
FMAD     R5, v[9], c[A1], R5;
FMUL     R11, v[19], R11;
FMAD     R8, v[9], c[A0], R8;
FMAD     R5, v[11], c[A1 + 2], R5;
FMAD     R11, v[20], R9, R11;
FMAD     R9, v[11], c[A0 + 2], R8;
FMUL     R8, v[10], c[A2 + 9];
FMAD     R11, v[21], R5, R11;
FMUL     R5, v[10], c[A3 + 9];
FMAD     R8, v[9], c[A2 + 8], R8;
FMAD     R9, v[22], R9, R11;
FMAD     R5, v[9], c[A3 + 8], R5;
FMAD     R8, v[11], c[A2 + 10], R8;
FMAD     R11, R2, R9, R6;
FMAD     R12, v[11], c[A3 + 10], R5;
FMUL     R6, v[10], c[A1 + 9];
FMUL     R5, v[10], c[A0 + 9];
FMUL     R12, v[19], R12;
FMAD     R6, v[9], c[A1 + 8], R6;
FMAD     R5, v[9], c[A0 + 8], R5;
FMAD     R12, v[20], R8, R12;
FMAD     R6, v[11], c[A1 + 10], R6;
FMAD     R8, v[11], c[A0 + 10], R5;
FMUL     R5, v[13], c[A2 + 5];
FMAD     R12, v[21], R6, R12;
FMUL     R6, v[13], c[A3 + 5];
FMAD     R5, v[12], c[A2 + 4], R5;
FMAD     R8, v[22], R8, R12;
FMAD     R6, v[12], c[A3 + 4], R6;
FMAD     R12, v[14], c[A2 + 6], R5;
FMAD     o[18], R1, R8, R11;
FMAD     R11, v[14], c[A3 + 6], R6;
FMUL     R6, v[13], c[A1 + 5];
FMUL     R5, v[13], c[A0 + 5];
FMUL     R11, v[19], R11;
FMAD     R6, v[12], c[A1 + 4], R6;
FMAD     R5, v[12], c[A0 + 4], R5;
FMAD     R12, v[20], R12, R11;
FMAD     R6, v[14], c[A1 + 6], R6;
FMAD     R11, v[14], c[A0 + 6], R5;
FMUL     R5, v[13], c[A2 + 1];
FMAD     R12, v[21], R6, R12;
FMUL     R6, v[13], c[A3 + 1];
FMAD     R5, v[12], c[A2], R5;
FMAD     R13, v[22], R11, R12;
FMAD     R6, v[12], c[A3], R6;
FMAD     R11, v[14], c[A2 + 2], R5;
FMUL32   R5, R4, R13;
FMAD     R12, v[14], c[A3 + 2], R6;
FMUL     R4, v[13], c[A1 + 1];
FMUL     R6, v[13], c[A0 + 1];
FMUL     R12, v[19], R12;
FMAD     R4, v[12], c[A1], R4;
FMAD     R6, v[12], c[A0], R6;
FMAD     R12, v[20], R11, R12;
FMAD     R4, v[14], c[A1 + 2], R4;
FMAD     R11, v[14], c[A0 + 2], R6;
FMUL     R6, v[13], c[A2 + 9];
FMAD     R12, v[21], R4, R12;
FMUL     R4, v[13], c[A3 + 9];
FMAD     R6, v[12], c[A2 + 8], R6;
FMAD     R12, v[22], R11, R12;
FMAD     R4, v[12], c[A3 + 8], R4;
FMAD     R6, v[14], c[A2 + 10], R6;
FMAD     R2, R2, R12, R5;
FMAD     R11, v[14], c[A3 + 10], R4;
FMUL     R5, v[13], c[A1 + 9];
FMUL     R4, v[13], c[A0 + 9];
FMUL     R11, v[19], R11;
FMAD     R5, v[12], c[A1 + 8], R5;
FMAD     R4, v[12], c[A0 + 8], R4;
FMAD     R11, v[20], R6, R11;
FMAD     R5, v[14], c[A1 + 10], R5;
FMAD     R6, v[14], c[A0 + 10], R4;
MOV32    R4, -c[963];
FMAD     R11, v[21], R5, R11;
MOV32    R5, -c[963];
FMAD     R4, R18, R4, c[961];
FMAD     R11, v[22], R6, R11;
FMAD     R6, R19, R5, c[960];
MOV32    R5, -c[963];
FMAD     o[19], R1, R11, R2;
FMUL32   R1, R6, c[948];
FMAD     R2, R17, R5, c[962];
FMUL32   R5, R6, c[949];
FMAD     R1, R4, c[952], R1;
FMUL32   R6, R6, c[950];
FMAD     R5, R4, c[953], R5;
FMAD     R1, R2, c[956], R1;
FMAD     R4, R4, c[954], R6;
FMAD     R5, R2, c[957], R5;
FMAD     R2, R2, c[958], R4;
FMUL32   R4, R5, R7;
FMUL32   R6, R5, R10;
FMUL32   R5, R5, R13;
FMAD     R4, R1, R3, R4;
FMAD     R6, R1, R9, R6;
FMAD     R1, R1, R12, R5;
FMAD     o[14], R2, R0, R4;
FMAD     o[15], R2, R8, R6;
FMAD     o[16], R2, R11, R1;
MOV32    R4, c[953];
MOV32    R2, c[949];
MOV32    R1, c[957];
MOV32    R5, R4;
MOV32    R4, R1;
MOV32    R1, c[952];
FMUL32   R6, R2, c[968];
MOV32    R2, c[948];
FMAD     R5, R5, c[969], R6;
MOV32    R6, R2;
MOV32    R2, c[956];
FMAD     R4, R4, c[970], R5;
FMUL32   R5, R6, c[968];
FMUL32   R6, R4, R7;
FMAD     R7, R1, c[969], R5;
MOV32    R1, c[954];
MOV32    R5, c[950];
FMAD     R2, R2, c[970], R7;
FMAD     R3, R2, R3, R6;
FMUL32   R6, R4, R10;
FMUL32   R5, R5, c[968];
FMUL32   R4, R4, R13;
FMAD     R6, R2, R9, R6;
FMAD     R1, R1, c[969], R5;
FMAD     R4, R2, R12, R4;
MOV32    R2, c[958];
FMUL32   R7, R18, c[936];
FMUL32   R5, R16, c[907];
FMAD     R7, R19, c[932], R7;
FMAD     R5, R15, c[903], R5;
FMAD     R1, R2, c[970], R1;
FMAD     R2, R17, c[940], R7;
FMAD     R5, R14, c[911], R5;
FMAD     o[20], R1, R0, R3;
FMAD     o[21], R1, R8, R6;
FMAD     o[22], R1, R11, R4;
FADD32   R0, R5, c[915];
FMUL32   R1, R18, c[937];
FMUL32   R3, R18, c[938];
FMAD     R2, R0, c[944], R2;
FMAD     R1, R19, c[933], R1;
FMAD     R3, R19, c[934], R3;
FMUL32   R4, R18, c[939];
FMAD     R1, R17, c[941], R1;
FMAD     R3, R17, c[942], R3;
FMAD     R4, R19, c[935], R4;
FMAD     R1, R0, c[945], R1;
FMAD     R3, R0, c[946], R3;
FMAD     R4, R17, c[943], R4;
MOV32    o[6], R2;
MOV32    o[0], R2;
FMAD     R0, R0, c[947], R4;
MOV32    o[7], R1;
MOV32    o[1], R1;
MOV32    o[8], R3;
MOV32    o[2], R3;
MOV32    o[9], R0;
MOV32    o[3], R0;
FMUL32   R0, R15, c[916];
FMUL32   R1, R15, c[917];
FMUL32   R2, R15, c[918];
FMAD     R0, R16, c[920], R0;
FMAD     R1, R16, c[921], R1;
FMAD     R2, R16, c[922], R2;
FMAD     R0, R14, c[924], R0;
FMAD     R1, R14, c[925], R1;
FMAD     R2, R14, c[926], R2;
FADD32   o[10], R0, c[928];
FADD32   o[11], R1, c[929];
FADD32   o[12], R2, c[930];
FMUL32   R0, R15, c[919];
MOV      o[4], v[4];
MOV      o[5], v[5];
FMAD     R0, R16, c[923], R0;
FMAD     R0, R14, c[927], R0;
FADD32   o[13], R0, c[931];
END
# 345 instructions, 20 R-regs
# 345 inst, (29 mov, 0 mvi, 0 tex, 0 complex, 316 math)
#    281 64-bit, 64 32-bit, 0 32-bit-const
