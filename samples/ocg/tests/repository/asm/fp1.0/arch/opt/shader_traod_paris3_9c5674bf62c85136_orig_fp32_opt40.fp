!!FP2.0
DECLARE C0={ 0.1, 0.3, 0.5, 0.9};
DECLARE C1={ 0.25, 0.20, 0.3, 0.23};
DECLARE C2={ 0.33, 0.13, 0.25, 0.20};
TEX R0, f[TEX2], TEX2, 2D;
DP3R R0.w, R0, R0;
LG2R R0.w, |R0.w|;
MULR R0.w, R0, C0.w;
EX2R R0.w, R0.w;
MULR R0.xyz, R0, R0.w;
MADR R0.xyz, R0, C0.y, f[TEX1];
DP3R R0.w, R0, R0;
LG2R R0.w, |R0.w|;
MULR R0.w, R0, C0.w;
EX2R R0.w, R0.w;
MULR R3.xyz, R0, R0.w;
ADDR R0.xyz, R3, R3;
DP3R R0.w, R3, C1;
MADR R5.xyz, R0, R0.w, -C1;
DP3R R5.w, f[TEX3], f[TEX3];
LG2R R0.x, |R5.w|;
MULR R0.x, R0, C0.w;
EX2R R5.w, R0.x;
MOVR R2, f[TEX0];
MADR R2.xy, R3, C1, R2;
MOVR R0, f[TEX0];
DP4R R0.x, R0, C1;
DP4R R4.w, R2, C1;
DP4R R0.y, f[TEX0], C1;
DP4R R0.w, f[TEX0], C2;
TXP R1, R0, TEX0, 2D;
DP4R R4.x, R2, C1;
DP4R R4.y, R2, C2;
ADDR R2.xyw, R4, -R0;
MULR R2.xyw, R1.w, R2;
MULR R6.xyz, R5.w, f[TEX3];
DP3R_SAT R1.w, R5, R6;
LG2R R1.w, |R1.w|;
MULR R1.w, R1, C0.w;
EX2R R1.w, R1.w;
ADDR R2.xyw, R0, R2;
TXP R2, R2, TEX0, 2D;
ADDR R2.xyz, R2, -R1;
MADR R1.xyz, R2, R2.w, R1;
MULR R5.xyz, R1, C1;
MADR R5.xyz, R1.w, C1, R5;
MOVR R1.zw, f[TEX0];
MADR R1.xy, R3, C0.y, f[TEX0];
DP4R R4.x, R1, C2;
DP4R R4.w, R1, C1;
DP4R R4.y, R1, C2;
ADDR R1.xyw, -R0, R4;
TXP R2, R0, TEX1, 2D;
MADR R0.xyw, R1, R2.w, R0;
DP3R R2.w, R3, R6;
MADR_SAT R5.w, -R2, C0.y, C0.z;
TXP R0, R0, TEX2, 2D;
ADDR R0.xyz, R0, -R2;
MADR R2.xyz, R0, R0.w, R2;
MOVR R0.w, C0.z;
MADR R0.xyz, R2, C0.w, -R5;
MADR R0.xyz, R0, R5.w, R5;
END

# Passes = -1 

# Registers = -1 

# Textures = -1 
