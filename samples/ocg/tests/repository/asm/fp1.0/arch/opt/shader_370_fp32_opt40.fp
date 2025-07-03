!!FP2.0
DECLARE C0={0.200000, 0.400000, 6.000000, 1.000000};
DECLARE C1={0.000000, 0.000000, 0.500000, 0.454545};
DECLARE C2={0.058594, 0.000000, 0.000000, 0.000000};
TEX R0, f[TEX2], TEX0, 2D;
MOVR RC, R0;
MOVR R2.xyz(GE), R0;
MOVR R2.xyz(LT), C1.x;
MOVR R3.xyz, R0.w;
MOVR R3.w, C2.x;
LG2R R2.x, R2.x;
MOVR R0, C0;
LG2R R2.y, R2.y;
LG2R R2.z, R2.z;
MULR R2.xyz, R2, C1.w;
EX2R R2.x, R2.x;
MOVR R2.w, C0.w;
EX2R R2.y, R2.y;
TEX H8, f[TEX2], TEX1, 2D;
EX2R R2.z, R2.z;
NRMH H8.xyz, H8;
MADR R4.xyz, H8, C1.z, C1.z;
MOVR R4.w, C0.w;
END

# Passes = 13 

# Registers = 5 

# Textures = 1 
