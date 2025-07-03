!!FP2.0
DECLARE C0={0.000000, 0.000000, 0.000000, 0.000000};
MOVR R0.xyz, f[TEX1];
MOVR R0.w, C0.x;
MOVR R3.xyz, f[TEX2];
DP3R R1.w, R3, R3;
TEX R1.xyz, f[TEX0], TEX0, 2D;
LG2R_d2 R1.w, R1.w;
EX2R R1.w, R1.w;
MULR R2.xyz, R1.w, R3;
MOVR R3.w, C0.x;
MOVR R2.w, f[COL0].x;
MADR R3.xyz, {0, 0, 0, 0}, R1, R1;
END

# Passes = 6 

# Registers = 5 

# Textures = 3 
