!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
DECLARE C3={1, 2, 3, 4};
TEX R1, f[TEX1], TEX1, 2D;
DP3R_SAT R0, R1, C0;
MULR R2.xyz, R0, f[TEX2];
DP3R_SAT R0, R1, C1;
MADR R2.xyz, R0, f[TEX3], R2;
DP3R_SAT R0, R1, C2;
MADR R2.xyz, R0, f[COL0], R2;
TEX R0.xyz, f[TEX0], TEX0, 2D;
MULR R2.xyz, R2, C3;
MADR_m2 R0.xyz, R2, R0, {0, 0, 0, 0}.x;
MOVR R0.w, C3;
END

# Passes = 8 

# Registers = 3 

# Textures = 4 
