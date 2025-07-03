!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX R2, f[TEX2], TEX2, 2D;
DP3R_SAT R3.x, R2, f[TEX4];
TEX R0, f[TEX0], TEX0, 2D;
DP3R_SAT R3.y, R2, f[TEX5];
MOVR R3.z, C0.x;
TEX R2, R3, TEX5, 2D;
MULR R3, R3.x, R0;
TEX R1, f[TEX0], TEX1, 2D;
MADR R3, R2, R1, R3;
DP3R_SAT R0, f[TEX1], f[TEX1];
MOVR R1, f[COL0];
ADDR R0.x, {1, 1, 1, 1}, -R0.x;
MULR R0, R0.x, R3;
MULR_m2 R0, R0, R1;
END

# Passes = 10 

# Registers = 4 

# Textures = 5 
