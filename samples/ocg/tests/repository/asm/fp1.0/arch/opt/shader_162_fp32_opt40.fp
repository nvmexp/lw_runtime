!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX R0, f[TEX2], TEX2, 2D;
TEX R2, f[TEX5], TEX3, 2D;
DP3R_SAT R2.y, R0, R2;
MOVR R2.z, C0.x;
TEX R1, f[TEX0], TEX1, 2D;
DP3R_SAT R2.x, R0, f[TEX4];
TEX R0, f[TEX0], TEX0, 2D;
MULR R2, R2.x, R0;
TEX R0, f[TEX3], TEX5, 2D;
MADR R2, R0, R1, R2;
DP3R_SAT R0, f[TEX1], f[TEX1];
ADDR R0.x, {1, 1, 1, 1}, -R0.x;
MULR R0, R0.x, R2;
MULR_m2 R0, R0, f[COL0];
END

# Passes = 10 

# Registers = 3 

# Textures = 6 
