!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX R0, f[TEX0], TEX0, 2D;
DP3R R1.x, f[TEX1], R0;
DP3R R1.y, f[TEX2], R0;
TEX R0, f[TEX3], TEX3, 2D;
MADR R2, f[COL0], {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
DP3R_SAT R2, R2, R0;
ADDR R2.w, {1, 1, 1, 1}, -R2.w;
MULR R0.w, R2.w, R2.w;
MULR R0.w, R0.w, R0.w;
MULR R0.w, R0.w, R2.w;
ADDR R2.w, {1, 1, 1, 1}, -C0.w;
MADR R0.w, R0.w, R2.w, C0.w;
TEX R1, R1, TEX2, 2D;
MULR R0, R0.w, R1;
END

# Passes = 11 

# Registers = 3 

# Textures = 4 
