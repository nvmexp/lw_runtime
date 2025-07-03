!!FP1.0
TEXC RC, f[TEX0], TEX0, 2D;
KIL LT.xyzz;
TEX R1, f[TEX1], TEX1, 2D;
DP3R_SAT R2, f[TEX2], f[TEX2];
MOVR R0.xyz, R1;
ADDR R2, {1, 1, 1, 1}, -R2;
MULR R0.w, R1, R2;
MOVR o[COLR], R0; 
END

# Passes = 5 

# Registers = 3 

# Textures = 3 
