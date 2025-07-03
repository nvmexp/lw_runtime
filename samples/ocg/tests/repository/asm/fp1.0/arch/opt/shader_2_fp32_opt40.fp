!!FP2.0 
TEX RC, f[TEX0], TEX0, 2D;
KIL LT.xyzz;
DP3R_SAT R1, f[TEX2], f[TEX2];
ADDR R1, {1, 1, 1, 1}, -R1;
TEX R0, f[TEX1], TEX1, 2D;
MULR R0.w, R0, R1;
END

# Passes = 4 

# Registers = 2 

# Textures = 3 
