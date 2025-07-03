!!FP2.0
TEX R0, f[TEX0], TEX2, 2D;
TEX R1, f[TEX1], TEX2, 2D;
ADDR R0, R0, R1;
TEX R1, f[TEX2], TEX2, 2D;
ADDR R1, R1, R0;
TEX R0, f[TEX3], TEX2, 2D;
ADDR R0, R0, R1;
TEX R2, f[TEX0], TEX0, 2D;
MULR R1, R0, {0.98, 0, 0, 0}.x;
TEX R0, f[TEX1], TEX0, 2D;
ADDR R0.w, R2, R0;
TEX R2, f[TEX2], TEX0, 2D;
ADDR R2.w, R2, R0;
TEX R0, f[TEX3], TEX0, 2D;
ADDR R0.w, R0, R2;
TEX R2, f[TEX0], TEX1, 2D;
MADR R1, R0.w, {0.3, 0, 0, 0}.x, R1;
TEX R0, f[TEX1], TEX1, 2D;
ADDR R0, R2, R0;
TEX R2, f[TEX2], TEX1, 2D;
ADDR R2, R2, R0;
TEX R0, f[TEX3], TEX1, 2D;
ADDR R0, R0, R2;
MULR R0, R0, R1;
ADDR R0, R0.w, R0;
ADDR R0, R0, {-9.36, 0, 0, 0}.x;
END

# Passes = 15 

# Registers = 3 

# Textures = 4 
