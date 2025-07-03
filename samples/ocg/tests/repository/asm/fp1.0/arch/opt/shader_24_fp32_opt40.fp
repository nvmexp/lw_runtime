!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0};
TEX RC, f[TEX3], TEX3, 2D;
KIL LT.xyzz;
TEX R2, f[TEX2], TEX2, 2D;
ADDR R0.w, {1, 1, 1, 1}, -R2.w;
TEX R3, f[TEX0], TEX0, 2D;
MADR R3, R0.w, -R3, R3;
TEX R1, f[TEX1], TEX1, 2D;
MADR R0, R0.w, R1, R3;
MULR R0, R0, R2;
MADR_m2 H0.xyz, C0, R0, C0.w;
MOVR H0.w, R0.w;
END

# Passes = 6 

# Registers = 4 

# Textures = 4 
