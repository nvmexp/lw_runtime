!!FP2.0
DECLARE C0={0.5, 0.5, 0.5, 0.5};
TEX R0, f[TEX0], TEX0, 2D;
TEX R2, f[TEX2], TEX2, 2D;
ADDR_SAT R0.w, {1, 1, 1, 1}, -R2.w;
TEX R1, f[TEX1], TEX1, 2D;
MADR R0, R0.w, -R0, R0;
MADR R0, R0.w, R1, R0;
MULR R0, R0, R2;
MOVR R1.x, {0, 0, 0, 0};
MADR_m2 H0.xyz, C0, R0, R1.x;
MOVR H0.w, R0.w;
END

# Passes = 6 

# Registers = 3 

# Textures = 3 
