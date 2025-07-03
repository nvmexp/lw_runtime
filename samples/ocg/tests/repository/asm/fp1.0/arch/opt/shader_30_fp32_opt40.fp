!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
TEX R1, f[TEX0], TEX0, 2D;
MULR R0, R1, f[COL0];
TEX R1, f[TEX3], TEX3, 2D;
ADDR R1.w, {1, 1, 1, 1}, -R1;
TEX R1.xyz, f[TEX2], TEX2, 2D;
MULR R1, R1, R1.w;
TEX R2, f[TEX1], TEX1, 2D;
MADR R0.xyz, R1, C2, R0;
MULR R0.xyz, R2, R0;
MADR_m2 H0.xyz, C0, R0, C0.w;
MOVR H0.w, R0.w;
END

# Passes = 6 

# Registers = 3 

# Textures = 4 
