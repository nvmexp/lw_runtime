!!FP2.0 
DECLARE C0 = {0.9, 0.8, 0.7, 0};
TEX R0, f[TEX0], TEX0, 2D;
MULR R0, R0, f[COL0];
TEX R1, f[TEX1], TEX1, 2D;
MULR R0.xyz, R1, R0;
MADR_m2 H0.xyz, C0, R0, C0.w;
MOVR H0.w, R0.w;
END

# Passes = 3 

# Registers = 2 

# Textures = 2 
