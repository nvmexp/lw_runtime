!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
TEX R0, f[TEX0], TEX0, 2D;
MULR R0, R0, f[COL0];
TEX R1, f[TEX2], TEX2, 2D;
MADR R0.xyz, R1, C2, R0;
TEX R1, f[TEX1], TEX1, 2D;
MULR R0.xyz, R1, R0;
MADR_m2 R0.xyz, C0, R0, C0.w;
END

# Passes = 4 

# Registers = 2 

# Textures = 3 
