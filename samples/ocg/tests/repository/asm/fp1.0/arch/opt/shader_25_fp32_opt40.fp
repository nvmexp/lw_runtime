!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0};
DECLARE C3={1, 2, 3, 4};
TEX RC, f[TEX3], TEX3, 2D;
KIL LT.xyzz;
TEX R0, f[TEX0], TEX0, 2D;
MULR R0, R0, C3;
MULR R0.xyz, f[COL0], R0;
MADR_m2 H0.xyz, C0, R0, C0.w;
MOVR H0.w, R0.w;
END

# Passes = 4 

# Registers = 2 

# Textures = 2 
