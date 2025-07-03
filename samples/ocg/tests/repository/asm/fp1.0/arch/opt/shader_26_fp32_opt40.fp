!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0};
TEX RC, f[TEX2], TEX2, 2D;
KIL LT.xyzz;
TEX R1, f[TEX1], TEX1, 2D;
ADDR R0.w, {1, 1, 1, 1}, -R1;
TEX R0.xyz, f[TEX0], TEX0, 2D;
MULR R0.xyz, R0, R1;
MADR_m2 H0.xyz, C0, R0, C0.w;
MOVR H0.w, R0.w;
END

# Passes = 4 

# Registers = 2 

# Textures = 3 
