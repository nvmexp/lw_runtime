!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX RC, f[TEX1], TEX1, 2D;
KIL LT.xyzz;
TEX RC, f[TEX2], TEX2, 2D;
KIL LT.xyzz;
TEX RC, f[TEX3], TEX3, 2D;
KIL LT.xyzz;
TEX R0, f[TEX0], TEX0, 2D;
MADR R0.xyz, R0.w, -C0, C0;
MADR H0.xyz, R0.w, f[COL0], R0;
MOVR H0.w, C0.w;
END

# Passes = 5 

# Registers = 2 

# Textures = 4 
