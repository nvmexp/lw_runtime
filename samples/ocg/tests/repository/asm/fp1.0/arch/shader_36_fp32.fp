!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX R0, f[TEX0], TEX0, 2D;
TEXC RC, f[TEX1], TEX1, 2D;
KIL LT.xyzz;
TEXC RC, f[TEX2], TEX2, 2D;
KIL LT.xyzz;
TEXC RC, f[TEX3], TEX3, 2D;
KIL LT.xyzz;
MADR R2.xyz, R0.w, -C0, C0;
MADR H0.xyz, R0.w, f[COL0], R2;
MOVR H0.w, C0.w;
MOVH o[COLH], H0; 
END

# Passes = 6 

# Registers = 3 

# Textures = 4 
