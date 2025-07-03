!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX H0, f[TEX0], TEX0, 2D;
TEXC HC, f[TEX1], TEX1, 2D;
KIL LT.xyzz;
TEXC HC, f[TEX2], TEX2, 2D;
KIL LT.xyzz;
TEXC HC, f[TEX3], TEX3, 2D;
KIL LT.xyzz;
MADH H2.xyz, H0.w, -C0, C0;
MADH H0.xyz, H0.w, f[COL0], H2;
MOVH H0.w, C0.w;
MOVH o[COLH], H0; 
END

# Passes = 6 

# Registers = 2 

# Textures = 4 
