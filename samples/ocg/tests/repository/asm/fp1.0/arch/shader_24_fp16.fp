!!FP1.0
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX H3, f[TEX0], TEX0, 2D;
TEX H1, f[TEX1], TEX1, 2D;
TEX H2, f[TEX2], TEX2, 2D;
TEXC HC, f[TEX3], TEX3, 2D;
KIL LT.xyzz;
ADDH H0.w, {1, 1, 1, 1}, -H2.w;
MADH H4, H0.w, -H3, H3;
MADH H0, H0.w, H1, H4;
MULH H0, H0, H2;
MULH H0.xyz, C0, H0;
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MOVH o[COLH], H0; 
END

# Passes = 9 

# Registers = 3 

# Textures = 4 
