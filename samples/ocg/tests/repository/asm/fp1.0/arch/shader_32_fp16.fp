!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
DECLARE C3={1, 2, 3, 4};
TEX H2, f[TEX0], TEX0, 2D;
TEX H3, f[TEX1], TEX1, 2D;
TEX H4, f[TEX2], TEX2, 2D;
TEXC HC, f[TEX3], TEX3, 2D;
KIL LT.xyzz;
MULH H0, H2, C3;
ADDH H4.w, {1, 1, 1, 1}, -H4;
MULH H1, H3, H4.w;
MADH H0.xyz, H1, C2, H0;
MULH H0.xyz, f[COL0], H0;
MULH H0.xyz, C0, H0;
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MOVH o[COLH], H0; 
END

# Passes = 9 

# Registers = 3 

# Textures = 4 
