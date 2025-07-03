!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX H1, f[TEX0], TEX0, 2D;
TEX H2, f[TEX1], TEX1, 2D;
TEXC HC, f[TEX2], TEX2, 2D;
KIL LT.xyzz;
ADDH H0.w, {1, 1, 1, 1}, -H2;
MULH H0.xyz, H1, H2;
MULH H0.xyz, C0, H0;
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MOVH o[COLH], H0; 
END

# Passes = 5 

# Registers = 2 

# Textures = 3 
