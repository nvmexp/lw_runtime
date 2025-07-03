!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C3={1, 2, 3, 4};
TEX H2, f[TEX0], TEX0, 2D;
TEXC HC, f[TEX3], TEX3, 2D;
KIL LT.xyzz;
MULH H0.xyz, H2, C3;
MOVH H0.w, C3.w;
MULH H0.xyz, f[COL0], H0;
MULH H0.xyz, C0, H0;
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MULH H1, H2, C1;
MADH H0.xyz, H2.w, -H0, H0;
MADH H0.xyz, H2.w, H1, H0;
MOVH o[COLH], H0; 
END

# Passes = 7 

# Registers = 2 

# Textures = 2 
