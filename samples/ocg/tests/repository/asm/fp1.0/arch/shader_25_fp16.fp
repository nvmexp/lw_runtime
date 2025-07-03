!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C3={1, 2, 3, 4};
TEX H0, f[TEX0], TEX0, 2D;
TEXC HC, f[TEX3], TEX3, 2D;
KIL LT.xyzz;
MULH H0, H0, C3;
MULH H0.xyz, f[COL0], H0;
MULH H0.xyz, C0, H0;
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MOVH o[COLH], H0; 
END

# Passes = 6 

# Registers = 2 

# Textures = 2 
