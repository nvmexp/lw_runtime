!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
TEX H0, f[TEX0], TEX0, 2D;
TEX H1, f[TEX1], TEX1, 2D;
TEX H2, f[TEX2], TEX2, 2D;
MULH H0, H0, f[COL0];
MADH H0.xyz, H2, C2, H0;
MULH H0.xyz, H1, H0;
MULH H0.xyz, C0, H0;
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MOVH o[COLH], H0; 
END

# Passes = 5 

# Registers = 2 

# Textures = 3 
