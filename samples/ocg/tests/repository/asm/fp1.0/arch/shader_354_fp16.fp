!!FP1.0
DECLARE C0={1, 2, 3, 4};
TEX H0, f[TEX0], TEX0, 2D;
TEX H1, f[TEX1], TEX1, 2D;
MULH H0.xyz, H1, H0;
ADDH H0.w, {1, 1, 1, 1}, -H1;
MULH H0.xyz, C0, H0;
MULH H0.xyz, H0, {2, 0, 0, 0}.x; 
MOVH o[COLH], H0; 
END

# Passes = 3 

# Registers = 1 

# Textures = 2 
