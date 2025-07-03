!!FP1.0
TEX R0, f[TEX0], TEX0, 2D;
MULR R0.w, R0, {1.987654, 1.788889, 1.610000, 1.449000}.x;
MULR H0.xyz, R0, {1.304100, 1.173690, 1.056321, 0.0};
MOVR H0.w, R0;
MOVH o[COLH], H0; 
END

# Passes = 3 

# Registers = 2 

# Textures = 1 
