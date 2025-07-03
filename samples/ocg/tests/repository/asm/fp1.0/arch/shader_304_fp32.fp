!!FP1.0
TEX R0, f[TEX0], TEX0, 2D;
MULR R0.w, R0, {1.987654, 0, 0, 0}.x;
MULR R0, R0, f[COL0];
MULR H0.xyz, R0, {1.304100, 1.173690, 1.056321, 0.950689};
MOVR H0.w, R0.w;
MOVH o[COLH], H0; 
END

# Passes = 4 

# Registers = 2 

# Textures = 1 
