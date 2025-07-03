!!FP1.0 
MADR R0, f[TEX1], {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
DP3R_SAT R0, R0, R0;
ADDR R0.w, {1, 1, 1, 1}, -R0;
MOVR o[COLR], R0; 
END

# Passes = 3 

# Registers = 1 

# Textures = 1 
