!!FP1.0
DP3R R0.w, f[TEX0], f[TEX0];
LG2R R0.x, |R0.w|;
MULR R0.x, R0, {0.5, 0, 0, 0}.x; 
EX2R R0.x, -R0.x;
MULR R0, R0.x, R0.w;
MOVR o[COLR], R0; 
END

# Passes = 5 

# Registers = 1 

# Textures = 1 
