!!FP1.0
DECLARE C0={0.1, 0.2, 0.3, 0.4};
MOVR R0.xyz, C0;
MOVR R0.w, f[TEX0].z;
MOVR o[COLR], R0; 
END

# Passes = 2 

# Registers = 1 

# Textures = 1 
