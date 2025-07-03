!!FP1.0 
MADH H0, f[TEX1], {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
DP3H_SAT H0, H0, H0;
ADDH H0.w, {1, 1, 1, 1}, -H0;
MOVH o[COLH], H0; 
END

# Passes = 3 

# Registers = 1 

# Textures = 1 
