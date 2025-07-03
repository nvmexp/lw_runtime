!!FP1.0
DECLARE C1={ 0.4, 0.5, 0.34, 0.45 };
DECLARE C4={ 0.7, 0.65, 0.45, 0.57 };
TEX H0, f[TEX0], TEX0, 2D;
MULH H1.xyz, f[COL0], C1;
MULH H0.xyz, H0, H1;
MULH H0.xyz, H0, C4;
MULH H0.w, H0.w, C1.w;
ADDH H0.xyz, H0, H0;
MOVH o[COLH], H0; 
END

# Passes = 3 

# Registers = 2 

# Textures = 1 
