!!FP1.0
DECLARE C0={-0.000975, 0.000975, 0.000000, 0.000000};
DECLARE C1={0.000975, -0.000975, 0.000000, 0.000000};
ADDR R0.xy, f[TEX0], C0.x;
ADDR R1.xy, f[TEX0], C1;
ADDR R2.xy, f[TEX0], C0;
ADDR R3.xy, f[TEX0], C0.y;
TEX R0, R0, TEX0, 2D;
TEX R1, R1, TEX0, 2D;
TEX R2, R2, TEX0, 2D;
TEX R3, R3, TEX0, 2D;
MOVR R0.y, R1.x;
MOVR R0.z, R2.x;
MOVR R0.w, R3.x;
MOVR o[COLR], R0; 
END

# Passes = 9 

# Registers = 4 

# Textures = 1 
