!!FP2.0
DECLARE C0={-0.000975, 0.000975, 0.000000, 0.000000};
DECLARE C1={0.000975, -0.000975, 0.000000, 0.000000};
MOVR R3.xy, f[TEX0];
ADDR R0.xy, R3, C0.x;
TEX R0, R0, TEX0, 2D;
ADDR R1.xy, R3, C1;
TEX R1, R1, TEX0, 2D;
ADDR R2.xy, R3, C0.y;
TEX R2, R2, TEX0, 2D;
MOVR R0.w, R2.x;
ADDR R2.xy, R3, C0;
TEX R2, R2, TEX0, 2D;
MOVR R0.y, R1.x;
MOVR R0.z, R2.x;
END

# Passes = 5 

# Registers = 4 

# Textures = 1 
