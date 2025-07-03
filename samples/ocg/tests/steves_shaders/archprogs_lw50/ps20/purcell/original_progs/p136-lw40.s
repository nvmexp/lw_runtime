texld r0, t0
texld r1, t1
texld r2, t2
texld r3, t3
mul r1, r1, r0.r
mad r1, r2, r0.g, r1
mad r1, r3, r0.b, r1
mul_x2 r0, r1, v0
