ps_2_0
def c0, 0.300000, 0.590000, 0.110000, 0.062500
dcl t0.rg
dcl_2d s0
texld r0, t0, s0
mul r0.rgb, r0, c1
dp3 r1.r, r0, c0
mul r0.a, r1.r, c0.a
mov oC0, r0
