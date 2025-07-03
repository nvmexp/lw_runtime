ps_2_0
def c0, 0.300000, 0.590000, 0.110000, 0.062500
dcl t0.rg
dcl v0
dcl_2d s0
texld r0, t0, s0
mul r0.rgb, r0, v0
mul r0.a, r0.a, c1.a
mul r0.rgb, r0, c1
mul r0.a, r0.a, v0.a
dp3 r1.r, r0, c0
mul r0.a, r0.a, r1.r
mul r0.a, r0.a, c0.a
mov oC0, r0
