ps_2_0
def c1, 1.000000, 0.000000, 0.000000, 0.000000
dcl t0.rg
dcl t1
dcl t2
dcl_2d s0
dcl_2d s2
texld r0, t0, s0
rcp_pp r1.a, t1.a
rcp_pp r0.a, t2.a
mul r2.rgb, r1.a, t1
mul r1.rgb, r0.a, t2
dp3 r2.r, r2, r0
dp3 r2.g, r1, r0
texld r0, r2, s2
mul r0.rgb, r0, c0
mov r0.a, c1.r
mov oC0, r0
