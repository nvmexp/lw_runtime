ps_2_0
dcl t0.rg
dcl v0
dcl_2d s0
texld r0, t0, s0
mul r0.a, r0.a, c1.a
mul r0.rgb, r0, v0
mul r0.a, r0.a, v0.a
mul r0.rgb, r0, c1
mov oC0, r0
