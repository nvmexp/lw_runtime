;; Id: 553   pixel count: 1087754 lw40 ppc: 5.33333333333
ps_2_0
dcl t0.rg
dcl v0.rgb
dcl_2d s0
texld r0, t0, s0
mul r1.rgb, v0, c1
mul r0.rgb, r0, r1
mul r0.a, r0.a, c1.a
mov oC0, r0
