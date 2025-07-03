//;; Id: 551   pixel count: 2526916 lw40 ppc: 4.0
ps_2_0
def c0, 16.000000, 0.000000, 0.000000, 0.000000
dcl t0.rg
dcl_pp t3.rg
dcl v0
dcl_2d s0
dcl_2d s1
texld r1, t3, s1
texld r0, t0, s0
mul r1.rgb, r1.a, r1
mul r1.rgb, r1, c0.r
mul r0.rgb, r0, v0
mul r0.a, r0.a, v0.a
mul r0.rgb, r1, r0
mov oC0, r0
