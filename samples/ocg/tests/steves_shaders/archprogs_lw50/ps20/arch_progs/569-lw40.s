//;; Id: 569   pixel count: 84114 lw40 ppc: 4.0
ps_2_0
def c0, 1.000000, 0.000000, 0.000000, 0.000000
dcl t0.rg
dcl_pp t1.rg
dcl_2d s0
dcl_2d s1
dcl_2d s2
texld r2, t0, s0
texld r1, t1, s1
texld r0, t1, s2
mul r0.rgb, r2, r1
add r0.rgb, r0, r0
add r0.a, -r0.a, c0.r
mov oC0, r0
