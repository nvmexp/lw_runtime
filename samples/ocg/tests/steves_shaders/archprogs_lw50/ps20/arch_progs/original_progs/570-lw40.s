;; Id: 570   pixel count: 84114 lw40 ppc: 5.33333333333
ps_2_0
def c0, 0.333330, 1.000000, 0.000000, 0.000000
def c1, 0.000000, 1.000000, 0.000000, 0.000000
dcl t0.rg
dcl_pp t1.rg
dcl_2d s0
dcl_2d s1
dcl_2d s2
texld r2, t0, s0
texld r1, t1, s1
texld r0, t1, s2
dp3 r0.r, r2, c0.r
mul r1.a, r1.a, r0.r
add r1.a, r1.a, r1.a
add r0.a, -r0.a, c0.g
mov r0.gb, c1.brga
mul r0.a, r1.a, r0.a
mov r0.r, r0.g
mov oC0, r0
