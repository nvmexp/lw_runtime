ps_3_0

def c3, -0.50000000, 0.00000000, 1.00000000, -1.44269502 ; 0xbf000000 0x000000 0x3f800000 0xbfb8aa3b
dcl_texcoord0 v0.rg
dcl_2d s0
dcl_2d s1
texld_pp r0, v0, s0
texld_pp r1, v0, s1
add r2.xy, v0, c3.x
add_pp r0.xyz, r0, r1
dp2add r0.w, r2, r2, c3.y
mul r0.xyz, r0, c0
mov r1.y, c3.z
mad r1.xyz, r0.w, -c1, r1.y
mul r0.xyz, r0, r1
mul r0.xyz, r0, c3.w
exp r0.x, r0.x
exp r0.y, r0.y
exp r0.z, r0.z
add r0.xyz, -r0, c3.z
log r0.x, r0.x
log r0.y, r0.y
log r0.z, r0.z
mul r0.xyz, r0, c2.x
exp_pp r0.x, r0.x
exp_pp r0.y, r0.y
exp_pp r0.z, r0.z
mov_pp oC0, r0.xyz
;Auto options added
;#PASM_OPTS: -srcalpha 0 -fog 0 -signtex fffc -texrange 5 -partialtexld 5
