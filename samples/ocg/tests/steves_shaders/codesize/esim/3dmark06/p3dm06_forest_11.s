ps_3_0

def c0, 1.00000000, 0.00000000, -0.50000000, 0.00000000 ; 0x3f800000 0x80000000 0xbf000000 0x000000
dcl_texcoord0 v0.rg
dcl_texcoord1 v1.rg
dcl_2d s0
cmp r0.xy, v1, c0.x, c0.y
add r0.w, -r0.y, r0.x
add r0, r0.w, c0.z
texkill r0
texld oC0, v0, s0
;Auto options added
;#PASM_OPTS: -srcalpha 1 -fog 0 -signtex 0 -texrange 3
