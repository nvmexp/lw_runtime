ps_3_0

def c0, 1.00000000, 0.00000000, -0.50000000, 0.00000000 ; 0x3f800000 0x80000000 0xbf000000 0x000000
dcl_texcoord0 v0.rg
cmp r0.xy, v0, c0.x, c0.y
add r0.w, -r0.y, r0.x
add r0, r0.w, c0.z
texkill r0
mov oC0, c0.y
;Auto options added
;#PASM_OPTS: -srcalpha 0 -fog 0 -signtex 0 -texrange 0
