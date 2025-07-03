vs_2_0
def c[7], -1.000000, 0.000000, 0.000000, 0.000000
def c[8], 2.000000, -1.000000, 0.000000, 0.000000
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
dcl_color0 v3
m4x4 oPos, v0, c[0]
m4x3 oT1.xyz, v0, c[4]
mad r7.xyz, c[8].x, v1, c[8].y
m3x3 oT2.xyz, r7, c[4]
mov oD0, v3
mov oT0.xy, v2
