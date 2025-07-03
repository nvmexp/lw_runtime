vs_2_0
def c[7], 4.000000, 0.000000, 0.000000, -2.000000
dcl_position0 v0
dcl_normal0 v1
dcl_tangent0 v2
dcl_binormal0 v3
dcl_texcoord0 v4
dcl_color0 v5
m4x4 oPos, v0, c[0]
m4x3 oT1.xyz, v0, c[4]
mad r7.xyz, v2, c[7].x, c[7].w
mad r8.xyz, v3, c[7].x, c[7].w
mad r9.xyz, v1, c[7].x, c[7].w
m3x3 oT2.xyz, c[4], r7
m3x3 oT3.xyz, c[5], r7
m3x3 oT4.xyz, c[6], r7
mov oD0, v5
mov oT0.xy, v4
