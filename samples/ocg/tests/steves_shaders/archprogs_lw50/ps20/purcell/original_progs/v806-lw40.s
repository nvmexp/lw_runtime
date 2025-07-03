vs_1_1
dcl_position0 v0
dcl_texcoord0 v1
dcl_texcoord2 v2
m4x4 oPos, v0, c[0]
mov oT0, v1
mov oT1, v2
mov oT2, c[6]
mov oT3, c[7]
mov oT4, c[8]
dp4 r0.x, v0, c[3]
add r0.x, r0.x, -c[4].x
mul oT5, r0.x, c[4].y
