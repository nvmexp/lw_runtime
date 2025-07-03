vs_1_1
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
m4x4 oPos, v0, c[0]
mov oT0, v2
dp4 oT1.x, v0, c[10]
dp4 oT1.y, v0, c[12]
dp4 oT2.x, v0, c[6]
dp4 oT2.y, v0, c[8]
mov oT3, v1
mov oT4, c[4]
mov oD0, c[5]
mov oD1, c[9]
