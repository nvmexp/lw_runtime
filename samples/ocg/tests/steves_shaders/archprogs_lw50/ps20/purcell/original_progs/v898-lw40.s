vs_1_1
dcl_position0 v0
dcl_texcoord0 v7
dcl_texcoord1 v8
dp4 r0.x, v0, c[4]
dp4 r0.y, v0, c[5]
dp4 r0.z, v0, c[6]
dp4 r0.w, v0, c[7]
mov oPos, r0
mad oFog, -r0.z, c[16].w, c[16].x
dp4 oT0.x, v7, c[90]
dp4 oT0.y, v7, c[91]
dp4 oT1.x, v7, c[92]
dp4 oT1.y, v7, c[93]
mov oT2, v8
