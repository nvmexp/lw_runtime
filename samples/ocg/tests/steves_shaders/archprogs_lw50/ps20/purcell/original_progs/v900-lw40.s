vs_1_1
dcl_position0 v0
dcl_texcoord0 v7
dcl_texcoord1 v8
dp4 r0.x, v0, c[4]
dp4 r0.y, v0, c[5]
dp4 r0.z, v0, c[6]
dp4 r0.w, v0, c[7]
mov oPos, r0
dp4 r1.z, v0, c[44]
mad oFog, -r0.z, c[16].w, c[16].x
mad oT2, -c[37].x, r1.z, c[37].y
mov oT0, v7
mov oT1, v8
