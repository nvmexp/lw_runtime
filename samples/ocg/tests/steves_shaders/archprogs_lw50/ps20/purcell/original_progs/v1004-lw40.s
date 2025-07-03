vs_1_1
dcl_position0 v0
dcl_psize0 v1
dcl_color0 v2
m4x4 oPos, v0, c[0]
dp4 r0.z, v0, c[4]
rcp r0.z, r0.z
mul r1.x, v1.x, r0.z
mul r1.x, c[5].z, r1.x
max r2.x, r1.x, c[5].x
min oPts, r2.x, c[5].y
mov oD0, v2
