vs_2_0
def c[1], 765.005859, 0.000000, 0.000000, 0.000000
dcl_position0 v0
dcl_blendindices0 v1
dcl_normal0 v2
dcl_texcoord0 v3
dcl_tangent0 v5
mul r0.w, v1.z, c[1].x
mova a0.w, r0.w
dp3 r1.x, v5, c[a0.w+42]
dp3 r1.y, v5, c[a0.w+43]
dp3 r1.z, v5, c[a0.w+44]
dp3 r2.x, v2, c[a0.w+42]
dp3 r2.y, v2, c[a0.w+43]
dp3 r2.z, v2, c[a0.w+44]
mul r0.xyz, r1.yzxw, r2.zxyw
mad r0.xyz, r2.yzxw, r1.zxyw, -r0
mov oT4.xyz, r1
dp4 r1.x, v0, c[a0.w+42]
dp4 r1.y, v0, c[a0.w+43]
dp4 r1.z, v0, c[a0.w+44]
mov r1.w, c[0].y
mul oT5.xyz, r0, v5.w
dp4 oPos.x, r1, c[8]
dp4 oPos.y, r1, c[9]
add r0.xy, -r1.z, c[2].wzzw
max r0.w, r0.x, c[0].x
rcp r2.w, r0.y
dp4 r0.y, r1, c[10]
mul r0.w, r0.w, r2.w
dp4 oPos.w, r1, c[11]
mul r0.w, r0.y, r0.w
mov oPos.z, r0.y
mad oFog, -r0.w, c[16].w, c[16].y
add oT3.xyz, -r1, c[2]
add r4.xyz, -r1, c[29]
dp3 r0.x, r4, r4
mov r5.xz, c[0].y
rsq r5.y, r0.x
mov r0.yw, c[0].y
add r3.xyz, -r1, c[34]
mul r1.xyz, r0.yxxw, r5
mov r6.z, c[0].y
mad r1.w, r0.x, -c[31].w, r6.z
mul r0.xyz, r4, r5.y
dp3 r1.x, r1, c[31]
max r2.w, r1.w, c[0].x
rcp r1.w, r1.x
min r2.w, r2.w, c[0].y
mul r1.w, r1.w, r2.w
dp3 r4.x, c[28], -r0
mul r1.xyz, r1.w, c[27]
add r1.w, r4.x, -c[30].z
dp3 r6.x, r2, r0
mul r1.w, r1.w, c[30].w
max r2.w, r1.w, c[0].x
slt r0.xyz, r2, c[0].x
mova a0.xyz, r0
mul r0.xyz, r2, r2
pow r1.w, r2.w, c[30].x
mul r4.xyz, r0.y, c[a0.y+23]
min r1.w, r1.w, c[0].y
mad r4.xyz, r0.x, c[a0.x+21], r4
mul r1.xyz, r1, r1.w
mad r5.xyz, r0.z, c[a0.z+25], r4
max r1.w, r6.x, c[0].x
dp3 r0.z, r3, r3
rsq r4.y, r0.z
mov r4.xz, c[0].y
mad r1.xyz, r1, r1.w, r5
mul r5.xyz, r0.wzzw, r4
mad r0.w, r0.z, -c[36].w, r6.z
dp3 r0.x, r5, c[36]
mul r3.xyz, r3, r4.y
rcp r1.w, r0.x
max r0.w, r0.w, c[0].x
dp3 r0.x, c[33], -r3
min r2.w, r0.w, c[0].y
add r0.w, r0.x, -c[35].z
mul r1.w, r1.w, r2.w
mul r0.w, r0.w, c[35].w
mul r0.xyz, r1.w, c[32]
max r1.w, r0.w, c[0].x
dp3 r3.x, r2, r3
pow r0.w, r1.w, c[35].x
mov oT6.xyz, r2
min r0.w, r0.w, c[0].y
mul r0.xyz, r0, r0.w
max r0.w, r3.x, c[0].x
mad oD0.xyz, r0, r0.w, r1
mul r0.xy, v3, c[90]
add r0.w, r0.y, r0.x
mul r0.xy, v3, c[91]
mov oT0.x, r0.w
add r0.z, r0.y, r0.x
mov oT0.y, r0.z
mov oT1.x, r0.w
mov oT2.x, r0.w
mov oT1.y, r0.z
mov oT2.y, r0.z
mov oD0.w, c[0].x
mov oD1, c[0].x
mov oT7.xyz, c[0].x
