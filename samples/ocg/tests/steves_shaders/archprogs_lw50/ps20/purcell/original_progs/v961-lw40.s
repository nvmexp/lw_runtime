vs_2_0
def c[3], 765.005859, 1.000000, 0.000000, 0.000000
dcl_position0 v0
dcl_blendindices0 v1
dcl_normal0 v2
dcl_texcoord0 v3
dcl_tangent0 v5
mul r0.w, v1.z, c[3].x
mova a0.w, r0.w
dp3 r3.x, v5, c[a0.w+42]
dp3 r3.y, v5, c[a0.w+43]
dp3 r3.z, v5, c[a0.w+44]
dp3 r2.x, v2, c[a0.w+42]
dp3 r2.y, v2, c[a0.w+43]
dp3 r2.z, v2, c[a0.w+44]
mul r0.xyz, r3.yzxw, r2.zxyw
mad r0.xyz, r2.yzxw, r3.zxyw, -r0
mov oT4.xyz, r3
mul oT5.xyz, r0, v5.w
dp4 r0.x, v0, c[a0.w+42]
dp4 r0.y, v0, c[a0.w+43]
dp4 r0.z, v0, c[a0.w+44]
mov r0.w, c[3].y
dp4 oPos.x, r0, c[8]
dp4 oPos.y, r0, c[9]
dp4 oPos.w, r0, c[11]
dp4 r3.y, r0, c[10]
mad oFog, -r3.y, c[16].w, c[16].x
mov oPos.z, r3.y
add oT3.xyz, -r0, c[2]
add r3.xyz, -r0, c[29]
add r0.xyz, -r0, c[34]
dp3 r1.z, r3, r3
rsq r0.w, r1.z
mul r3.xyz, r3, r0.w
dst r1.xy, r1.z, r0.w
dp3 r1.x, c[31], r1
rcp r0.w, r1.x
mov r1.x, c[31].w
mad r1.w, r1.z, -r1.x, c[3].y
max r1.w, r1.w, c[3].z
min r1.w, r1.w, c[3].y
mul r0.w, r0.w, r1.w
slt r1.xyz, r2, c[3].z
mova a0.xyz, r1
mul r1.xyz, r2, r2
mul r4.xyz, r1.y, c[a0.y+23]
mad r4.xyz, r1.x, c[a0.x+21], r4
mad r1.xyz, r1.z, c[a0.z+25], r4
dp3 r3.x, r2, r3
max r1.w, r3.x, c[0].x
mul r3.xyz, r1.w, c[27]
mad r1.xyz, r3, r0.w, r1
dp3 r3.z, r0, r0
rsq r0.w, r3.z
mul r0.xyz, r0, r0.w
dst r3.xy, r3.z, r0.w
dp3 r3.x, c[36], r3
rcp r0.w, r3.x
mov r3.x, c[36].w
mad r1.w, r3.z, -r3.x, c[3].y
max r1.w, r1.w, c[3].z
min r1.w, r1.w, c[3].y
mul r0.w, r0.w, r1.w
dp3 r0.x, r2, r0
mov oT6.xyz, r2
max r1.w, r0.x, c[0].x
mul r0.xyz, r1.w, c[32]
mad r0.xyz, r0, r0.w, r1
log r0.x, r0.x
log r0.y, r0.y
log r0.z, r0.z
mul r0.xyz, r0, c[1].x
exp r0.x, r0.x
exp r0.y, r0.y
exp r0.z, r0.z
mul r0.xyz, r0, c[1].w
max r0.w, r0.y, r0.x
max r1.w, r0.z, c[3].y
max r0.w, r0.w, r1.w
rcp r0.w, r0.w
mul oD0.xyz, r0, r0.w
dp4 oT0.x, v3, c[90]
dp4 oT0.y, v3, c[91]
dp4 oT1.x, v3, c[92]
dp4 oT1.y, v3, c[93]
dp4 oT2.x, v3, c[94]
dp4 oT2.y, v3, c[95]
mov oD0.w, c[3].z
mov oD1.xyz, c[3].z
mov oT7.xyz, c[3].z
