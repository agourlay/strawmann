push   rbp
mov    rbp,rsp
cmp    rdx,0x40
jae    10036d1 <sm_dot_f32+0x31>
vxorps xmm1,xmm1,xmm1
xor    ecx,ecx
vxorps xmm6,xmm6,xmm6
vxorps xmm2,xmm2,xmm2
vxorps xmm3,xmm3,xmm3
vxorps xmm5,xmm5,xmm5
vxorps xmm7,xmm7,xmm7
vxorps xmm4,xmm4,xmm4
vxorps xmm0,xmm0,xmm0
jmp    1003792 <sm_dot_f32+0xf2>
vxorps xmm0,xmm0,xmm0
xor    eax,eax
vxorps xmm4,xmm4,xmm4
vxorps xmm7,xmm7,xmm7
vxorps xmm5,xmm5,xmm5
vxorps xmm3,xmm3,xmm3
vxorps xmm2,xmm2,xmm2
vxorps xmm6,xmm6,xmm6
vxorps xmm1,xmm1,xmm1
data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
vmovups ymm8,YMMWORD PTR [rdi+rax*4]
vmovups ymm9,YMMWORD PTR [rdi+rax*4+0x20]
vmovups ymm10,YMMWORD PTR [rdi+rax*4+0x40]
vmovups ymm11,YMMWORD PTR [rdi+rax*4+0x60]
vfmadd231ps ymm7,ymm8,YMMWORD PTR [rsi+rax*4]
vfmadd231ps ymm5,ymm9,YMMWORD PTR [rsi+rax*4+0x20]
vfmadd231ps ymm3,ymm10,YMMWORD PTR [rsi+rax*4+0x40]
vfmadd231ps ymm2,ymm11,YMMWORD PTR [rsi+rax*4+0x60]
vmovups ymm8,YMMWORD PTR [rdi+rax*4+0x80]
vfmadd231ps ymm6,ymm8,YMMWORD PTR [rsi+rax*4+0x80]
vmovups ymm8,YMMWORD PTR [rdi+rax*4+0xa0]
vfmadd231ps ymm1,ymm8,YMMWORD PTR [rsi+rax*4+0xa0]
vmovups ymm8,YMMWORD PTR [rdi+rax*4+0xc0]
vfmadd231ps ymm4,ymm8,YMMWORD PTR [rsi+rax*4+0xc0]
vmovups ymm8,YMMWORD PTR [rdi+rax*4+0xe0]
vfmadd231ps ymm0,ymm8,YMMWORD PTR [rsi+rax*4+0xe0]
lea    rcx,[rax+0x40]
sub    rax,0xffffffffffffff80
cmp    rax,rdx
mov    rax,rcx
jbe    1003700 <sm_dot_f32+0x60>
mov    rax,rcx
or     rax,0x8
cmp    rax,rdx
jbe    10037b0 <sm_dot_f32+0x110>
mov    rax,rcx
jmp    10037cb <sm_dot_f32+0x12b>
data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
vmovups ymm8,YMMWORD PTR [rdi+rcx*4]
vfmadd231ps ymm7,ymm8,YMMWORD PTR [rsi+rcx*4]
lea    rax,[rcx+0x8]
add    rcx,0x10
cmp    rcx,rdx
mov    rcx,rax
jbe    10037b0 <sm_dot_f32+0x110>
vaddps ymm6,ymm6,ymm7
vaddps ymm1,ymm1,ymm5
vaddps ymm3,ymm3,ymm4
vaddps ymm3,ymm3,ymm6
vaddps ymm0,ymm2,ymm0
vaddps ymm0,ymm1,ymm0
vaddps ymm0,ymm0,ymm3
vextractf128 xmm1,ymm0,0x1
vaddps xmm0,xmm0,xmm1
vshufpd xmm1,xmm0,xmm0,0x1
vaddps xmm0,xmm0,xmm1
vmovshdup xmm1,xmm0
vaddss xmm0,xmm0,xmm1
mov    rcx,rax
sub    rcx,rdx
jae    10038af <sm_dot_f32+0x20f>
mov    r8d,edx
sub    r8d,eax
and    r8d,0x7
je     1003833 <sm_dot_f32+0x193>
nop    WORD PTR [rax+rax*1+0x0]
vmovss xmm1,DWORD PTR [rsi+rax*4]
vfmadd231ss xmm0,xmm1,DWORD PTR [rdi+rax*4]
inc    rax
dec    r8
jne    1003820 <sm_dot_f32+0x180>
cmp    rcx,0xfffffffffffffff8
ja     10038af <sm_dot_f32+0x20f>
nop    DWORD PTR [rax+0x0]
vmovss xmm1,DWORD PTR [rsi+rax*4]
vmovss xmm2,DWORD PTR [rsi+rax*4+0x4]
vfmadd132ss xmm1,xmm0,DWORD PTR [rdi+rax*4]
vfmadd231ss xmm1,xmm2,DWORD PTR [rdi+rax*4+0x4]
vmovss xmm0,DWORD PTR [rsi+rax*4+0x8]
vfmadd132ss xmm0,xmm1,DWORD PTR [rdi+rax*4+0x8]
vmovss xmm1,DWORD PTR [rsi+rax*4+0xc]
vfmadd132ss xmm1,xmm0,DWORD PTR [rdi+rax*4+0xc]
vmovss xmm0,DWORD PTR [rsi+rax*4+0x10]
vfmadd132ss xmm0,xmm1,DWORD PTR [rdi+rax*4+0x10]
vmovss xmm1,DWORD PTR [rsi+rax*4+0x14]
vfmadd132ss xmm1,xmm0,DWORD PTR [rdi+rax*4+0x14]
vmovss xmm2,DWORD PTR [rsi+rax*4+0x18]
vfmadd132ss xmm2,xmm1,DWORD PTR [rdi+rax*4+0x18]
vmovss xmm0,DWORD PTR [rsi+rax*4+0x1c]
vfmadd132ss xmm0,xmm2,DWORD PTR [rdi+rax*4+0x1c]
add    rax,0x8
cmp    rdx,rax
jne    1003840 <sm_dot_f32+0x1a0>
pop    rbp
vzeroupper
ret
int3
int3
int3
int3
int3
int3
int3
int3
int3
int3
int3
int3
