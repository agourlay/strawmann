cmp    rdx,0x80
jae    1002d90 <sm_manhattan_f32+0x30>
vxorps xmm2,xmm2,xmm2
xor    ecx,ecx
vxorps xmm7,xmm7,xmm7
vxorps xmm1,xmm1,xmm1
vxorps xmm4,xmm4,xmm4
vxorps xmm5,xmm5,xmm5
vxorps xmm6,xmm6,xmm6
vxorps xmm3,xmm3,xmm3
vxorps xmm0,xmm0,xmm0
jmp    1002eb7 <sm_manhattan_f32+0x157>
vxorps xmm0,xmm0,xmm0
xor    eax,eax
vbroadcastss zmm8,DWORD PTR [rip+0xffffffffffffd5bc]        # 100035c <__anon_3745+0x24>
vxorps xmm3,xmm3,xmm3
vxorps xmm6,xmm6,xmm6
vxorps xmm5,xmm5,xmm5
vxorps xmm4,xmm4,xmm4
vxorps xmm1,xmm1,xmm1
vxorps xmm7,xmm7,xmm7
vxorps xmm2,xmm2,xmm2
nop    DWORD PTR [rax+0x0]
vmovups zmm9,ZMMWORD PTR [rdi+rax*4]
vmovups zmm10,ZMMWORD PTR [rdi+rax*4+0x40]
vmovups zmm11,ZMMWORD PTR [rdi+rax*4+0x80]
vmovups zmm12,ZMMWORD PTR [rdi+rax*4+0xc0]
vsubps zmm9,zmm9,ZMMWORD PTR [rsi+rax*4]
vandps zmm9,zmm9,zmm8
vaddps zmm6,zmm9,zmm6
vsubps zmm9,zmm10,ZMMWORD PTR [rsi+rax*4+0x40]
vandps zmm9,zmm9,zmm8
vaddps zmm5,zmm9,zmm5
vsubps zmm9,zmm11,ZMMWORD PTR [rsi+rax*4+0x80]
vandps zmm9,zmm9,zmm8
vsubps zmm10,zmm12,ZMMWORD PTR [rsi+rax*4+0xc0]
vaddps zmm4,zmm9,zmm4
vandps zmm9,zmm10,zmm8
vmovups zmm10,ZMMWORD PTR [rdi+rax*4+0x100]
vsubps zmm10,zmm10,ZMMWORD PTR [rsi+rax*4+0x100]
vaddps zmm1,zmm9,zmm1
vandps zmm9,zmm10,zmm8
vmovups zmm10,ZMMWORD PTR [rdi+rax*4+0x140]
vsubps zmm10,zmm10,ZMMWORD PTR [rsi+rax*4+0x140]
vaddps zmm7,zmm9,zmm7
vandps zmm9,zmm10,zmm8
vmovups zmm10,ZMMWORD PTR [rdi+rax*4+0x180]
vsubps zmm10,zmm10,ZMMWORD PTR [rsi+rax*4+0x180]
vaddps zmm2,zmm9,zmm2
vandps zmm9,zmm10,zmm8
vmovups zmm10,ZMMWORD PTR [rdi+rax*4+0x1c0]
vsubps zmm10,zmm10,ZMMWORD PTR [rsi+rax*4+0x1c0]
vaddps zmm3,zmm9,zmm3
vandps zmm9,zmm10,zmm8
vaddps zmm0,zmm9,zmm0
lea    rcx,[rax+0x80]
add    rax,0x100
cmp    rax,rdx
mov    rax,rcx
jbe    1002dc0 <sm_manhattan_f32+0x60>
push   rbp
mov    rbp,rsp
mov    rax,rcx
or     rax,0x10
cmp    rax,rdx
jbe    1002ecc <sm_manhattan_f32+0x16c>
mov    rax,rcx
jmp    1002f0a <sm_manhattan_f32+0x1aa>
vbroadcastss zmm8,DWORD PTR [rip+0xffffffffffffd486]        # 100035c <__anon_3745+0x24>
cs nop WORD PTR [rax+rax*1+0x0]
vmovups zmm9,ZMMWORD PTR [rdi+rcx*4]
vsubps zmm9,zmm9,ZMMWORD PTR [rsi+rcx*4]
vandps zmm9,zmm9,zmm8
vaddps zmm6,zmm9,zmm6
lea    rax,[rcx+0x10]
add    rcx,0x20
cmp    rcx,rdx
mov    rcx,rax
jbe    1002ee0 <sm_manhattan_f32+0x180>
vaddps zmm6,zmm7,zmm6
vaddps zmm2,zmm2,zmm5
vaddps zmm3,zmm4,zmm3
vaddps zmm3,zmm3,zmm6
vaddps zmm0,zmm1,zmm0
vaddps zmm0,zmm2,zmm0
vaddps zmm0,zmm0,zmm3
vextractf64x4 ymm1,zmm0,0x1
vaddps zmm0,zmm0,zmm1
vextractf128 xmm1,ymm0,0x1
vaddps xmm0,xmm0,xmm1
vshufpd xmm1,xmm0,xmm0,0x1
vaddps xmm0,xmm0,xmm1
vmovshdup xmm1,xmm0
vaddss xmm0,xmm0,xmm1
mov    rcx,rax
sub    rcx,rdx
jae    1003007 <sm_manhattan_f32+0x2a7>
mov    r8d,edx
sub    r8d,eax
and    r8d,0x3
je     1002f9a <sm_manhattan_f32+0x23a>
vbroadcastss xmm1,DWORD PTR [rip+0xffffffffffffd3df]        # 100035c <__anon_3745+0x24>
nop    DWORD PTR [rax]
vmovss xmm2,DWORD PTR [rdi+rax*4]
vsubss xmm2,xmm2,DWORD PTR [rsi+rax*4]
vandps xmm2,xmm2,xmm1
vaddss xmm0,xmm2,xmm0
inc    rax
dec    r8
jne    1002f80 <sm_manhattan_f32+0x220>
cmp    rcx,0xfffffffffffffffc
ja     1003007 <sm_manhattan_f32+0x2a7>
vbroadcastss xmm1,DWORD PTR [rip+0xffffffffffffd3b3]        # 100035c <__anon_3745+0x24>
nop    DWORD PTR [rax+0x0]
vmovss xmm2,DWORD PTR [rdi+rax*4]
vmovss xmm3,DWORD PTR [rdi+rax*4+0x4]
vsubss xmm2,xmm2,DWORD PTR [rsi+rax*4]
vandps xmm2,xmm2,xmm1
vaddss xmm0,xmm2,xmm0
vsubss xmm2,xmm3,DWORD PTR [rsi+rax*4+0x4]
vmovss xmm3,DWORD PTR [rdi+rax*4+0x8]
vsubss xmm3,xmm3,DWORD PTR [rsi+rax*4+0x8]
vandps xmm2,xmm2,xmm1
vandps xmm3,xmm3,xmm1
vaddss xmm2,xmm3,xmm2
vaddss xmm0,xmm2,xmm0
vmovss xmm2,DWORD PTR [rdi+rax*4+0xc]
vsubss xmm2,xmm2,DWORD PTR [rsi+rax*4+0xc]
vandps xmm2,xmm2,xmm1
vaddss xmm0,xmm2,xmm0
add    rax,0x4
cmp    rdx,rax
jne    1002fb0 <sm_manhattan_f32+0x250>
vxorps xmm0,xmm0,DWORD BCST [rip+0xffffffffffffd347]        # 1000358 <__anon_3745+0x20>
pop    rbp
vzeroupper
ret
cs nop WORD PTR [rax+rax*1+0x0]
