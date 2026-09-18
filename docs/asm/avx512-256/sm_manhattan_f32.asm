cmp    rdx,0x40
jae    1002dcd <sm_manhattan_f32+0x2d>
vxorps xmm2,xmm2,xmm2
xor    ecx,ecx
vxorps xmm7,xmm7,xmm7
vxorps xmm1,xmm1,xmm1
vxorps xmm4,xmm4,xmm4
vxorps xmm5,xmm5,xmm5
vxorps xmm6,xmm6,xmm6
vxorps xmm3,xmm3,xmm3
vxorps xmm0,xmm0,xmm0
jmp    1002ed2 <sm_manhattan_f32+0x132>
vxorps xmm0,xmm0,xmm0
xor    eax,eax
vbroadcastss ymm8,DWORD PTR [rip+0xffffffffffffd46c]        # 1000248 <__init_array_end+0x248>
vxorps xmm3,xmm3,xmm3
vxorps xmm6,xmm6,xmm6
vxorps xmm5,xmm5,xmm5
vxorps xmm4,xmm4,xmm4
vxorps xmm1,xmm1,xmm1
vxorps xmm7,xmm7,xmm7
vxorps xmm2,xmm2,xmm2
nop    DWORD PTR [rax+rax*1+0x0]
vmovups ymm9,YMMWORD PTR [rdi+rax*4]
vmovups ymm10,YMMWORD PTR [rdi+rax*4+0x20]
vmovups ymm11,YMMWORD PTR [rdi+rax*4+0x40]
vmovups ymm12,YMMWORD PTR [rdi+rax*4+0x60]
vsubps ymm9,ymm9,YMMWORD PTR [rsi+rax*4]
vandps ymm9,ymm9,ymm8
vaddps ymm6,ymm9,ymm6
vsubps ymm9,ymm10,YMMWORD PTR [rsi+rax*4+0x20]
vandps ymm9,ymm9,ymm8
vaddps ymm5,ymm9,ymm5
vsubps ymm9,ymm11,YMMWORD PTR [rsi+rax*4+0x40]
vandps ymm9,ymm9,ymm8
vsubps ymm10,ymm12,YMMWORD PTR [rsi+rax*4+0x60]
vaddps ymm4,ymm9,ymm4
vandps ymm9,ymm10,ymm8
vmovups ymm10,YMMWORD PTR [rdi+rax*4+0x80]
vsubps ymm10,ymm10,YMMWORD PTR [rsi+rax*4+0x80]
vaddps ymm1,ymm9,ymm1
vandps ymm9,ymm10,ymm8
vmovups ymm10,YMMWORD PTR [rdi+rax*4+0xa0]
vsubps ymm10,ymm10,YMMWORD PTR [rsi+rax*4+0xa0]
vaddps ymm7,ymm9,ymm7
vandps ymm9,ymm10,ymm8
vmovups ymm10,YMMWORD PTR [rdi+rax*4+0xc0]
vsubps ymm10,ymm10,YMMWORD PTR [rsi+rax*4+0xc0]
vaddps ymm2,ymm9,ymm2
vandps ymm9,ymm10,ymm8
vmovups ymm10,YMMWORD PTR [rdi+rax*4+0xe0]
vsubps ymm10,ymm10,YMMWORD PTR [rsi+rax*4+0xe0]
vaddps ymm3,ymm9,ymm3
vandps ymm9,ymm10,ymm8
vaddps ymm0,ymm9,ymm0
lea    rcx,[rax+0x40]
sub    rax,0xffffffffffffff80
cmp    rax,rdx
mov    rax,rcx
jbe    1002e00 <sm_manhattan_f32+0x60>
push   rbp
mov    rbp,rsp
mov    rax,rcx
or     rax,0x8
cmp    rax,rdx
jbe    1002ee7 <sm_manhattan_f32+0x147>
mov    rax,rcx
jmp    1002f13 <sm_manhattan_f32+0x173>
vbroadcastss ymm8,DWORD PTR [rip+0xffffffffffffd358]        # 1000248 <__init_array_end+0x248>
vmovups ymm9,YMMWORD PTR [rdi+rcx*4]
vsubps ymm9,ymm9,YMMWORD PTR [rsi+rcx*4]
vandps ymm9,ymm9,ymm8
vaddps ymm6,ymm9,ymm6
lea    rax,[rcx+0x8]
add    rcx,0x10
cmp    rcx,rdx
mov    rcx,rax
jbe    1002ef0 <sm_manhattan_f32+0x150>
vaddps ymm6,ymm7,ymm6
vaddps ymm2,ymm2,ymm5
vaddps ymm3,ymm4,ymm3
vaddps ymm3,ymm3,ymm6
vaddps ymm0,ymm1,ymm0
vaddps ymm0,ymm2,ymm0
vaddps ymm0,ymm0,ymm3
vextractf128 xmm1,ymm0,0x1
vaddps xmm0,xmm0,xmm1
vshufpd xmm1,xmm0,xmm0,0x1
vaddps xmm0,xmm0,xmm1
vmovshdup xmm1,xmm0
vaddss xmm0,xmm0,xmm1
mov    rcx,rax
sub    rcx,rdx
jae    1002ff7 <sm_manhattan_f32+0x257>
mov    r8d,edx
sub    r8d,eax
and    r8d,0x3
je     1002f8a <sm_manhattan_f32+0x1ea>
vbroadcastss xmm1,DWORD PTR [rip+0xffffffffffffd2dd]        # 1000248 <__init_array_end+0x248>
nop    DWORD PTR [rax+rax*1+0x0]
vmovss xmm2,DWORD PTR [rdi+rax*4]
vsubss xmm2,xmm2,DWORD PTR [rsi+rax*4]
vandps xmm2,xmm2,xmm1
vaddss xmm0,xmm2,xmm0
inc    rax
dec    r8
jne    1002f70 <sm_manhattan_f32+0x1d0>
cmp    rcx,0xfffffffffffffffc
ja     1002ff7 <sm_manhattan_f32+0x257>
vbroadcastss xmm1,DWORD PTR [rip+0xffffffffffffd2af]        # 1000248 <__init_array_end+0x248>
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
jne    1002fa0 <sm_manhattan_f32+0x200>
vxorps xmm0,xmm0,DWORD BCST [rip+0xffffffffffffd243]        # 1000244 <__init_array_end+0x244>
pop    rbp
vzeroupper
ret
cs nop WORD PTR [rax+rax*1+0x0]
