push   rbp
mov    rbp,rsp
cmp    rdx,0x100
jae    1002b64 <sm_dot_u8i8+0x24>
vpxor  xmm1,xmm1,xmm1
xor    eax,eax
vpxor  xmm2,xmm2,xmm2
vpxor  xmm3,xmm3,xmm3
vpxor  xmm0,xmm0,xmm0
jmp    1002bec <sm_dot_u8i8+0xac>
vpxor  xmm0,xmm0,xmm0
xor    ecx,ecx
vpxor  xmm3,xmm3,xmm3
vpxor  xmm2,xmm2,xmm2
vpxor  xmm1,xmm1,xmm1
cs nop WORD PTR [rax+rax*1+0x0]
vmovups zmm4,ZMMWORD PTR [rdi+rcx*1]
vmovups zmm5,ZMMWORD PTR [rdi+rcx*1+0x40]
vmovups zmm6,ZMMWORD PTR [rdi+rcx*1+0x80]
vmovups zmm7,ZMMWORD PTR [rdi+rcx*1+0xc0]
vmovups zmm8,ZMMWORD PTR [rsi+rcx*1]
vmovups zmm9,ZMMWORD PTR [rsi+rcx*1+0x40]
vmovups zmm10,ZMMWORD PTR [rsi+rcx*1+0x80]
vmovups zmm11,ZMMWORD PTR [rsi+rcx*1+0xc0]
vpdpbusd zmm2,zmm4,zmm8
vpdpbusd zmm1,zmm5,zmm9
vpdpbusd zmm3,zmm6,zmm10
vpdpbusd zmm0,zmm7,zmm11
lea    rax,[rcx+0x100]
add    rcx,0x200
cmp    rcx,rdx
mov    rcx,rax
jbe    1002b80 <sm_dot_u8i8+0x40>
mov    rcx,rax
or     rcx,0x40
cmp    rcx,rdx
jbe    1002c00 <sm_dot_u8i8+0xc0>
mov    rcx,rax
jmp    1002c24 <sm_dot_u8i8+0xe4>
nop    DWORD PTR [rax]
vmovups zmm4,ZMMWORD PTR [rdi+rax*1]
vmovups zmm5,ZMMWORD PTR [rsi+rax*1]
vpdpbusd zmm2,zmm4,zmm5
lea    rcx,[rax+0x40]
sub    rax,0xffffffffffffff80
cmp    rax,rdx
mov    rax,rcx
jbe    1002c00 <sm_dot_u8i8+0xc0>
vpaddd zmm1,zmm3,zmm1
vpaddd zmm0,zmm1,zmm0
vpaddd zmm0,zmm0,zmm2
vextracti64x4 ymm1,zmm0,0x1
vpaddd zmm0,zmm0,zmm1
vextracti128 xmm1,ymm0,0x1
vpaddd xmm0,xmm0,xmm1
vpshufd xmm1,xmm0,0xee
vpaddd xmm0,xmm0,xmm1
vpshufd xmm1,xmm0,0x55
vpaddd xmm0,xmm0,xmm1
vmovd  eax,xmm0
mov    r8,rcx
sub    r8,rdx
jae    1002d4c <sm_dot_u8i8+0x20c>
mov    r9d,edx
sub    r9d,ecx
and    r9d,0x7
je     1002c99 <sm_dot_u8i8+0x159>
nop    DWORD PTR [rax+rax*1+0x0]
movzx  r10d,BYTE PTR [rdi+rcx*1]
movsx  r11d,BYTE PTR [rsi+rcx*1]
imul   r11d,r10d
add    eax,r11d
inc    rcx
dec    r9
jne    1002c80 <sm_dot_u8i8+0x140>
cmp    r8,0xfffffffffffffff8
ja     1002d4c <sm_dot_u8i8+0x20c>
data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
movzx  r8d,BYTE PTR [rdi+rcx*1]
movsx  r9d,BYTE PTR [rsi+rcx*1]
imul   r9d,r8d
add    r9d,eax
movzx  eax,BYTE PTR [rdi+rcx*1+0x1]
movsx  r8d,BYTE PTR [rsi+rcx*1+0x1]
imul   r8d,eax
movzx  eax,BYTE PTR [rdi+rcx*1+0x2]
movsx  r10d,BYTE PTR [rsi+rcx*1+0x2]
imul   r10d,eax
add    r10d,r8d
add    r10d,r9d
movzx  eax,BYTE PTR [rdi+rcx*1+0x3]
movsx  r8d,BYTE PTR [rsi+rcx*1+0x3]
imul   r8d,eax
movzx  eax,BYTE PTR [rdi+rcx*1+0x4]
movsx  r9d,BYTE PTR [rsi+rcx*1+0x4]
imul   r9d,eax
add    r9d,r8d
movzx  eax,BYTE PTR [rdi+rcx*1+0x5]
movsx  r8d,BYTE PTR [rsi+rcx*1+0x5]
imul   r8d,eax
add    r8d,r9d
add    r8d,r10d
movzx  eax,BYTE PTR [rdi+rcx*1+0x6]
movsx  r9d,BYTE PTR [rsi+rcx*1+0x6]
imul   r9d,eax
movzx  r10d,BYTE PTR [rdi+rcx*1+0x7]
movsx  eax,BYTE PTR [rsi+rcx*1+0x7]
imul   eax,r10d
add    eax,r9d
add    eax,r8d
add    rcx,0x8
cmp    rdx,rcx
jne    1002cb0 <sm_dot_u8i8+0x170>
pop    rbp
vzeroupper
ret
data16 data16 data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
