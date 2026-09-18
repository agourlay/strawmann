push   rbp
mov    rbp,rsp
cmp    rdx,0x100
jae    10028fc <sm_dot_u8u8+0x5c>
vpxor  xmm1,xmm1,xmm1
xor    eax,eax
vpxor  xmm11,xmm11,xmm11
vpxor  xmm6,xmm6,xmm6
vpxor  xmm14,xmm14,xmm14
vpxor  xmm2,xmm2,xmm2
vpxor  xmm8,xmm8,xmm8
vpxor  xmm3,xmm3,xmm3
vpxor  xmm9,xmm9,xmm9
vpxor  xmm4,xmm4,xmm4
vpxor  xmm12,xmm12,xmm12
vpxor  xmm7,xmm7,xmm7
vpxor  xmm15,xmm15,xmm15
vpxor  xmm0,xmm0,xmm0
vpxor  xmm10,xmm10,xmm10
vpxor  xmm5,xmm5,xmm5
vpxor  xmm13,xmm13,xmm13
jmp    1002ac8 <sm_dot_u8u8+0x228>
vpxor  xmm0,xmm0,xmm0
xor    ecx,ecx
vpxor  xmm10,xmm10,xmm10
vpxor  xmm5,xmm5,xmm5
vpxor  xmm13,xmm13,xmm13
vpxor  xmm4,xmm4,xmm4
vpxor  xmm12,xmm12,xmm12
vpxor  xmm7,xmm7,xmm7
vpxor  xmm15,xmm15,xmm15
vpxor  xmm2,xmm2,xmm2
vpxor  xmm8,xmm8,xmm8
vpxor  xmm3,xmm3,xmm3
vpxor  xmm9,xmm9,xmm9
vpxor  xmm1,xmm1,xmm1
vpxor  xmm11,xmm11,xmm11
vpxor  xmm6,xmm6,xmm6
vpxor  xmm14,xmm14,xmm14
cs nop WORD PTR [rax+rax*1+0x0]
vpmovzxbd zmm16,XMMWORD PTR [rdi+rcx*1+0x30]
vpmovzxbd zmm17,XMMWORD PTR [rdi+rcx*1+0x20]
vpmovzxbd zmm18,XMMWORD PTR [rdi+rcx*1]
vpmovzxbd zmm19,XMMWORD PTR [rdi+rcx*1+0x10]
vpmovzxbd zmm20,XMMWORD PTR [rsi+rcx*1+0x30]
vpdpwssd zmm9,zmm20,zmm16
vpmovzxbd zmm16,XMMWORD PTR [rsi+rcx*1+0x20]
vpdpwssd zmm3,zmm16,zmm17
vpmovzxbd zmm16,XMMWORD PTR [rsi+rcx*1]
vpdpwssd zmm2,zmm16,zmm18
vpmovzxbd zmm16,XMMWORD PTR [rsi+rcx*1+0x10]
vpmovzxbd zmm17,XMMWORD PTR [rdi+rcx*1+0x70]
vpmovzxbd zmm18,XMMWORD PTR [rdi+rcx*1+0x60]
vpmovzxbd zmm20,XMMWORD PTR [rdi+rcx*1+0x40]
vpmovzxbd zmm21,XMMWORD PTR [rdi+rcx*1+0x50]
vpmovzxbd zmm22,XMMWORD PTR [rsi+rcx*1+0x70]
vpdpwssd zmm8,zmm16,zmm19
vpdpwssd zmm14,zmm22,zmm17
vpmovzxbd zmm16,XMMWORD PTR [rsi+rcx*1+0x60]
vpmovzxbd zmm17,XMMWORD PTR [rsi+rcx*1+0x40]
vpdpwssd zmm6,zmm16,zmm18
vpdpwssd zmm1,zmm17,zmm20
vpmovzxbd zmm16,XMMWORD PTR [rsi+rcx*1+0x50]
vpmovzxbd zmm17,XMMWORD PTR [rdi+rcx*1+0xb0]
vpmovzxbd zmm18,XMMWORD PTR [rdi+rcx*1+0xa0]
vpmovzxbd zmm19,XMMWORD PTR [rdi+rcx*1+0x80]
vpmovzxbd zmm20,XMMWORD PTR [rdi+rcx*1+0x90]
vpdpwssd zmm11,zmm16,zmm21
vpmovzxbd zmm16,XMMWORD PTR [rsi+rcx*1+0xb0]
vpdpwssd zmm15,zmm16,zmm17
vpmovzxbd zmm16,XMMWORD PTR [rsi+rcx*1+0xa0]
vpdpwssd zmm7,zmm16,zmm18
vpmovzxbd zmm16,XMMWORD PTR [rsi+rcx*1+0x80]
vpdpwssd zmm4,zmm16,zmm19
vpmovzxbd zmm16,XMMWORD PTR [rsi+rcx*1+0x90]
vpmovzxbd zmm17,XMMWORD PTR [rdi+rcx*1+0xf0]
vpmovzxbd zmm18,XMMWORD PTR [rdi+rcx*1+0xe0]
vpmovzxbd zmm19,XMMWORD PTR [rdi+rcx*1+0xc0]
vpdpwssd zmm12,zmm16,zmm20
vpmovzxbd zmm16,XMMWORD PTR [rdi+rcx*1+0xd0]
vpmovzxbd zmm20,XMMWORD PTR [rsi+rcx*1+0xf0]
vpdpwssd zmm13,zmm20,zmm17
vpmovzxbd zmm17,XMMWORD PTR [rsi+rcx*1+0xe0]
vpdpwssd zmm5,zmm17,zmm18
vpmovzxbd zmm17,XMMWORD PTR [rsi+rcx*1+0xc0]
vpdpwssd zmm0,zmm17,zmm19
vpmovzxbd zmm17,XMMWORD PTR [rsi+rcx*1+0xd0]
vpdpwssd zmm10,zmm17,zmm16
lea    rax,[rcx+0x100]
add    rcx,0x200
cmp    rcx,rdx
mov    rcx,rax
jbe    1002950 <sm_dot_u8u8+0xb0>
mov    rcx,rax
or     rcx,0x40
cmp    rcx,rdx
jbe    1002ae0 <sm_dot_u8u8+0x240>
mov    rcx,rax
jmp    1002b46 <sm_dot_u8u8+0x2a6>
nop    DWORD PTR [rax+0x0]
vpmovzxbd zmm16,XMMWORD PTR [rdi+rax*1+0x30]
vpmovzxbd zmm17,XMMWORD PTR [rdi+rax*1+0x20]
vpmovzxbd zmm18,XMMWORD PTR [rdi+rax*1]
vpmovzxbd zmm19,XMMWORD PTR [rdi+rax*1+0x10]
vpmovzxbd zmm20,XMMWORD PTR [rsi+rax*1+0x30]
vpdpwssd zmm9,zmm20,zmm16
vpmovzxbd zmm16,XMMWORD PTR [rsi+rax*1+0x20]
vpdpwssd zmm3,zmm16,zmm17
vpmovzxbd zmm16,XMMWORD PTR [rsi+rax*1]
vpdpwssd zmm2,zmm16,zmm18
vpmovzxbd zmm16,XMMWORD PTR [rsi+rax*1+0x10]
vpdpwssd zmm8,zmm16,zmm19
lea    rcx,[rax+0x40]
sub    rax,0xffffffffffffff80
cmp    rax,rdx
mov    rax,rcx
jbe    1002ae0 <sm_dot_u8u8+0x240>
vpaddd zmm11,zmm12,zmm11
vpaddd zmm10,zmm11,zmm10
vpaddd zmm8,zmm10,zmm8
vpaddd zmm10,zmm15,zmm14
vpaddd zmm10,zmm10,zmm13
vpaddd zmm9,zmm10,zmm9
vpaddd zmm8,zmm8,zmm9
vpaddd zmm1,zmm4,zmm1
vpaddd zmm0,zmm1,zmm0
vpaddd zmm0,zmm0,zmm2
vpaddd zmm1,zmm7,zmm6
vpaddd zmm1,zmm1,zmm5
vpaddd zmm1,zmm1,zmm3
vpaddd zmm0,zmm0,zmm1
vpaddd zmm0,zmm0,zmm8
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
jae    1002cbc <sm_dot_u8u8+0x41c>
mov    r9d,edx
sub    r9d,ecx
and    r9d,0x7
je     1002c09 <sm_dot_u8u8+0x369>
data16 cs nop WORD PTR [rax+rax*1+0x0]
movzx  r10d,BYTE PTR [rdi+rcx*1]
movzx  r11d,BYTE PTR [rsi+rcx*1]
imul   r11d,r10d
add    eax,r11d
inc    rcx
dec    r9
jne    1002bf0 <sm_dot_u8u8+0x350>
cmp    r8,0xfffffffffffffff8
ja     1002cbc <sm_dot_u8u8+0x41c>
data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
movzx  r8d,BYTE PTR [rdi+rcx*1]
movzx  r9d,BYTE PTR [rsi+rcx*1]
imul   r9d,r8d
add    r9d,eax
movzx  eax,BYTE PTR [rdi+rcx*1+0x1]
movzx  r8d,BYTE PTR [rsi+rcx*1+0x1]
imul   r8d,eax
movzx  eax,BYTE PTR [rdi+rcx*1+0x2]
movzx  r10d,BYTE PTR [rsi+rcx*1+0x2]
imul   r10d,eax
add    r10d,r8d
add    r10d,r9d
movzx  eax,BYTE PTR [rdi+rcx*1+0x3]
movzx  r8d,BYTE PTR [rsi+rcx*1+0x3]
imul   r8d,eax
movzx  eax,BYTE PTR [rdi+rcx*1+0x4]
movzx  r9d,BYTE PTR [rsi+rcx*1+0x4]
imul   r9d,eax
add    r9d,r8d
movzx  eax,BYTE PTR [rdi+rcx*1+0x5]
movzx  r8d,BYTE PTR [rsi+rcx*1+0x5]
imul   r8d,eax
add    r8d,r9d
add    r8d,r10d
movzx  eax,BYTE PTR [rdi+rcx*1+0x6]
movzx  r9d,BYTE PTR [rsi+rcx*1+0x6]
imul   r9d,eax
movzx  r10d,BYTE PTR [rdi+rcx*1+0x7]
movzx  eax,BYTE PTR [rsi+rcx*1+0x7]
imul   eax,r10d
add    eax,r9d
add    eax,r8d
add    rcx,0x8
cmp    rdx,rcx
jne    1002c20 <sm_dot_u8u8+0x380>
pop    rbp
vzeroupper
ret
data16 data16 data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
