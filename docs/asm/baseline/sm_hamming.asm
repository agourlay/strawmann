cmp    rdx,0x8
jae    1001bdd <sm_hamming+0x1d>
pxor   xmm2,xmm2
xor    ecx,ecx
pxor   xmm1,xmm1
pxor   xmm0,xmm0
pxor   xmm3,xmm3
jmp    1001da6 <sm_hamming+0x1e6>
pxor   xmm4,xmm4
xor    eax,eax
movdqa xmm5,XMMWORD PTR [rip+0xffffffffffffe675]        # 1000260 <__anon_4643+0x20>
movdqa xmm6,XMMWORD PTR [rip+0xffffffffffffe68d]        # 1000280 <__anon_4651+0x10>
movdqa xmm7,XMMWORD PTR [rip+0xffffffffffffe655]        # 1000250 <__anon_4643+0x10>
pxor   xmm3,xmm3
pxor   xmm0,xmm0
pxor   xmm1,xmm1
pxor   xmm2,xmm2
nop    DWORD PTR [rax+rax*1+0x0]
movdqu xmm8,XMMWORD PTR [rdi+rax*8]
movdqu xmm9,XMMWORD PTR [rdi+rax*8+0x10]
movdqu xmm11,XMMWORD PTR [rdi+rax*8+0x20]
movdqu xmm12,XMMWORD PTR [rdi+rax*8+0x30]
movdqu xmm13,XMMWORD PTR [rsi+rax*8]
pxor   xmm13,xmm8
movdqu xmm10,XMMWORD PTR [rsi+rax*8+0x10]
pxor   xmm10,xmm9
movdqu xmm9,XMMWORD PTR [rsi+rax*8+0x20]
pxor   xmm9,xmm11
movdqu xmm8,XMMWORD PTR [rsi+rax*8+0x30]
pxor   xmm8,xmm12
movdqa xmm11,xmm13
psrlw  xmm11,0x1
pand   xmm11,xmm5
psubb  xmm13,xmm11
movdqa xmm11,xmm13
pand   xmm11,xmm6
psrlw  xmm13,0x2
pand   xmm13,xmm6
paddb  xmm13,xmm11
movdqa xmm11,xmm13
psrlw  xmm11,0x4
paddb  xmm11,xmm13
pand   xmm11,xmm7
psadbw xmm11,xmm4
paddq  xmm1,xmm11
movdqa xmm11,xmm10
psrlw  xmm11,0x1
pand   xmm11,xmm5
psubb  xmm10,xmm11
movdqa xmm11,xmm10
pand   xmm11,xmm6
psrlw  xmm10,0x2
pand   xmm10,xmm6
paddb  xmm10,xmm11
movdqa xmm11,xmm10
psrlw  xmm11,0x4
paddb  xmm11,xmm10
pand   xmm11,xmm7
psadbw xmm11,xmm4
paddq  xmm2,xmm11
movdqa xmm10,xmm9
psrlw  xmm10,0x1
pand   xmm10,xmm5
psubb  xmm9,xmm10
movdqa xmm10,xmm9
pand   xmm10,xmm6
psrlw  xmm9,0x2
pand   xmm9,xmm6
paddb  xmm9,xmm10
movdqa xmm10,xmm9
psrlw  xmm10,0x4
paddb  xmm10,xmm9
pand   xmm10,xmm7
psadbw xmm10,xmm4
paddq  xmm0,xmm10
movdqa xmm9,xmm8
psrlw  xmm9,0x1
pand   xmm9,xmm5
psubb  xmm8,xmm9
movdqa xmm9,xmm8
pand   xmm9,xmm6
psrlw  xmm8,0x2
pand   xmm8,xmm6
paddb  xmm8,xmm9
movdqa xmm9,xmm8
psrlw  xmm9,0x4
paddb  xmm9,xmm8
pand   xmm9,xmm7
psadbw xmm9,xmm4
paddq  xmm3,xmm9
lea    rcx,[rax+0x8]
add    rax,0x10
cmp    rax,rdx
mov    rax,rcx
jbe    1001c10 <sm_hamming+0x50>
push   rbp
mov    rbp,rsp
push   r14
push   rbx
mov    rax,rcx
or     rax,0x2
cmp    rax,rdx
jbe    1001dc1 <sm_hamming+0x201>
mov    r8,rcx
jmp    1001e4f <sm_hamming+0x28f>
movdqa xmm4,XMMWORD PTR [rip+0xffffffffffffe497]        # 1000260 <__anon_4643+0x20>
movdqa xmm5,XMMWORD PTR [rip+0xffffffffffffe4af]        # 1000280 <__anon_4651+0x10>
movdqa xmm6,XMMWORD PTR [rip+0xffffffffffffe477]        # 1000250 <__anon_4643+0x10>
pxor   xmm7,xmm7
nop    DWORD PTR [rax]
movdqu xmm8,XMMWORD PTR [rdi+rcx*8]
movdqu xmm9,XMMWORD PTR [rsi+rcx*8]
pxor   xmm9,xmm8
movdqa xmm8,xmm9
psrlw  xmm8,0x1
pand   xmm8,xmm4
psubb  xmm9,xmm8
movdqa xmm8,xmm9
pand   xmm8,xmm5
psrlw  xmm9,0x2
pand   xmm9,xmm5
paddb  xmm9,xmm8
movdqa xmm8,xmm9
psrlw  xmm8,0x4
paddb  xmm8,xmm9
pand   xmm8,xmm6
psadbw xmm8,xmm7
paddq  xmm1,xmm8
lea    r8,[rcx+0x2]
add    rcx,0x4
cmp    rcx,rdx
mov    rcx,r8
jbe    1001de0 <sm_hamming+0x220>
paddq  xmm0,xmm2
paddq  xmm0,xmm3
paddq  xmm0,xmm1
pshufd xmm1,xmm0,0xee
paddq  xmm1,xmm0
movq   rax,xmm1
cmp    r8,rdx
jae    1001ee5 <sm_hamming+0x325>
movabs rcx,0x5555555555555555
movabs r9,0x3333333333333333
movabs r10,0xf0f0f0f0f0f0f0f
movabs r11,0x101010101010101
cs nop WORD PTR [rax+rax*1+0x0]
mov    rbx,QWORD PTR [rsi+r8*8]
xor    rbx,QWORD PTR [rdi+r8*8]
mov    r14,rbx
shr    r14,1
and    r14,rcx
sub    rbx,r14
mov    r14,rbx
and    r14,r9
shr    rbx,0x2
and    rbx,r9
add    rbx,r14
mov    r14,rbx
shr    r14,0x4
add    r14,rbx
and    r14,r10
imul   r14,r11
shr    r14,0x38
add    rax,r14
add    r8,0x1
cmp    rdx,r8
jne    1001ea0 <sm_hamming+0x2e0>
pop    rbx
pop    r14
pop    rbp
ret
nop    WORD PTR [rax+rax*1+0x0]
