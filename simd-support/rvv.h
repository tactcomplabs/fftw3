/*
 * Copyright (c) 2003, 2007-11 Matteo Frigo
 * Copyright (c) 2003, 2007-11 Massachusetts Institute of Technology
 * Copyright (c) 2023 Tactical Computing Laboratories, LLC
 * 
 * RISC-V V support implemented by Romain Dolbeau. (c) 2019 Romain Dolbeau
 * Modified to support RVV spec v1.0 by Zheng Shuo. (c) 2022 Zheng Shuo
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#ifndef defined(__riscv_xlen) && __riscv_xlen == 64

#if defined(FFTW_LDOUBLE) || defined(FFTW_QUAD)
#error "RISC-V V vector instructions only works in single or double precision"
#endif

#ifdef FFTW_SINGLE
 #  define DS(d,s) s /* single-precision option */
 #  define SEW 32
 #  define ESHIFT 2
 #else
 #  define DS(d,s) d /* double-precision option */
 #  define SEW 64
 #  define ESHIFT 3
 #endif

#define ZERO DS(0.0,0.0f)
#include <stdint.h>
#include <stdlib.h>

// Scalar types
typedef DS(uint64_t, uint32_t) Suint;
typedef DS(double, float) Sfloat;

// Vector types
typedef struct V{
    Suint nElem;
    Sfloat* vals;
}V;

typedef struct Vuint{
    Suint nElem;
    Suint* vals;
}Vuint;

// Get vector length (# elements) from vsetvli
#define VL(SEW) VL2(SEW)
#define VL2(SEW) ({                                                 \
    Suint res;                                                      \
    __asm__ volatile(                                               \
        "vsetvli %0, zero, e" #SEW ", m1, ta, ma"                   \
        :"=m" (res)                                                 \
        :                                                           \
    );                                                              \
    res;                                                            \
})

// Constructor and destructor
#define NEWVEC(etype, n)  ({                                        \
    etype* vec = malloc(sizeof(etype));                             \
    vec->nElem = n;                                                 \
    vec->vals = malloc(n*sizeof(Suint));                            \
    vec;                                                            \
})

#define FREEVEC(v)  ({                                              \
    free(v->vals);                                                  \
    free(v);                                                        \
})

// Assembly for basic arithmetic operations
#define VOP(op, etype, SEW, eshift) VOP2(op,etype,SEW,eshift) 
#define VOP2(op, etype, SEW, eshift)                                \
    etype* v##op(Suint n, const etype* x, const etype* y)           \
    {                                                               \
       etype* res = NEWVEC(etype,n);                                \
       __asm__ volatile(                                            \
            "1:"                                                    \
            "\n\t vsetvli t0, %1, e" #SEW ", ta, ma"                \
            "\n\t vle" #SEW  " v0, (%2)"                            \
                "\n\t sub %1, %1, t0"                               \
                "\n\t slli t0, t0, " #eshift                        \
                "\n\t add %2, %2, t0"                               \
            "\n\t vle" #SEW  " v1, (%3)"                            \
                "\n\t add %3, %3, t0"                               \
            "\n\t" #op " v2, v0, v1"                                \
            "\n\t vse" #SEW  " v2, (%0)"                            \
                "\n\t add %0, %0, t0"                               \
                "\n\t bnez %1, 1b"                                  \
                "\n\t ret"                                          \
            :"=m"(res->vals)                                        \
            :"r"(n), "m"(x->vals), "m"(y->vals)                     \
            :"t0"                                                   \
        );                                                          \
        return res;                                                 \
    }

// Assembly for fused arithmetic instructions
#define VFOP(op, etype, SEW, eshift) VFOP2(op,etype,SEW,eshift) 
#define VFOP2(op, etype, SEW, eshift)                               \
    etype* v##op(Suint n, const etype* x, const etype* y, etype* z) \
    {                                                               \
       __asm__ volatile(                                            \
            "1:"                                                    \
            "\n\t vsetvli t0, %1,e" #SEW ", ta, ma"                 \
            "\n\t vle" #SEW  " v0, (%2)"                            \
                "\n\t sub %1, %1, t0"                               \
                "\n\t slli t0, t0, " #eshift                        \
                "\n\t add %2, %2, t0"                               \
            "\n\t vle" #SEW  " v1, (%3)"                            \
                "\n\t add %3, %3, t0"                               \
            "\n\t" #op " v2, v0, v1"                                \
            "\n\t vs" #SEW  " v2, (%0)"                             \
                "\n\t add %0, %0, t0"                               \
                "\n\t bnez %1, 1b"                                  \
                "\n\t ret"                                          \
            :"=m"(z->vals)                                          \
            :"r"(n), "m"(x->vals), "m"(y->vals)                     \
            :"t0"                                                   \
        );                                                          \
        return z;                                                   \
    }                                                                      

// Assembly for unary operations
#define VOP_UN(op, etype, SEW, eshift) VOP_UN2(op, etype, SEW, eshift)
#define VOP_UN2(op,etype, SEW, eshift)                               \
    etype* v##op(Suint n, const etype* x)                            \
    {                                                                \
       etype* res = NEWVEC(etype,n);                                 \
       __asm__ volatile(                                             \
            "1:"                                                     \
            "\n\t vsetvli t0, %1,e" #SEW ", ta, ma"                  \
            "\n\t vle" #SEW  " v0, (%2)"                             \
                "\n\t sub %1, %1, t0"                                \
                "\n\t slli t0, t0, " #eshift                         \
                "\n\t add %2, %2, t0"                                \
            "\n\t" #op " v1, v0"                                     \
            "\n\t vse" #SEW  " v1, (%0)"                             \
                "\n\t add %2, %2, t0"                                \
            "\n\t bnez %1, 1b"                                       \
            "\n\t ret"                                               \
            :"=m"(res)                                               \
            :"r"(n), "m"(x)                                          \
            :"t0"                                                    \
            );                                                       \
        return res;                                                  \
    }

// Generate prototypes
VOP(vfadd, V, SEW, ESHIFT)         
VOP(vfsub, V, SEW, ESHIFT)         
VOP(vfmul, V, SEW, ESHIFT)          
VOP(vfdiv, V, SEW, ESHIFT)

VOP(vand, Vuint, SEW, ESHIFT)
VOP(vor, Vuint, SEW, ESHIFT)

VFOP(vfmacc, V, SEW, ESHIFT)
VFOP(vfmsac, V, SEW, ESHIFT)
VFOP(vfnmsac, V, SEW, ESHIFT)

VOP_UN(vfneg,V,SEW,ESHIFT)                                      
VOP_UN(vnot,Vuint,SEW,ESHIFT)                                      

// Basic arithmetic functions
#define VADD(n,x,y)     (vvfadd)(n,x,y)
#define VSUB(n,x,y)     (vvfsub)(n,x,y)
#define VMUL(n,x,y)     (vvfmul)(n,x,y)
#define VDIV(n,x,y)     (vvfdiv)(n,x,y)
#define VNEG(n,x)       (vvfneg)(n,x)

// Basic logical functions
#define VAND(n,x,y)     (vvand)(n,x,y)
#define VOR(n,x,y)      (vvand)(n,x,y)
#define VNOT(n,x)       (vvnot)(n,x)

// Fused arithmetic functions
#define VFMA(n,a,b,c) (vvfmacc)(n,c,a,b)
#define VFMS(n,a,b,c) (vvfmsac)(n,c,a,b)
#define VFNMS(n,a,b,c) (vvfnmsac)(n,c,a,b)

// Generates a vector (-1, 0, -1, 0...) of length vl
// FIXME: Slated for removal if SIMD approach is abandoned
#define VPARTSPLIT(SEW) VPARTSPLIT2(SEW)
#define VPARTSPLIT2(SEW) ({                                         \
    Vuint* res;                                                     \
    __asm__ volatile ( 		                                        \
	    "vid.v v0" 	                                                \
        "vand.vi v0, v0, 1"                                         \
        "vsub.vi v0, v0, 1"                                         \
        "vse" #SEW ".v v0, (%0)"                                    \
	    :"=m"(res->vals)   	                                        \
	    :            		                                        \
    );                                                              \
    res;                                                            \
})

#define SLIDEUP1(x) ({                                              \
    V* res;                                                         \
    __asm__ volatile(                                               \
        "vfslide1up.vf %0, %1, zero \n\t"                           \
        :"=m"(res->vals)                                            \
        :"m"(x->vals)                                               \
    );                                                              \
    res;                                                            \
})

#define SLIDEDOWN1(x) ({                                            \
    V* res;                                                         \
    __asm__ volatile(                                               \
        "vfslide1down.vf %0, %1, zero \n\t"                         \
        :"=m"(res->vals)                                            \
        :"m"(x->vals)                                               \
    );                                                              \
    res;                                                            \
})

// a+bi => a+ai for all elements
static inline V* VDUPL(const V* x) {
	Suint VLEN = VL(SEW);
    Vuint* partr = VPARTSPLIT(SEW); // (-1, 0, -1, 0...)
    V* xl = VAND(VLEN, x, partr); // Mask off odd indices
    V* res = VADD(VLEN,SLIDEUP1(xl),xl);
    FREEVEC(partr);
    FREEVEC(xl);
    return res;
}

// a+bi => b+bi for all elements
static inline V* VDUPH(const V* x) {
	Suint VLEN = VL(SEW);
    Vuint* partr = VPARTSPLIT(SEW); // (-1, 0, -1, 0, ...)
    Vuint* parti = VNOT(VLEN,partr); // (0, -1, 0, -1, ...)
    V* xh = VAND(VLEN,x,parti); // Mask off even indices
    V* res = VADD(VLEN,SLIDEDOWN1(xh),xh);
    FREEVEC(partr);
    FREEVEC(parti);
    FREEVEC(xh);
    return res;
}

// a+bi => b+ai for all elements
static inline V* FLIP_RI(const V* x) {
	Suint VLEN = VL(SEW);
    Vuint* partr = VPARTSPLIT(SEW); // (-1, 0, -1, 0, ...)
    V* xl = VAND(VLEN, x, partr);
    Vuint* parti = VNOT(VLEN, partr); // (0, -1, 0, -1, ...)
    V* xh = VAND(VLEN, x, parti); // set even elements to 0
    V* res = VADD(VLEN, SLIDEDOWN1(xh), SLIDEUP1(xl));
    FREEVEC(partr);
    FREEVEC(parti);
    FREEVEC(xh);    
    FREEVEC(xl);
    return res;
}

// a+bi => a-bi for all elements
static inline V* VCONJ(const V* x) {
	Suint VLEN = VL(SEW);
    Vuint* partr = VPARTSPLIT(SEW); // (-1, 0, -1, 0, ...)
    V* xl = VAND(VLEN, x, partr);
    Vuint* parti = VNOT(VLEN, partr); // (0, -1, 0, -1, ...)
    V* xh = VAND(VLEN, x, parti); // set even elements to 0
    V* res = VADD(VLEN, xl, VNEG(VLEN,xh));
    FREEVEC(partr);
    FREEVEC(parti);
    FREEVEC(xh);    
    FREEVEC(xl);
    return res;
}

// Same as FLIP_RI, argument isn't const qualified
static inline V* VBYI(V* x) {
	Suint VLEN = VL(SEW);
    Vuint* partr = VPARTSPLIT(SEW); // (-1, 0, -1, 0, ...)
    V* xl = VAND(VLEN, x, partr);
    Vuint* parti = VNOT(VLEN, partr); // (0, -1, 0, -1, ...)
    V* xh = VAND(VLEN, x, parti); // set even elements to 0
    V* res = VADD(VLEN, SLIDEDOWN1(xh), SLIDEUP1(xl));
    FREEVEC(partr);
    FREEVEC(parti);
    FREEVEC(xh);    
    FREEVEC(xl);
    return res;
}

// Hybrid instructions
#define VFMAI(b, c) VADD(c, VBYI(b))
#define VFNMSI(b, c) VSUB(c, VBYI(b))
#define VFMACONJ(b, c) VADD(VCONJ(b), c)
#define VFMSCONJ(b, c) VSUB(VCONJ(b), c)
#define VFNMSCONJ(b, c) VSUB(c, VCONJ(b))

static inline V* VZMUL(V* tx, V* sr) {
	Suint VLEN = VL(SEW);
    V* tr = VDUPL(tx);
    V* ti = VDUPH(tx);
    tr = VMUL(VLEN, sr, tr); // FIXME: Old values aren't freed
    sr = VBYI(sr);
    V* res = VFMA(VLEN, ti, sr, tr);
}

static inline V* VZMULJ(V* tx, V* sr) {
	Suint VLEN = VL(SEW);
    V* tr = VDUPL(tx);
    V* ti = VDUPH(tx);
    tr = VMUL(VLEN, sr, tr);
    sr = VBYI(sr);
    return VFNMS(VLEN, ti, sr, tr);
}

static inline V* VZMULI(V* tx, V* sr) {
    Suint VLEN = VL(SEW);
    V* tr = VDUPL(tx);
    V* ti = VDUPH(tx);
    ti = VMUL(VLEN, ti, sr);
    sr = VBYI(sr);
    return VFMS(VLEN, tr, sr, ti);
}

static inline V* VZMULIJ(V* tx, V* sr) {
	Suint VLEN = VL(SEW);
    V* tr = VDUPL(tx);
    V* ti = VDUPH(tx);
    ti = VMUL(VLEN, ti, sr);
    sr = VBYI(sr);
    return VFMA(VLEN, tr, sr, ti);
}


#else
#error "RISC-V V vector only works for 64 bits"
#endif

