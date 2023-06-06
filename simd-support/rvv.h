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

#ifndef __riscv_xlen && __riscv_xlen == 64

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
#define VL2(SEW) ({                                                         \
    Suint res;                                                              \
    __asm__ volatile(                                                       \
        "vsetvli %0, zero, e" #SEW ", m1, ta, ma"                           \
        :"=m" (res)                                                         \
        :                                                                   \
    );                                                                      \
    res;                                                                    \
})

// Constructor/destructor
static inline void newVector(const Suint n, void* vec) {
    V* tmp = (V*)vec;
    tmp->nElem = n;
    tmp->vals = calloc(n, SEW);
}

static inline void freeVector(void* vec) {
    V* tmp = (V*)vec;
    free(tmp->vals);
}

// Assembly for basic arithmetic operations
#define VOP(op, etype, SEW, eshift) VOP2(op,etype,SEW,eshift) 
#define VOP2(op, etype, SEW, eshift)                                        \
    void v##op(const Suint n, const etype* x, const etype* y, etype* z)     \
    {                                                                       \
       __asm__ volatile(                                                    \
            "1:"                                                            \
            "\n\t vsetvli t0, %1, e" #SEW ", ta, ma"                        \
            "\n\t vle" #SEW  " v0, (%2)"                                    \
                "\n\t sub %1, %1, t0"                                       \
                "\n\t slli t0, t0, " #eshift                                \
                "\n\t add %2, %2, t0"                                       \
            "\n\t vle" #SEW  " v1, (%3)"                                    \
                "\n\t add %3, %3, t0"                                       \
            "\n\t" #op " v2, v0, v1"                                        \
            "\n\t vse" #SEW  " v2, (%0)"                                    \
                "\n\t add %0, %0, t0"                                       \
                "\n\t bnez %1, 1b"                                          \
                "\n\t ret"                                                  \
            :"=m"(z->vals)                                                  \
            :"r"(n), "m"(x->vals), "m"(y->vals)                             \
            :"t0"                                                           \
        );                                                                  \
    }

// Assembly for fused arithmetic instructions
#define VFOP(op, etype, SEW, eshift) VFOP2(op,etype,SEW,eshift) 
#define VFOP2(op, etype, SEW, eshift)                                       \
    void v##op(Suint n, const etype* x, const etype* y, etype* z)           \
    {                                                                       \
       __asm__ volatile(                                                    \
            "1:"                                                            \
            "\n\t vsetvli t0, %1,e" #SEW ", ta, ma"                         \
            "\n\t vle" #SEW  " v0, (%2)"                                    \
                "\n\t sub %1, %1, t0"                                       \
                "\n\t slli t0, t0, " #eshift                                \
                "\n\t add %2, %2, t0"                                       \
            "\n\t vle" #SEW  " v1, (%3)"                                    \
                "\n\t add %3, %3, t0"                                       \
            "\n\t" #op " v2, v0, v1"                                        \
            "\n\t vs" #SEW  " v2, (%0)"                                     \
                "\n\t add %0, %0, t0"                                       \
                "\n\t bnez %1, 1b"                                          \
                "\n\t ret"                                                  \
            :"=m"(z->vals)                                                  \
            :"r"(n), "m"(x->vals), "m"(y->vals)                             \
            :"t0"                                                           \
        );                                                                  \
    }                                                                      

// Assembly for unary operations
#define VOP_UN(op, etype, SEW, eshift) VOP_UN2(op, etype, SEW, eshift)
#define VOP_UN2(op,etype, SEW, eshift)                                      \
    void v##op(const Suint n, const etype* x, etype* y)                     \
    {                                                                       \
       __asm__ volatile(                                                    \
            "1:"                                                            \
            "\n\t vsetvli t0, %1,e" #SEW ", ta, ma"                         \
            "\n\t vle" #SEW  " v0, (%2)"                                    \
                "\n\t sub %1, %1, t0"                                       \
                "\n\t slli t0, t0, " #eshift                                \
                "\n\t add %2, %2, t0"                                       \
            "\n\t" #op " v1, v0"                                            \
            "\n\t vse" #SEW  " v1, (%0)"                                    \
                "\n\t add %2, %2, t0"                                       \
            "\n\t bnez %1, 1b"                                              \
            "\n\t ret"                                                      \
            :"=m"(y->vals)                                                  \
            :"r"(n), "m"(x->vals)                                           \
            :"t0"                                                           \
            );                                                              \
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

VOP_UN(vfneg, V, SEW, ESHIFT)                                      
VOP_UN(vnot, Vuint, SEW, ESHIFT)                                      

// Macro wrappers for arithmetic functions
#define VADD(n,x,y,z)     (vvfadd)(n,x,y,z)
#define VSUB(n,x,y,z)     (vvfsub)(n,x,y,z)
#define VMUL(n,x,y,z)     (vvfmul)(n,x,y,z)
#define VDIV(n,x,y,z)     (vvfdiv)(n,x,y,z)
#define VNEG(n,x,y)       (vvfneg)(n,x,y)

// Macro wrappers for logical functions
#define VAND(n,x,y,z)     (vvand)(n,x,y,z)
#define VOR(n,x,y,z)      (vvand)(n,x,y,z)
#define VNOT(n,x,y)       (vvnot)(n,x,y)

// Macro wrappers for fused arithmetic functions
#define VFMA(n,a,b,c)     (vvfmacc)(n,c,a,b)
#define VFMS(n,a,b,c)     (vvfmsac)(n,c,a,b)
#define VFNMS(n,a,b,c)    (vvfnmsac)(n,c,a,b)

// Generates a vector (-1, 0, -1, 0...) of length vl
// FIXME: Slated for removal if SIMD approach is abandoned
#define VPARTSPLIT(VLEN, SEW, dest) VPARTSPLIT2(VLEN, SEW, dest)
#define VPARTSPLIT2(VLEN, SEW, dest) ({                             \
    __asm__ volatile ( 		                                        \
	    "vid.v v0" 	                                                \
        "vand.vi v0, v0, 1"                                         \
        "vsub.vi v0, v0, 1"                                         \
        "vse" #SEW ".v v0, (%0)"                                    \
	    :"=m"(dest->vals)   	                                    \
	    :            		                                        \
    );                                                              \
})

static inline void slide1Up(const Suint n, const V* src, V* dest) {
    __asm__ volatile(                                               \
        "vfslide1up.vf %0, %1, zero \n\t"                           \
        :"=m"(dest->vals)                                           \
        :"m"(src->vals)                                             \
    );       
}

static inline void slide1Down(const Suint n, const V* src, V* dest) {
    __asm__ volatile(                                               \
        "vfslide1down.vf %0, %1, zero \n\t"                         \
        :"=m"(dest->vals)                                           \
        :"m"(src->vals)                                             \
    );       
}

// a+bi => a+ai for all elements
static inline V VDUPL(const V* x) {
	const Suint VLEN = VL(SEW);
    V xl, xs, res;
    Vuint partr;
    newVector(VLEN, &xl);
    newVector(VLEN, &xs);
    newVector(VLEN, &res);
    newVector(VLEN, &partr);

    VPARTSPLIT(VLEN, SEW, &partr); // (-1, 0, -1, 0...)
    VAND(VLEN, x, &partr, &xl); // Mask off odd indices
    slide1Up(VLEN, &xl, &xs);
    VADD(VLEN, &xs, &xl, &res);

    freeVector(&partr);
    freeVector(&xl);
    freeVector(&xs);
    return res;
}

// a+bi => b+bi for all elements
static inline V VDUPH(const V* x) {
	const Suint VLEN = VL(SEW);
    V xh, xs, res;
    Vuint partr, parti;
    newVector(VLEN, &xh);
    newVector(VLEN, &xs);
    newVector(VLEN, &res);
    newVector(VLEN, &partr);
    newVector(VLEN, &parti);

    VPARTSPLIT(VLEN, SEW, &partr); // (-1, 0, -1, 0, ...)
    VNOT(VLEN, &partr, &parti); // (0, -1, 0, -1, ...)
    VAND(VLEN, x, &parti, &xh); // Mask off even indices
    slide1Down(VLEN, &xh, &xs);
    VADD(VLEN, &xs, &xh, &res);

    freeVector(&partr);
    freeVector(&parti);
    freeVector(&xh);
    freeVector(&xs);
    return res;
}

// a+bi => b+ai for all elements
static inline V FLIP_RI(const V* x) {
	const Suint VLEN = VL(SEW);
    V xh, xl, xs, xs2, res;
    Vuint partr, parti;
    newVector(VLEN, &xh);
    newVector(VLEN, &xs);
    newVector(VLEN, &xs2);
    newVector(VLEN, &res);
    newVector(VLEN, &partr);
    newVector(VLEN, &parti);

    VPARTSPLIT(VLEN, SEW, &partr); // (-1, 0, -1, 0, ...)
    VAND(VLEN, x, &partr, &xl);
    VNOT(VLEN, &partr, &parti); // (0, -1, 0, -1, ...)
    VAND(VLEN, x, &parti, &xh); // set even elements to 0
    slide1Down(VLEN, &xh, &xs);
    slide1Up(VLEN, &xl, &xs2);
    VADD(VLEN, &xs, &xs2, &res);

    freeVector(&partr);
    freeVector(&parti);
    freeVector(&xh);    
    freeVector(&xs);    
    freeVector(&xs2);    
    freeVector(&xl);
    return res;
}

// a+bi => a-bi for all elements
static inline V VCONJ(const V* x) {
	const Suint VLEN = VL(SEW);
    V xh, xl, res;
    Vuint partr, parti;
    newVector(VLEN, &xh);
    newVector(VLEN, &res);
    newVector(VLEN, &partr);
    newVector(VLEN, &parti);

    VPARTSPLIT(VLEN, SEW, &partr); // (-1, 0, -1, 0, ...)
    VAND(VLEN, x, &partr, &xl);
    VNOT(VLEN, &partr, &parti); // (0, -1, 0, -1, ...)
    VAND(VLEN, x, &parti, &xh); // set even elements to 0
    VNEG(VLEN, &xh, &xh);
    VADD(VLEN, &xl, &xh, &res);

    freeVector(&partr);
    freeVector(&parti);
    freeVector(&xh);    
    freeVector(&xl);
    return res;
}

// Same as FLIP_RI, argument isn't const qualified
static inline V VBYI(V* x) {
	const Suint VLEN = VL(SEW);
    V xh, xl, res;
    Vuint partr, parti;
    newVector(VLEN, &xh);
    newVector(VLEN, &res);
    newVector(VLEN, &partr);
    newVector(VLEN, &parti);

    VPARTSPLIT(VLEN, SEW, &partr); // (-1, 0, -1, 0, ...)
    VAND(VLEN, x, &partr, &xl); // set odd elements to 0
    VNOT(VLEN, &partr, &parti); // (0, -1, 0, -1, ...)
    VAND(VLEN, x, &parti, &xh); // set even elements to 0
    VNEG(VLEN, &xh, &xh);
    VADD(VLEN, &xl, &xh, &res);

    freeVector(&partr);
    freeVector(&parti);
    freeVector(&xh);    
    freeVector(&xl);
    return res;
}

// Hybrid instructions
#define VFMAI(b, c) VADD(c, VBYI(b))
#define VFNMSI(b, c) VSUB(c, VBYI(b))
#define VFMACONJ(b, c) VADD(VCONJ(b), c)
#define VFMSCONJ(b, c) VSUB(VCONJ(b), c)
#define VFNMSCONJ(b, c) VSUB(c, VCONJ(b))

static inline V VZMUL(V* tx, V* sr) {
	const Suint VLEN = VL(SEW);
    V tr, ti, srs, res;
    newVector(VLEN, &tr);
    newVector(VLEN, &ti);
    newVector(VLEN, &res);
    newVector(VLEN, &srs);

    tr = VDUPL(tx);
    ti = VDUPH(tx);
    srs = VBYI(sr);
    VMUL(VLEN, sr, &tr, &res);
    VFMA(VLEN, &ti, &srs, &res);

    freeVector(&ti);
    freeVector(&tr);
    freeVector(&srs);
    return res;
}

static inline V VZMULJ(V* tx, V* sr) {
	const Suint VLEN = VL(SEW);
    V tr, ti, srs, res;
    newVector(VLEN, &tr);
    newVector(VLEN, &ti);
    newVector(VLEN, &res);
    
    tr = VDUPL(tx);
    ti = VDUPH(tx);
    srs = VBYI(sr); 
    VMUL(VLEN, sr, &tr, &res);
    VFNMS(VLEN, &ti, &srs, &res);

    freeVector(&ti);
    freeVector(&tr);
    freeVector(&srs);
    return res;
}

static inline V VZMULI(V* tx, V* sr) {
    const Suint VLEN = VL(SEW);
    V tr, ti, srs, res;
    newVector(VLEN, &tr);
    newVector(VLEN, &ti);
    newVector(VLEN, &res);
    newVector(VLEN, &srs);

    tr = VDUPL(tx);
    ti = VDUPH(tx);
    srs = VBYI(sr);
    VMUL(VLEN, sr, &tr, &res);
    VFMS(VLEN, &ti, &srs, &res);

    freeVector(&ti);
    freeVector(&tr);
    freeVector(&srs);
    return res;
}

static inline V VZMULIJ(V* tx, V* sr) {
    const Suint VLEN = VL(SEW);
    V tr, ti, srs, res;
    newVector(VLEN, &tr);
    newVector(VLEN, &ti);
    newVector(VLEN, &res);
    newVector(VLEN, &srs);

    tr = VDUPL(tx);
    ti = VDUPH(tx);
    srs = VBYI(sr);
    VMUL(VLEN, sr, &tr, &res);
    VFMA(VLEN, &ti, &srs, &res);

    freeVector(&ti);
    freeVector(&tr);
    freeVector(&srs);
    return res;
}


#else
#error "RISC-V V vector only works for 64 bits"
#endif

