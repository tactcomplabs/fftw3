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
 #  define DS(d, s) s /* single-precision option */
 #  define SEW 32
 #  define ESHIFT 2
#else
 #  define DS(d, s) d /* double-precision option */
 #  define SEW 64
 #  define ESHIFT 3
#endif

#define ZERO DS(0.0,0.0f)
#define str(x) xstr(x)
#define xstr(x) #x

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
#define VL() ({                                                             \
    Suint res;                                                              \
    __asm__ volatile(                                                       \
        "vsetvli %0, zero, e" str(SEW) ", m1, ta, ma"                       \
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
#define VOP(op, etype)                                                      \
    void v##op(const Suint n, const etype* x, const etype* y, etype* z)     \
    {                                                                       \
       __asm__ volatile(                                                    \
            "1:"                                                            \
            "\n\t vsetvli t0, %1, e" str(SEW) ", ta, ma"                    \
            "\n\t vle" str(SEW)  " v0, (%2)"                                \
                "\n\t sub %1, %1, t0"                                       \
                "\n\t slli t0, t0, " str(eshift)                            \
                "\n\t add %2, %2, t0"                                       \
            "\n\t vle" str(SEW)  " v1, (%3)"                                \
                "\n\t add %3, %3, t0"                                       \
            "\n\t" str(op) " v2, v0, v1"                                    \
            "\n\t vse" str(SEW)  " v2, (%0)"                                \
                "\n\t add %0, %0, t0"                                       \
                "\n\t bnez %1, 1b"                                          \
                "\n\t ret"                                                  \
            :"=m"(z->vals)                                                  \
            :"r"(n), "m"(x->vals), "m"(y->vals)                             \
            :"t0"                                                           \
        );                                                                  \
    }

// Assembly for fused arithmetic instructions
#define VFOP(op, etype)                                                     \
    void v##op(Suint n, const etype* x, const etype* y, etype* z)           \
    {                                                                       \
       __asm__ volatile(                                                    \
            "1:"                                                            \
            "\n\t vsetvli t0, %1, e" str(SEW) ", ta, ma"                    \
            "\n\t vle" str(SEW)  " v0, (%2)"                                \
                "\n\t sub %1, %1, t0"                                       \
                "\n\t slli t0, t0, " str(eshift)                            \
                "\n\t add %2, %2, t0"                                       \
            "\n\t vle" str(SEW)  " v1, (%3)"                                \
                "\n\t add %3, %3, t0"                                       \
            "\n\t" str(op) " v2, v0, v1"                                    \
            "\n\t vs" str(SEW)  " v2, (%0)"                                 \
                "\n\t add %0, %0, t0"                                       \
                "\n\t bnez %1, 1b"                                          \
                "\n\t ret"                                                  \
            :"=m"(z->vals)                                                  \
            :"r"(n), "m"(x->vals), "m"(y->vals)                             \
            :"t0"                                                           \
        );                                                                  \
    }                                                                      

// Assembly for unary operations
#define VOP_UN(op, etype)                                                   \
    void v##op(const Suint n, const etype* x, etype* y)                     \
    {                                                                       \
       __asm__ volatile(                                                    \
            "1:"                                                            \
            "\n\t vsetvli t0, %1, e" str(SEW) ", ta, ma"                    \
            "\n\t vle" str(SEW)  " v0, (%2)"                                \
                "\n\t sub %1, %1, t0"                                       \
                "\n\t slli t0, t0, " str(eshift)                            \
                "\n\t add %2, %2, t0"                                       \
            "\n\t" str(op) " v1, v0"                                        \
            "\n\t vse" str(SEW)  " v1, (%0)"                                \
                "\n\t add %2, %2, t0"                                       \
            "\n\t bnez %1, 1b"                                              \
            "\n\t ret"                                                      \
            :"=m"(y->vals)                                                  \
            :"r"(n), "m"(x->vals)                                           \
            :"t0"                                                           \
            );                                                              \
    }

// Generate prototypes
VOP(vfadd, V)         
VOP(vfsub, V)         
VOP(vfmul, V)          
VOP(vfdiv, V)

VOP(vand, Vuint)
VOP(vor, Vuint)

VFOP(vfmacc, V)
VFOP(vfmsac, V)
VFOP(vfnmsac, V)

VOP_UN(vfneg, V)                                      
VOP_UN(vnot, Vuint)                                      

// Macro wrappers for arithmetic functions
#define VADD(n, x, y, z)     (vvfadd)(n, x, y, z)
#define VSUB(n, x, y, z)     (vvfsub)(n, x, y, z)
#define VMUL(n, x, y, z)     (vvfmul)(n, x, y, z)
#define VDIV(n, x, y, z)     (vvfdiv)(n, x, y, z)
#define VNEG(n, x, y)        (vvfneg)(n, x, y)

// Macro wrappers for logical functions
#define VAND(n, x, y, z)     (vvand)(n, x, y, z)
#define VOR(n, x, y, z)      (vvand)(n, x, y, z)
#define VNOT(n, x, y)        (vvnot)(n, x, y)

// Macro wrappers for fused arithmetic functions
#define VFMA(n, a, b, c)     (vvfmacc)(n, c, a, b)
#define VFMS(n, a, b, c)     (vvfmsac)(n, c, a, b)
#define VFNMS(n, a, b, c)    (vvfnmsac)(n, c, a, b)

// Generates a vector (-1, 0, -1, 0...) of length vl
// FIXME: Slated for removal if SIMD approach is abandoned
static inline void vpartsplit(const Suint n, Vuint* dest) {
    __asm__ volatile ( 		                                        \
        "1:"                                                        \
        "\n\t vsetvli t0, %1, e" str(SEW) ", ta, ma"                \
            "\n\t sub %1, %1, t0"                                   \
            "\n\t slli t0, t0, " str(eshift)                        \
	    "vid.v v0" 	                                                \
        "vand.vi v0, v0, 1"                                         \
        "vsub.vi v0, v0, 1"                                         \
        "\n\t vs" str(SEW)  " v0, (%0)"                             \
            "\n\t add %0, %0, t0"                                   \
            "\n\t bnez %1, 1b"                                      \
            "\n\t ret"                                              \
	    :"=m"(dest->vals)   	                                    \
	    :"r"(n)            		                                    \
        :"t0"                                                       \
    );                                                              \
}

// Shifts elements right one in vector
// FIXME: Needs nontrivial strip mining approach
// Must track element erased by slide and place it at front of next vector
// Likely to be removed in favor of alternative approach
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
	const Suint VLEN = VL();
    V xl, xls, res;
    Vuint partr;
    newVector(VLEN, &xl);
    newVector(VLEN, &xls);
    newVector(VLEN, &partr);
    newVector(VLEN, &res);

    vpartsplit(VLEN, &partr);         // (-1, 0, -1, 0...)
    VAND(VLEN, x, &partr, &xl); // Mask off even indices
    slide1Up(VLEN, &xl, &xls);
    VADD(VLEN, &xls, &xl, &res);

    freeVector(&xl);
    freeVector(&xls);
    freeVector(&partr);
    return res;
}

// a+bi => b+bi for all elements
static inline V VDUPH(const V* x) {
	const Suint VLEN = VL();
    V xh, xhs, res;
    Vuint partr, parti;
    newVector(VLEN, &xh);
    newVector(VLEN, &xhs);
    newVector(VLEN, &partr);
    newVector(VLEN, &parti);
    newVector(VLEN, &res);

    vpartsplit(VLEN, &partr);         // (-1, 0, -1, 0, ...)
    VNOT(VLEN, &partr, &parti); // (0, -1, 0, -1, ...)
    VAND(VLEN, x, &parti, &xh); // Mask off odd indices
    slide1Down(VLEN, &xh, &xhs);
    VADD(VLEN, &xhs, &xh, &res);

    freeVector(&xh);
    freeVector(&xhs);
    freeVector(&partr);
    freeVector(&parti);
    return res;
}

// a+bi => b+ai for all elements
static inline V FLIP_RI(const V* x) {
	const Suint VLEN = VL();
    V xh, xl, xhs, xls, res;
    Vuint partr, parti;
    newVector(VLEN, &xh);
    newVector(VLEN, &xl);
    newVector(VLEN, &xhs);
    newVector(VLEN, &xls);
    newVector(VLEN, &partr);
    newVector(VLEN, &parti);
    newVector(VLEN, &res);

    vpartsplit(VLEN, &partr);         // (-1, 0, -1, 0, ...)
    VAND(VLEN, x, &partr, &xl);
    VNOT(VLEN, &partr, &parti); // (0, -1, 0, -1, ...)
    VAND(VLEN, x, &parti, &xh); // set even elements to 0
    slide1Down(VLEN, &xh, &xhs);
    slide1Up(VLEN, &xl, &xls);
    VADD(VLEN, &xhs, &xls, &res);

    freeVector(&xh);    
    freeVector(&xl);
    freeVector(&xhs);    
    freeVector(&xls);    
    freeVector(&partr);
    freeVector(&parti);
    return res;
}

// a+bi => a-bi for all elements
static inline V VCONJ(const V* x) {
	const Suint VLEN = VL();
    V xh, xl, res;
    Vuint partr, parti;
    newVector(VLEN, &xh);
    newVector(VLEN, &xl);
    newVector(VLEN, &partr);
    newVector(VLEN, &parti);
    newVector(VLEN, &res);

    vpartsplit(VLEN, &partr);         // (-1, 0, -1, 0, ...)
    VAND(VLEN, x, &partr, &xl);
    VNOT(VLEN, &partr, &parti); // (0, -1, 0, -1, ...)
    VAND(VLEN, x, &parti, &xh); // set even elements to 0
    VNEG(VLEN, &xh, &xh);       // take conjugate
    VADD(VLEN, &xh, &xl, &res);

    freeVector(&xh);    
    freeVector(&xl);
    freeVector(&partr);
    freeVector(&parti);
    return res;
}

// a+bi => -b+ai for all elements
static inline V VBYI(V* x) {
	const Suint VLEN = VL();
    V xh, xl, xls, xhs, res;
    Vuint partr, parti;
    newVector(VLEN, &xh);
    newVector(VLEN, &xl);
    newVector(VLEN, &xls);
    newVector(VLEN, &xhs);
    newVector(VLEN, &partr);
    newVector(VLEN, &parti);
    newVector(VLEN, &res);

    vpartsplit(VLEN, &partr);         // (-1, 0, -1, 0, ...)
    VAND(VLEN, x, &partr, &xl); // set odd elements to 0
    VNOT(VLEN, &partr, &parti); // (0, -1, 0, -1, ...)
    VNEG(VLEN, x, x);           // take conjugate
    VAND(VLEN, x, &parti, &xh); // set even elements to 0
    slide1Down(VLEN, &xh, &xhs);
    slide1Up(VLEN, &xl, &xls);
    VADD(VLEN, &xls, &xhs, &res);

    newVector(VLEN, &xh);
    newVector(VLEN, &xl);
    newVector(VLEN, &xls);
    newVector(VLEN, &xhs);
    newVector(VLEN, &partr);
    newVector(VLEN, &parti);
    return res;
}

// Hybrid instructions
#define VFMAI(b, c) VADD(c, VBYI(b))
#define VFNMSI(b, c) VSUB(c, VBYI(b))
#define VFMACONJ(b, c) VADD(VCONJ(b), c)
#define VFMSCONJ(b, c) VSUB(VCONJ(b), c)
#define VFNMSCONJ(b, c) VSUB(c, VCONJ(b))

static inline V VZMUL(V* tx, V* sr) {
	const Suint VLEN = VL();
    V tr, ti, srs, res;
    newVector(VLEN, &tr);
    newVector(VLEN, &ti);
    newVector(VLEN, &srs);
    newVector(VLEN, &res);

    tr = VDUPL(tx);
    ti = VDUPH(tx);
    srs = VBYI(sr);
    VMUL(VLEN, sr, &tr, &res);
    VFMA(VLEN, &ti, &srs, &res);

    freeVector(&tr);
    freeVector(&ti);
    freeVector(&srs);
    return res;
}

static inline V VZMULJ(V* tx, V* sr) {
	const Suint VLEN = VL();
    V tr, ti, srs, res;
    newVector(VLEN, &tr);
    newVector(VLEN, &ti);
    newVector(VLEN, &srs);
    newVector(VLEN, &res);
    
    tr = VDUPL(tx);
    ti = VDUPH(tx);
    srs = VBYI(sr); 
    VMUL(VLEN, sr, &tr, &res);
    VFNMS(VLEN, &ti, &srs, &res);

    freeVector(&tr);
    freeVector(&ti);
    freeVector(&srs);
    return res;
}

static inline V VZMULI(V* tx, V* sr) {
    const Suint VLEN = VL();
    V tr, ti, srs, res;
    newVector(VLEN, &tr);
    newVector(VLEN, &ti);
    newVector(VLEN, &srs);
    newVector(VLEN, &res);

    tr = VDUPL(tx);
    ti = VDUPH(tx);
    srs = VBYI(sr);
    VMUL(VLEN, sr, &tr, &res);
    VFMS(VLEN, &ti, &srs, &res);

    freeVector(&tr);
    freeVector(&ti);
    freeVector(&srs);
    return res;
}

static inline V VZMULIJ(V* tx, V* sr) {
    const Suint VLEN = VL();
    V tr, ti, srs, res;
    newVector(VLEN, &tr);
    newVector(VLEN, &ti);
    newVector(VLEN, &srs);
    newVector(VLEN, &res);

    tr = VDUPL(tx);
    ti = VDUPH(tx);
    srs = VBYI(sr);
    VMUL(VLEN, sr, &tr, &res);
    VFMA(VLEN, &ti, &srs, &res);

    freeVector(&tr);
    freeVector(&ti);
    freeVector(&srs);
    return res;
}


#else
#error "RISC-V V vector only works for 64 bits"
#endif

