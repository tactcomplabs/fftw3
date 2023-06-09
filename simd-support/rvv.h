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

// Constructor/destructor
static inline void newVector(const Suint n, void* vec) {
    V* tmp = (V*)vec;
    tmp->nElem = n;
    tmp->vals = calloc(2*n, SEW);
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
            :"r"(2*n), "m"(x->vals), "m"(y->vals)                             \
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
            :"r"(2*n), "m"(x->vals), "m"(y->vals)                           \
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
            :"r"(2*n), "m"(x->vals)                                         \
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
            "\n\t slli t0, t0, " str(ESHIFT)                        \
	    "vid.v v0" 	                                                \
        "vand.vi v0, v0, 1"                                         \
        "vsub.vi v0, v0, 1"                                         \
        "\n\t vs" str(SEW)  " v0, (%0)"                             \
            "\n\t add %0, %0, t0"                                   \
            "\n\t bnez %1, 1b"                                      \
            "\n\t ret"                                              \
	    :"=m"(dest->vals)   	                                    \
	    :"r"(2*n)            		                                \
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
static inline void DUPL_RE(const V* src, V* res) {
    Suint n = 2 * src->nElem;                                                           \
    __asm__ volatile(                                                                   \
        "1:"                                                                            \
        "\n\t vsetvli t0, %1, e" str(SEW) ", ta, ma"  /* # of elems left */             \
        "\n\t vlseg2e" str(SEW)  ".v v0, (%2)"        /* load Re/Im into v0/v1 */       \
            "\n\t slli t0, t0, 1"                     /* adjust t0 for double load */   \
            "\n\t sub %1, %1, t0"                     /* decrement elems left by t0 */  \
            "\n\t slli t0, t0, " str(ESHIFT)          /* convert # elems to bytes */    \
            "\n\t add %2, %2, t0"                     /* bump pointer */                \
        "\n\t vmv.v.v v1, v0"                         /* replace Im with Re */          \
        "\n\t vsseg2e" str(SEW) ".v v0, (%0)"         /* store back into memory */      \
            "\n\t add %0, %0, t0"                     /* bump res pointer */            \
        "\n\t bnez %1, 1b"                            /* loop back? */                  \
        "\n\t ret"                                                                      \
        :"=m"(res->vals)                                                                \
        :"r"(n), "m"(src->vals)                                                         \
        :"t0"                                                                           \
    );                                                                                  \                                                           
}

// a+bi => b+bi for all elements
static inline void DUPL_IM(const V* src, V* res) {
    Suint n = 2 * src->nElem;                                                           \
    __asm__ volatile(                                                                   \
        "1:"                                                                            \
        "\n\t vsetvli t0, %1, e" str(SEW) ", ta, ma"  /* # of elems left */             \
        "\n\t vlseg2e" str(SEW) ".v v0, (%2)"         /* load Re/Im into v0/v1 */       \
            "\n\t slli t0, t0, 1"                     /* adjust t0 for double load */   \
            "\n\t sub %1, %1, t0"                     /* decrement elems left by t0 */  \
            "\n\t slli t0, t0, " str(ESHIFT)          /* convert t0 to bytes */         \
            "\n\t add %2, %2, t0"                     /* bump pointer */                \
        "\n\t vmv.v.v v0, v1"                         /* replace Re with Im */          \
        "\n\t vsseg2e" str(SEW) ".v v0, (%0)"         /* store back into memory */      \
            "\n\t add %0, %0, t0"                     /* bump res pointer */            \
        "\n\t bnez %1, 1b"                            /* loop back? */                  \
        "\n\t ret"                                                                      \
        :"=m"(res->vals)                                                                \
        :"r"(n), "m"(src->vals)                                                         \
        :"t0"                                                                           \
    );                                                                                  \
}

// a+bi => b+ai for all elements
static inline void FLIP_RI(const V* src, V* res) {
    Suint n = 2 * src->nElem;
    const Suint SEWB = SEW / 8; // SEW in bytes
    __asm__ volatile(
            "\n\t slli t2, %3, 1"                       /* stride in bytes */               \
        "1:"                                                                                \
        "\n\t vsetvli t0, %1, e" str(SEW) ", ta, ma"    /* # of elems left */               \
        "\n\t vlse" str(SEW) ".v v1, (%2), t2"          /* load Re into v1 */               \
        "\n\t add t1, %2, %3"                           /* advance one elem */              \
        "\n\t vlse" str(SEW) ".v v0, t1, t2"            /* load Im into v0 */               \
            "\n\t slli t0, t0, 1"                       /* adjust t0 for double load */     \
            "\n\t sub %1, %1, t0"                       /* decrement elems left by t0 */    \
            "\n\t slli t0, t0, " str(ESHIFT)            /* convert t0 to bytes */           \
            "\n\t add %2, %2, t0"                       /* bump pointer */                  \
        "\n\t vsseg2e" str(SEW) ".v v0, (%0)"           /* store back into memory */        \
            "\n\t add %0, %0, t0"                       /* bump res pointer */              \
        "\n\t bnez %1, 1b"                              /* loop back? */                    \
        "\n\t ret"                                                                          \
        :"=m"(res->vals)                                                                    \
        :"r"(n), "m"(src->vals), "r"(SEWB)                                                  \
        :"t0","t1","t2"                                                                     \
    );
}

// a+bi => a-bi for all elements
static inline void VCONJ(const V* src, V* res) {
    Suint n = 2 * src->nElem;                                                           \
    __asm__ volatile(                                                                   \
        "1:"                                                                            \
        "\n\t vsetvli t0, %1, e" str(SEW) ", ta, ma"  /* # of elems left */             \
        "\n\t vlseg2e" str(SEW)  ".v v0, (%2)"        /* load Re/Im into v0/v1 */       \
            "\n\t sub %1, %1, t0"                     /* decrement elems left by t0 */  \
            "\n\t slli t0, t0, " str(ESHIFT)          /* convert t0 to bytes */         \
            "\n\t add %2, %2, t0"                     /* bump pointer */                \
        "\n\t vneg.v v1, v1"                          /* conjugate Im */                \
        "\n\t vsseg2e" str(SEW) ".v v0, (%0)"         /* store back into memory */      \
            "\n\t add %2, %2, t0"                     /* bump pointer */                \
        "\n\t bnez %1, 1b"                            /* loop back? */                  \
        "\n\t ret"                                                                      \
        :"=m"(res->vals)                                                                \
        :"r"(n), "m"(src->vals)                                                         \
        :"t0"                                                                           \
    );                                                                                  \
}

// a+bi => -b+ai for all elements
static inline void VBYI(const V* x, V* res) {
    Suint n = 2 * x->nElem;
    const Suint SEWB = SEW / 8; // SEW in bytes
    __asm__ volatile(
            "\n\t slli t2, %3, 1"                       /* stride in bytes */               \
        "1:"                                                                                \
        "\n\t vsetvli t0, %1, e" str(SEW) ", ta, ma"    /* # of elems left */               \
        "\n\t vlse" str(SEW) ".v v1, (%2), t2"          /* load Re into v1 */               \
            "\n\t add t1, %2, %3"                       /* advance one element*/            \
        "\n\t vlse" str(SEW) ".v v0, t1, t2"            /* load Im into v0 */               \
            "\n\t slli t0, t0, 1"                       /* adjust t0 for 2x load */         \
            "\n\t sub %1, %1, t0"                       /* decrement elems left by t0 */    \
            "\n\t slli t0, t0, " str(ESHIFT)            /* convert t0 to bytes */           \
            "\n\t add %2, %2, t0"                       /* bump pointer */                  \
        "\n\t vneg.v.v v0, v0"                          /* conjugate Im */                  \
        "\n\t vsseg2e" str(SEW) ".v v0, (%0)"           /* store back into memory */        \
            "\n\t add %0, %0, t0"                       /* bump res pointer */              \
        "\n\t bnez %1, 1b"                              /* loop back? */                    \
        "\n\t ret"                                                                          \
        :"=m"(res->vals)                                                                    \
        :"r"(n), "m"(src->vals), "r"(SEWB)                                                  \
        :"t0","t1","t2"                                                                     \
    );
}


// Hybrid instructions
#define VFMAI(b, c) VADD(c, VBYI(b))
#define VFNMSI(b, c) VSUB(c, VBYI(b))
#define VFMACONJ(b, c) VADD(VCONJ(b), c)
#define VFMSCONJ(b, c) VSUB(VCONJ(b), c)
#define VFNMSCONJ(b, c) VSUB(c, VCONJ(b))

static inline void VZMUL(V* tx, V* sr) {
	const Suint n = src->nElem;
    V tr, ti, srs, res;
    newVector(n, &tr);
    newVector(n, &ti);
    newVector(n, &srs);
    newVector(n, &res);

    tr = DUPL_RE(tx);
    ti = DUPL_IM(tx);
    srs = VBYI(sr);
    VMUL(n, sr, &tr, &res);
    VFMA(n, &ti, &srs, &res);

    freeVector(&tr);
    freeVector(&ti);
    freeVector(&srs);
    return res;
}

static inline void VZMULJ(V* tx, V* sr) {
	const Suint n = src->nElem;
    V tr, ti, srs, res;
    newVector(n, &tr);
    newVector(n, &ti);
    newVector(n, &srs);
    newVector(n, &res);
    
    tr = DUPL_RE(tx);
    ti = DUPL_IM(tx);
    srs = VBYI(sr); 
    VMUL(n, sr, &tr, &res);
    VFNMS(n, &ti, &srs, &res);

    freeVector(&tr);
    freeVector(&ti);
    freeVector(&srs);
    return res;
}

static inline void VZMULI(V* tx, V* sr) {
    const Suint n = src->nElem;
    V tr, ti, srs, res;
    newVector(n, &tr);
    newVector(n, &ti);
    newVector(n, &srs);
    newVector(n, &res);

    tr = DUPL_RE(tx);
    ti = DUPL_IM(tx);
    srs = VBYI(sr);
    VMUL(n, sr, &tr, &res);
    VFMS(n, &ti, &srs, &res);

    freeVector(&tr);
    freeVector(&ti);
    freeVector(&srs);
    return res;
}

static inline void VZMULIJ(V* tx, V* sr) {
    const Suint n = src->nElem;
    V tr, ti, srs, res;
    newVector(n, &tr);
    newVector(n, &ti);
    newVector(n, &srs);
    newVector(n, &res);

    tr = DUPL_RE(tx);
    ti = DUPL_IM(tx);
    srs = VBYI(sr);
    VMUL(n, sr, &tr, &res);
    VFMA(n, &ti, &srs, &res);

    freeVector(&tr);
    freeVector(&ti);
    freeVector(&srs);
    return res;
}


#else
#error "RISC-V V vector only works for 64 bits"
#endif

