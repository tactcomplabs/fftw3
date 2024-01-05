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

#if !defined(__riscv_xlen) || __riscv_xlen != 64
#error "RISCV-V vector extension targets RV64"
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

#include <stdint.h> // uint32/64_t
#include <string.h> // memset
#include <stdlib.h> // calloc

// Scalar types
typedef DS(uint64_t, uint32_t) Suint;
typedef ptrdiff_t INT; // from kernel/ifftw.h
typedef DS(double, float) R;

// Vector types
typedef Suint* Vuint;
typedef R* V;

// Assembly for basic arithmetic operations
#define VOP(op, etype)                                                          \
    void v##op(const etype x, const etype y, etype z, const Suint nElem) {      \
        __asm__ volatile(                                                       \
            "1:"                                                                \
            "\n\t vsetvli t0, %3, e" str(SEW) ", m1, ta, ma"                    \
            "\n\t vle" str(SEW)  ".v v0, (%0)"                                  \
            "\n\t vle" str(SEW)  ".v v1, (%1)"                                  \
            "\n\t" #op ".vv v2, v0, v1"                                         \
            "\n\t vse" str(SEW)  ".v v2, (%2)"                                  \
                "\n\t sub %3, %3, t0"                                           \
                "\n\t slli t0, t0, " str(ESHIFT)                                \
                "\n\t add %0, %0, t0"                                           \
                "\n\t add %1, %1, t0"                                           \
                "\n\t add %2, %2, t0"                                           \
                "\n\t bnez %3, 1b"                                              \
            :                                                                   \
            :"r"(x), "r"(y), "r"(z), "r"(2*nElem)                               \
            :"t0","memory"                                                      \
        );                                                                      \
    }                                                                           \

// Assembly for fused arithmetic instructions
#define VFOP(op, etype)                                                         \
    void v##op(const etype x, const etype y, etype z, const Suint nElem) {  	\
        __asm__ volatile(                                                       \
            "1:"                                                                \
            "\n\t vsetvli t0, %3, e" str(SEW) ", m1, ta, ma"                    \
            "\n\t vle" str(SEW) ".v v0, (%0)"                                   \
            "\n\t vle" str(SEW) ".v v1, (%1)"                                   \
            "\n\t vle" str(SEW) ".v v2, (%2)"                                   \
            "\n\t" #op ".vv v2, v0, v1"                                         \
            "\n\t vse" str(SEW)  ".v v2, (%2)"                                  \
                "\n\t sub %3, %3, t0"                                           \
                "\n\t slli t0, t0, " str(ESHIFT)                                \
                "\n\t add %0, %0, t0"                                           \
                "\n\t add %1, %1, t0"                                           \
                "\n\t add %2, %2, t0"                                           \
                "\n\t bnez %3, 1b"                                              \
            :                                                                   \
            :"r"(x), "r"(y), "r"(z), "r"(2*nElem)                               \
            :"t0","memory"                                                      \
        );                                                                      \
    }                                                                           \

// Assembly for unary operations
#define VOP_UN(op, etype)                                                       \
    void v##op(const etype x, etype y, const Suint nElem) {                	    \
        __asm__ volatile(                                                       \
            "1:"                                                                \
            "\n\t vsetvli t0, %2, e" str(SEW) ", m1, ta, ma"                    \
            "\n\t vle" str(SEW)  ".v v0, (%0)"                                  \
            "\n\t" #op ".v v1, v0"                                              \
            "\n\t vse" str(SEW)  ".v v1, (%1)"                                  \
                "\n\t sub %2, %2, t0"                                           \
                "\n\t slli t0, t0, " str(ESHIFT)                                \
                "\n\t add %0, %0, t0"                                           \
                "\n\t add %1, %1, t0"                                           \
            "\n\t bnez %2, 1b"                                                  \
            :                                                                   \
            :"r"(x), "r"(y), "r"(2*nElem)                                       \
            :"t0","memory"                                                      \
        );                                                                      \
    }                                                                           \

// Generate prototypes
VOP(vfadd, V)
VOP(vfsub, V)
VOP(vfmul, V)
VFOP(vfmacc, V)
VFOP(vfmsac, V)
VFOP(vfnmsac, V)
VOP_UN(vfneg, V)

// Macro wrappers for arithmetic functions
#define VADD(x, y, z, n)     (vvfadd)(x, y, z, n)
#define VSUB(x, y, z, n)     (vvfsub)(x, y, z, n)
#define VMUL(x, y, z, n)     (vvfmul)(x, y, z, n)
#define VNEG(x, y, n)        (vvfneg)(x, y, n)

// Macro wrappers for fused arithmetic functions
#define VFMA(a, b, c, n)     (vvfmacc)(a, b, c, n)
#define VFMS(a, b, c, n)     (vvfmsac)(a, b, c, n)
#define VFNMS(a, b, c, n)    (vvfnmsac)(a, b, c, n)

// a+bi => a+ai for all elements
static inline void VDUPL(const V src, V res, const Suint nElem) {
    __asm__ volatile(
        "1:"
        "\n\t vsetvli t0, %2, e" str(SEW) ", m1, ta, ma"  /* complex nums left */
        "\n\t vlseg2e" str(SEW) ".v v0, (%0)"             /* load Re/Im */
		"\n\t vmv.v.v v1, v0"							  /* copy Re into Im */
            "\n\t sub %2, %2, t0"                         /* decrement elems left by t0 */
            "\n\t slli t1, t0," str(ESHIFT)               /* convert # elems to bytes */
			"\n\t slli t1, t1, 1"						  /* account for double load */
        "\n\t vsseg2e" str(SEW) ".v v0, (%1)"             /* store into res */
            "\n\t add %0, %0, t1"                         /* bump pointer */
            "\n\t add %1, %1, t1"                         /* bump pointer */
        "\n\t bnez %2, 1b"                                /* make final pass */
        :
        :"r"(src),"r"(res),"r"(nElem)
        :"t0","t1","memory"
    );
}

// a+bi => b+bi for all elements
static inline void VDUPH(const V src, V res, const Suint nElem) {
    __asm__ volatile(
        "1:"
        "\n\t vsetvli t0, %2, e" str(SEW) ", m1, ta, ma"  /* complex nums left */
        "\n\t vlseg2e" str(SEW) ".v v0, (%0)"             /* load Re/Im */
		"\n\t vmv.v.v v0, v1"							  /* copy Im into Re */
            "\n\t sub %2, %2, t0"                         /* decrement elems left by t0 */
            "\n\t slli t1, t0," str(ESHIFT)               /* convert # elems to bytes */
			"\n\t slli t1, t1, 1"						  /* account for double load */
        "\n\t vsseg2e" str(SEW) ".v v0, (%1)"             /* store into res */
            "\n\t add %0, %0, t1"                         /* bump pointer */
            "\n\t add %1, %1, t1"                         /* bump pointer */
        "\n\t bnez %2, 1b"                                /* make final pass */
        :
        :"r"(src),"r"(res),"r"(nElem)
        :"t0","t1","memory"
    );
}

// a+bi => b+ai for all elements
static inline void FLIP_RI(const V src, V res, const Suint nElem) {
    __asm__ volatile(
        "1:"
        "\n\t vsetvli t0, %2, e" str(SEW) ", m1, ta, ma"  /* complex nums left */
        "\n\t vlseg2e" str(SEW)  ".v v0, (%0)"
        "\n\t vmv.v.v v2, v1"                             /* swap v0/v1 */
        "\n\t vmv.v.v v1, v0"
        "\n\t vmv.v.v v0, v2"
        "\n\t vsseg2e" str(SEW) ".v v0, (%1)"
            "\n\t sub %2, %2, t0"                         /* decrement elems left by t0 */
            "\n\t slli t1, t0," str(ESHIFT)               /* convert # elems to bytes */
			"\n\t slli t1, t1, 1"						  /* account for double load */
            "\n\t add %0, %0, t1"                         /* bump pointer */
            "\n\t add %1, %1, t1"                         /* bump pointer */
        "\n\t bnez %2, 1b"								  /* loop back */
		:
        :"r"(src),"r"(res),"r"(nElem)
        :"t0","t1","memory"
    );
}

// a+bi => a-bi for all elements
static inline void VCONJ(const V src, V res, const Suint nElem) {
    __asm__ volatile(
        "1:"
        "\n\t vsetvli t0, %2, e" str(SEW) ", m1, ta, ma"  /* complex nums left */
        "\n\t vlseg2e" str(SEW)  ".v v0, (%0)"            /* load Re/Im into v0/v1 */
        "\n\t vfneg.v v1, v1"                             /* conjugate Im */
        "\n\t vsseg2e" str(SEW) ".v v0, (%1)"             /* store back into memory */
            "\n\t sub %2, %2, t0"                         /* decrement elems left by t0 */
            "\n\t slli t1, t0," str(ESHIFT)               /* convert # elems to bytes */
			"\n\t slli t1, t1, 1"						  /* account for double load */
            "\n\t add %0, %0, t1"                         /* bump pointer */
            "\n\t add %1, %1, t1"                         /* bump pointer */
        "\n\t bnez %2, 1b"                                /* loop back */
        :
        :"r"(src),"r"(res),"r"(nElem)
        :"t0","t1","memory"
    );
}

// a+bi => -b+ai for all elements
static inline void VBYI(const V src, V res, const Suint nElem) {
    __asm__ volatile(
        "1:"
        "\n\t vsetvli t0, %2, e" str(SEW) ", m1, ta, ma"
        "\n\t vlseg2e" str(SEW)  ".v v0, (%0)"
        "\n\t vfneg.v v2, v1"                             /* conjugate Im */
        "\n\t vmv.v.v v1, v0"                             /* swap Re/Im */
        "\n\t vmv.v.v v0, v2"
            "\n\t sub %2, %2, t0"                         /* decrement elems left by t0 */
            "\n\t slli t1, t0," str(ESHIFT)               /* convert # elems to bytes */
			"\n\t slli t1, t1, 1"						  /* account for double load */
        "\n\t vsseg2e" str(SEW) ".v v0, (%1)"             /* store into res */
            "\n\t add %0, %0, t1"                         /* bump pointer */
            "\n\t add %1, %1, t1"                         /* bump pointer */
        "\n\t bnez %2, 1b"                                /* loop back */
        :
        :"r"(src),"r"(res),"r"(nElem)
        :"t0","t1","memory"
    );
}

// Hybrid instructions
#define LDK(x) x
#define VFMAI(b, c, n)     VADD(c, VBYI(b, n), n)
#define VFNMSI(b, c, n)    VSUB(c, VBYI(b, n), n)
#define VFMACONJ(b, c, n)  VADD(c, VCONJ(b, n), n)
#define VFMSCONJ(b, c, n)  VSUB(VCONJ(b, n), c, n)
#define VFNMSCONJ(b, c, n) VSUB(c, VCONJ(b, n), n)

// (a+bi) * (c+di)
static inline void VZMUL(const V tx, const V sr, V res, const Suint nElem) {
	R srf[2*nElem];
    R txIm[2*nElem];
    VDUPL(tx, res, nElem);
    VDUPH(tx, &txIm[0], nElem);
    VMUL(sr, res, res, nElem);
    VBYI(sr, &srf[0], nElem);
    VFMA(&srf[0], &txIm[0], res, nElem);
}

// conj(a+bi) * (c+di)
static inline void VZMULJ(const V tx, const V sr, V res, const Suint nElem) {
    R srf[2*nElem];
    R txIm[2*nElem];
    VDUPL(tx, res, nElem);
    VDUPH(tx, &txIm[0], nElem);
    VMUL(sr, res, res, nElem);
    VBYI(sr, &srf[0], nElem);
    VFNMS(&srf[0], &txIm[0], res, nElem);
}

// (a+bi) * (c+di) -> conj -> flip R/I
static inline void VZMULI(const V tx, const V sr, V res, const Suint nElem) {
    R tr[2*nElem];
    R srf[2*nElem];
    VDUPL(tx, &tr[0], nElem);
    VDUPH(tx, res, nElem);
    VMUL(res, sr, res, nElem);
    VBYI(sr, &sr[0], nElem);
    VFMS(&tr[0], &sr[0], res, nElem);
}

// (b+ai) * (c+di)
static inline void VZMULIJ(const V tx, const V sr, V res, const Suint nElem) {
    R tr[2*nElem];
    R srf[2*nElem];
    VDUPL(tx, &tr[0], nElem);
    VDUPH(tx, res, nElem);
    VMUL(res, sr, res, nElem);
    VBYI(sr, &sr[0], nElem);
    VFMA(&tr[0], &sr[0], res, nElem);
}

// Loads data from x into new vector
// Assumes that it can fit into single vector
static inline V LDA(R* x, INT ivs, R* aligned_like, const Suint nElem) {
    (void) aligned_like; // suppress unused var warning
    V res = calloc(nElem*2, sizeof(R));
    memcpy(res, x, 2*nElem*sizeof(R));
    return res;
}

// Stores data from v into x
static inline void STA(R* x, V v, INT ovs, R* aligned_like, const Suint nElem) {
    (void) aligned_like; // suppress unused var warning
    memcpy(v, x, 2*nElem*sizeof(R));
}

