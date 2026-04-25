/**
 * @file rdtsc.h
 * @brief Cross-platform RDTSC timing functions.
 *
 * Provides macros and a helper to measure elapsed CPU cycles using the
 * hardware timestamp counter. 
 */
#ifndef _RDTSC_H
#define _RDTSC_H

/** Assumed processor frequency in Hz; used to convert cycles to wall time. */
#define PROCESSOR_FREQ 3200000000

/** Extract the low/high 32-bit halves and the full 64-bit value of a tsc_counter. */
#define COUNTER_LO(a) ((a).int32.lo)
#define COUNTER_HI(a) ((a).int32.hi)
#define COUNTER_VAL(a) ((a).int64)

/** Convert a raw counter value to a unit-scaled double (divide by b). */
#define COUNTER(a,b) \
	(((double)COUNTER_VAL(a))/(b))

/** Difference between two counters, scaled by divisor c. */
#define COUNTER_DIFF(a,b,c) \
	(COUNTER(a,c)-COUNTER(b,c))

/** Raw 64-bit difference between two counters (no scaling). */
#define COUNTER_DIFF_SIMPLE(a,b) \
	(COUNTER_VAL(a)-COUNTER_VAL(b))

/** Divisor constants for COUNTER / COUNTER_DIFF. */
#define CYCLES		1
#define OPERATIONS	1
#define SEC		PROCESSOR_FREQ
#define MILI_SEC	(SEC/1E3)
#define MICRO_SEC	(SEC/1E6)
#define NANO_SEC	(SEC/1E9)

#if ! (defined(WIN32 ) || defined(WIN64))

#if defined(__GNUC__) || defined(__linux__)
#define VOLATILE __volatile__
#define ASM __asm__
#else
#define ASM asm
#define VOLATILE 
#endif

#define myInt64 unsigned long long
#define INT32 unsigned int

typedef union
{       myInt64 int64;
        struct {INT32 lo, hi;} int32;
} tsc_counter;

#if defined(__ia64__)
	#if defined(__INTEL_COMPILER)
		#define RDTSC(tsc) (tsc).int64=__getReg(3116)
	#else
		#define RDTSC(tsc) ASM VOLATILE ("mov %0=ar.itc" : "=r" ((tsc).int64) )
	#endif

	#define CPUID() do{/*No need for serialization on Itanium*/}while(0)
#else
	#define RDTSC(cpu_c) \
		ASM VOLATILE ("rdtsc" : "=a" ((cpu_c).int32.lo), "=d"((cpu_c).int32.hi))
	#define CPUID() \
		ASM VOLATILE ("cpuid" : : "a" (0) : "bx", "cx", "dx" )
#endif
	/**
	 * @brief Returns non-zero if the RDTSC counter is functional on this CPU.
	 * Reads the counter twice and checks that time advanced.
	 */
	int rdtsc_works(void) {
		tsc_counter t0,t1;
		RDTSC(t0);
		RDTSC(t1);
		return COUNTER_DIFF(t1,t0,1) > 0;
	}
#else
	#define myInt64 signed __int64
	#define INT32 unsigned __int32

	typedef union
	{       myInt64 int64;
		struct {INT32 lo, hi;} int32;
	} tsc_counter;


#ifdef _MSC_VER

	#include <intrin.h>

	#define RDTSC(cpu_c)    { cpu_c.int64 = __rdtsc(); }
	#define CPUID()         { int CPUInfo[4]; __cpuid(CPUInfo, 0); }

#else /* not _MSC_VER */
	#define RDTSC(cpu_c)   \
	{       __asm rdtsc    \
			__asm mov (cpu_c).int32.lo,eax  \
			__asm mov (cpu_c).int32.hi,edx  \
	}

	#define CPUID() \
	{ \
		__asm mov eax, 0 \
		__asm cpuid \
	}

#endif
	/**
	 * @brief Returns non-zero if RDTSC is functional (Windows version).
	 * Uses SEH to catch the illegal-instruction exception on machines where
	 * the TSC is disabled.
	 */
	int rdtsc_works(void) {
		tsc_counter t0,t1;
		__try {
		    RDTSC(t0);
		    RDTSC(t1);
		} __except ( 1) {
		    return 0;
		}
		return COUNTER_DIFF(t1,t0,1) > 0;
	}
#endif

// serialize reading the timer 
#define TIME(a) RDTSC(a)


#endif // _RDTSC_H
