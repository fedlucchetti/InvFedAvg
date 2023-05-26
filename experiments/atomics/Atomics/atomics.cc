#include <iostream>
#include <atomic>

#include "multithread.h"

#define CINC
//#define NINC
//#define AINC

#define VOLVAR
//#define AVAR
//#define NVAR

/* variables */

#if defined VOLVAR
volatile unsigned int a;
#endif

#if defined AVAR
std::atomic<unsigned int> a;
#endif

#if defined NVAR
unsigned int a;
#endif

/* assembly increment */

void increment (unsigned int volatile * a, unsigned int inc) {
  asm volatile ("addl %1, %0\n\t"
		: /* out */ "+m" (*a)
		: /* in */ "r" (inc)
		: /* clobber */);
}

/* atomic increment */

void atomic_increment (unsigned int volatile * a, unsigned int inc) {
  asm volatile ("lock; addl %1, %0\n\t"
		: /* out */ "+m" (*a)
		: /* in */ "r" (inc)
		: /* clobber */);
}



void init_thread(unsigned int myself) {}


void run(unsigned int myself) {

  unsigned int m = 0;
  unsigned int cmp = 0;
  
  switch (myself) {
  case 0:
    for (unsigned int i = 0; i < 0xfff; i++) {
      cmp += 1;
#if defined CINC
      a += 1;
#endif
#if defined NINC
      increment(reinterpret_cast<unsigned int volatile *>(&a), 1);
#endif
#if defined AINC
      atomic_increment(reinterpret_cast<unsigned int volatile *>(&a), 1);
#endif      
      if (((a & 0xffff) != cmp) && m < MAX_MEASURE) {
	measure[myself][m][0] = cmp;
	measure[myself][m][1] = a;
	m++;
      }
      
      if (m == MAX_MEASURE) break;
    }
    break;
  case 1:
    for (unsigned int i = 0; i < 0xfff; i++) {
      cmp += 1 << 16;
#if defined CINC
      a += 1 << 16;
#endif
#if defined NINC
      increment(reinterpret_cast<unsigned int volatile *>(&a), 1 << 16);
#endif
#if defined AINC
      atomic_increment(reinterpret_cast<unsigned int volatile *>(&a), 1 << 16);
#endif
      if (((a & 0xffff0000) != cmp) && 
	  m < MAX_MEASURE) {
	measure[myself][m][0] = cmp;
	measure[myself][m][1] = a;
	m++;
      }

      if (m == MAX_MEASURE) break;
    }
    break;
  }
}

void dump_measurements() {
  for (unsigned int t = 0; t < MAX_THREADS; t++) {
    for (unsigned int i = 0; i < MAX_MEASURE; i++) {
      if (measure[t][i][0] != 0) {
	std::cout << "[" << t << "] " << std::hex << measure[t][i][0] << ": a "
		  << measure[t][i][1] << std::endl;
      }
    }
  }
}
