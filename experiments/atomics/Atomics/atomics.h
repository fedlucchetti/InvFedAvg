#pragma once

inline void asm_pause() {
  asm volatile ("pause \n\t":::);
}

inline unsigned int rdtsc() {
  unsigned long val;

  asm volatile ("rdtsc \n\t" : "=a" (val) :: "rdx");
  return val;
}
	       
inline unsigned int xchg(unsigned int volatile * ptr, unsigned int val) {

  asm volatile ("lock; xchgl %0, %1\n\t": "+r" (val) : "m" (*ptr) : "memory");

  return val;
}

inline bool cas(unsigned int volatile * ptr, unsigned int oldval, unsigned int newval) 
{
  unsigned int tmp;

  asm volatile
    ("lock; cmpxchgl %1, %2"
     : "=a" (tmp)
     : "r" (newval), "m" (*ptr), "a" (oldval)
     : "memory");

  return tmp == oldval;
}
