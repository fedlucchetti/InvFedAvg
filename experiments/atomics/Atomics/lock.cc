#include <iostream>
#include <atomic>

#include "multithread.h"
#include "atomics.h"

std::atomic<unsigned int> lock_var;

void lock() {
  while (xchg(reinterpret_cast<unsigned int volatile *>(&lock_var), 1) != 0) {}
}

void unlock() {
  lock_var = 0;
}

void init_thread(unsigned int myself) {
  if (myself == 0) {
    lock_var = 0;
  }
}

void run(unsigned int myself) {

  for (int i = 0; i < MAX_MEASURE; i++) {
    measure[myself][i][0] = rdtsc(); // start
    lock();
    measure[myself][i][1] = rdtsc(); // acquired
    // wait a bit
    for (int j = 0; j < 10; j++) asm_pause();
    measure[myself][i][2] = rdtsc(); // release
    unlock();
    measure[myself][i][3] = rdtsc(); // done
  }

}

void dump_measurements() {
  // sort by time acquired
  unsigned idx[MAX_THREADS];
  for (unsigned int i = 0; i < MAX_THREADS; i++)
    idx[i] = 0;

  while (true) {
    // find next
    unsigned t_min = 0;
    unsigned cnt_max = 0;
    for (unsigned int t = 0; t < MAX_THREADS; t++) {
      if (idx[t_min] == MAX_MEASURE) {
	t_min++;
      }
      
      if (idx[t] == MAX_MEASURE) {
	cnt_max++;
	continue;
      }
      
      if (measure[t][idx[t]][1] < measure[t_min][idx[t_min]][1]) {
	t_min = t;
      }
    }

    if (cnt_max == MAX_THREADS)
      break;

    // t_id is the next thread who got the lock
    
    std::cout << "[" << t_min << "] " << measure[t_min][idx[t_min]][1] << " ("
	      << measure[t_min][idx[t_min]][1] - measure[t_min][idx[t_min]][0] << "; "
      	      << measure[t_min][idx[t_min]][2] - measure[t_min][idx[t_min]][1] << "; "
	      << measure[t_min][idx[t_min]][3] - measure[t_min][idx[t_min]][2] << ")" << std::endl;
      
    idx[t_min]++;
  }
  
}
