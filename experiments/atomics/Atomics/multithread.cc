#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <thread>
#include <atomic>

#include "multithread.h"
#include "atomics.h"

/* Layman's barrier to coordinate the starting and stopping of threads */
std::atomic<bool> go; 
std::atomic<bool> here[MAX_THREADS];
std::atomic<bool> done[MAX_THREADS];

volatile unsigned long measure[MAX_THREADS][MAX_MEASURE][MAX_ELEMS] __attribute__((aligned(4096)));

void generic_init(unsigned int myself) {

  init_thread(myself);

  // touch measure to make sure its cached
  for (unsigned int i = 0; i < MAX_MEASURE; i++) {
    for (unsigned int j = 0; j < MAX_ELEMS; j++) {
      measure[myself][i][j] = 0;
    }
  }

  here[myself] = true;

  if (myself == 0) {
    for (unsigned int t = 0; t < MAX_THREADS; t++) {
      while (!here[t]) {
	asm_pause();
      }
    }
    go = true;
  } else {
    while (!go) {
      asm_pause();
    }
  }
  run(myself);

  done[myself] = true;
  for (unsigned int there = 0; there < MAX_THREADS; there++) {
    while (!done[there]) {
      asm_pause();
    }
  }
}

int main (int argc, char * argv[]) {

  go = false;

  std::thread threads[MAX_THREADS];

  for (int i = 0; i < MAX_THREADS; i++) {
    /* initialize barrier variables */
    here[i] = false;
    done[i] = false;
    
    /* create thread: [i] {xxx}: is lambda function executed by thread */
    threads[i] = std::thread([i] {generic_init(i);}); 

    /* set affinity mask to pin thread to CPU */
    // !!! Linux pthread specific
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(1, &cpuset);
    int rc = pthread_setaffinity_np(threads[i].native_handle(),sizeof(cpu_set_t), &cpuset);    
    if (rc != 0) {
      std::cerr << "Error setting affinity " << rc << " for thread " << i << "\n";
    }
  }

  /* wait for threads to join */
  for (int i = 0; i < MAX_THREADS; i++) {
    threads[i].join();
  }

  dump_measurements();
  
  return 0;

}
