#pragma once

/* C++ constants */
enum {
  MAX_THREADS = 4,
  MAX_MEASURE = 1024,
  MAX_ELEMS = 4,
};

#define DBG(x) x

/* Blob of memory to store measurements */
extern volatile unsigned long measure[MAX_THREADS][MAX_MEASURE][MAX_ELEMS];

/* thread init function */
void init_thread(unsigned int myself);

/* thread main function */
void run(unsigned int myself);

void dump_measurements();
