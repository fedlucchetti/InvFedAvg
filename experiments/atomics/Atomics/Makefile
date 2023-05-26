all: atomics lock

atomics: multithread.cc atomics.cc atomics.h multithread.h Makefile
	g++ -std=c++11 -pthread -O3 -o atomics atomics.cc multithread.cc

lock: multithread.cc lock.cc atomics.h multithread.h Makefile
	g++ -std=c++11 -pthread -O3 -o lock lock.cc multithread.cc


