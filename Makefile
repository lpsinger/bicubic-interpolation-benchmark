SHELL = bash

all: bench.txt

bicubic: bicubic.c
	$(CC) -std=gnu11 -Ofast -ftree-vectorize -march=native $< $(shell pkg-config --cflags --libs gsl) -o $@

bicubic_cxx: bicubic.cpp
	$(CXX) -Ofast -ftree-vectorize -march=native -Ixtl/include -Ixtensor/include $< -o $@

c.txt: bicubic
	./bicubic > $@

cxx.txt: bicubic_cxx
	./bicubic_cxx | tail -n 1 > $@

py.txt: bicubic.py
	./bicubic.py > $@

bench.txt: c.txt cxx.txt py.txt
	echo -n 'C   ' > $@
	cat c.txt >> $@
	echo -n 'C++ ' >> $@
	cat cxx.txt >> $@
	echo -n 'Py  ' >> $@
	cat py.txt >> $@

clean:
	rm -f c.txt cxx.txt py.txt bench.txt bicubic bicubic_cxx
