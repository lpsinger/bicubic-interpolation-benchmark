all: bench.txt

bicubic: bicubic.c
	$(CC) -std=gnu11 -Ofast -march=native $< $(shell pkg-config --cflags --libs gsl) -o $@

c.txt: bicubic
	./bicubic > $@

py.txt: bicubic.py
	./bicubic.py > $@

bench.txt: c.txt py.txt
	echo -n 'C  ' > $@
	cat c.txt >> $@
	echo -n 'Py ' > $@
	cat py.txt >> $@

clean:
	rm -f c.txt py.txt bench.txt bicubic
