main: main.c
	gcc -O3 -march=native -mavx2 -ffast-math -funroll-loops -shared -o libhungarian.dll hungarian_avx.c

clean:
	/bin/rm -f gal