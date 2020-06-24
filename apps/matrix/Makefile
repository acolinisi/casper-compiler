# TODO: move this into apps/matrix/CMakeLists.txt

# Built by cmake
BUILD_DIR=../../build/apps/matrix

CC:=clang
CXX:=clang++
AR:=ar
CFLAGS:=-O0 -g

all: matrix

OBJ_LIBKERN:=kern.o
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<
libkern.a: $(OBJ_LIBKERN)
	$(AR) -crs $@ $^

# run twice, the first time just to show errors
matrix.ll: $(BUILD_DIR)/matrix.meta
	$<
	$< 2>$@
%.o: %.ll
	llc -filetype=obj -o $@ $<
matrix: matrix.o libkern.a
	$(CC) $(CFLAGS) -fPIE -o $@ matrix.o -L. -lkern

run: matrix
	./matrix

clean:
	rm -f *.o *.ll matrix

.SECONDARY:
