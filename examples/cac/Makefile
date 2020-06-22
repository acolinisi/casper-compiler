BIN_DIR?=../../build/bin/
CC:=clang
AR:=ar
CFLAGS:=-O0 -g

all: app

OBJ_LIBKERN:=kern.o
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<
libkern.a: $(OBJ_LIBKERN)
	$(AR) -crs $@ $^

# run twice, the first time just to show errors
%.ll: %.mlir
	$(BIN_DIR)cac -emit=llvm $<
	$(BIN_DIR)cac -emit=llvm $< 2>$@
%.mll: %.mlir
	$(BIN_DIR)cac -emit=mlir-llvm $<
	$(BIN_DIR)cac -emit=mlir-llvm $< 2>$@
%.o: %.ll
	llc -filetype=obj -o $@ $<
app: app.o libkern.a
	$(CC) $(CFLAGS) -fPIE -o $@ app.o -L. -lkern

run: app
	./app

clean:
	rm -f *.o *.ll app

.SECONDARY:
