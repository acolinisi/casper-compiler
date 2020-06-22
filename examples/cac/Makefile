BIN_DIR?=../../build/bin/

all: app

%.ll: %.mlir
	$(BIN_DIR)cac -emit=llvm $< 2>$@
%.o: %.ll
	llc -filetype=obj -o $@ $<
app: app.o
	clang -o $@ $^

clean:
	rm -f *.o *.ll app

.SECONDARY:
