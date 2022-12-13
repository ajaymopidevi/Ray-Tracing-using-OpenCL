# final
EXE=final

# Main target
all: $(EXE)

#  Msys/MinGW
ifeq "$(OS)" "Windows_NT"
CFLG=-O3 -Wall -fopenmp -DUSEGLEW
LIBS=-lglfw3 -lglew32 -lglu32 -lopengl32 -lm
CLEAN=rm -f *.exe *.o *.a
else
#  OSX
ifeq "$(shell uname)" "Darwin"
CFLG=-O3 -Wall -Wno-deprecated-declarations  -DUSEGLEW
LIBS=-lglfw -lglew -framework Cocoa -framework OpenGL -framework IOKit
#  Linux/Unix/Solaris
else
CFLG=-O3 -Wall -fopenmp
LIBS=-lglfw -lGLU -lGL -lm
endif
#  OSX/Linux/Unix/Solaris
CLEAN=rm -f $(EXE) *.o *.a
endif

# Dependencies
InitGPUcl.o: InitGPUcl.cpp InitGPUcl.h
fatal.o: fatal.c CSCIx239.h
errcheck.o: errcheck.c CSCIx239.h
print.o: print.c CSCIx239.h
axes.o: axes.c CSCIx239.h
loadtexbmp.o: loadtexbmp.cpp CSCIx239.h
loadobj.o: loadobj.cpp CSCIx239.h RayTrace.h
final.o: final.cpp CSCIx239.h RayTrace.h InitGPUcl.h loadobj.o
elapsed.o: elapsed.c CSCIx239.h
fps.o: fps.c CSCIx239.h
mat4.o: mat4.c CSCIx239.h
initwin.o: initwin.c CSCIx239.h

#  Create archive
CSCIx239.a:fatal.o errcheck.o print.o axes.o loadtexbmp.o loadobj.o elapsed.o fps.o mat4.o initwin.o
	ar -rcs $@ $^

# Compile rules
.c.o:
	gcc -c $(CFLG)  $<
.cpp.o:
	g++ -c $(CFLG)  $<

#  Link
final:final.o InitGPUcl.o CSCIx239.a
	g++ $(CFLG) -o $@ $^  -lOpenCL $(LIBS)

#  Clean
clean:
	$(CLEAN)
