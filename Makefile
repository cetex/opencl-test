CXX = g++
INCLUDE = -I./src/
CFLAGS = -c -g -O0 -D __CL_HPP_ENABLE_EXCEPTIONS -Wall
LINKFLAGS = -g
SDIR=src
ODIR=build
OUT = opencl-test

LDLIBS = -lOpenCL -lGL -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc -lsfml-system -lsfml-window -lsfml-graphics 

_OBJS = compute/compute-system.o compute/compute-program.o architect/architect.o inputlayer/camera.o opencl-test.o
OBJS = $(patsubst %,$(ODIR)/%,$(_OBJS))

$(ODIR)/%.o: $(SDIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(INCLUDE) -o $@ $< $(CFLAGS)

$(OUT): $(OBJS)
	@mkdir -p $(@D)
	$(CXX) $(LINKFLAGS) $(OBJS) $(LDLIBS) -o $(OUT)

# ==========
# Cleanup
# ==========
.PHONY : clean
clean:
	rm -rf build/* $(OUT)
