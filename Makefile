CXX = g++
INCLUDE = -I./src/
CFLAGS = -c -g #-Wall

LDLIBS = -lOpenCL -lGL -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc -lsfml-system -lsfml-window -lsfml-graphics 
#LDLIBS = -lOpenCL -lGL -lsfml-system -lsfml-window -lsfml-graphics -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs 

EXECUTE = opencl-test

all: $(EXECUTE)

OBJS_E = opencl-test.o
#OBJS_E = occlude.o stimuli.o forest.o area.o compute-system.o compute-program.o input-image.o

$(EXECUTE): $(OBJS_E)
	$(CXX) $(OBJS_E) $(LDLIBS) -o $(EXECUTE)

# ============
# opencl-test
# ============
PATH_O = ./src

opencl-test.o: $(PATH_O)/opencl-test.cpp
	$(CXX) $(INCLUDE) $(CFLAGS) $(PATH_O)/opencl-test.cpp

# ==========
# Demo
# ==========
#PATH_D = ./demos/
#
#ball.o: $(PATH_D)ball/ball.cpp
#	$(CXX) $(INCLUDE) $(CFLAGS) $(PATH_D)ball/ball.cpp
#
#occlude.o: $(PATH_D)occlude/occlude.cpp
#	$(CXX) $(INCLUDE) $(CFLAGS) $(PATH_D)occlude/occlude.cpp
#
# ===========
# Cortex
# ===========
#PATH_A = ./source/cortex/
#
#stimuli.o: $(PATH_A)stimuli.cpp
#	$(CXX) $(INCLUDE) $(CFLAGS) $(PATH_A)stimuli.cpp
#
#forest.o: $(PATH_A)forest.cpp
#	$(CXX) $(INCLUDE) $(CFLAGS) $(PATH_A)forest.cpp
#
#area.o: $(PATH_A)area.cpp
#	$(CXX) $(INCLUDE) $(CFLAGS) $(PATH_A)area.cpp

# ==========
# Compute
# ==========
#PATH_C = ./source/compute/
#
#compute-system.o: $(PATH_C)compute-system.cpp
#	$(CXX) $(CFLAGS) $(PATH_C)compute-system.cpp
#
#compute-program.o: $(PATH_C)compute-program.cpp
#	$(CXX) $(CFLAGS) $(PATH_C)compute-program.cpp

# ==========
# Utils
# ==========
#PATH_U = ./source/utils/

#input-image.o: $(PATH_U)input-image.cpp
#	$(CXX) $(CFLAGS) $(PATH_U)input-image.cpp


# ==========
# Cleanup
# ==========
.PHONY : clean
clean:
	rm -rf *o $(EXECUTE)
