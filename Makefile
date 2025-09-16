CXX := g++
TARGET := build/opencv_example
BUILD_DIR := build
SRC := Binary.cpp \
       CameraCalibration.cpp \
       Edges.cpp \
       Features.cpp \
       Geometric.cpp \
       Histograms.cpp \
       Images.cpp \
       MyApplication.cpp \
       Recognition.cpp \
       Region.cpp \
       Utilities.cpp \
       Video.cpp \
       main.cpp

OBJ := $(addprefix $(BUILD_DIR)/,$(SRC:.cpp=.o))
DEP := $(OBJ:.o=.d)

OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS := $(shell pkg-config --libs opencv4)

CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic $(OPENCV_CFLAGS)
LDFLAGS := $(OPENCV_LIBS)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ) | $(BUILD_DIR)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

-include $(DEP)
