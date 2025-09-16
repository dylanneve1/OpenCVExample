# Makefile for OpenCV Example Project

# Compiler settings
CXX := clang++
CXXFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -ffunction-sections -fdata-sections
CXXFLAGS_DEBUG := -O0 -g -D_DEBUG -D_CONSOLE
CXXFLAGS_RELEASE := -O3 -DNDEBUG -D_CONSOLE

# Directories
SRC_DIR := src
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj

# OpenCV configuration
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
OPENCV_LIBS := $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv)

# Include directories
INCLUDES := -I$(SRC_DIR) $(OPENCV_CFLAGS)

# Source files
OPENCV_EXAMPLE_SOURCES := \
	$(SRC_DIR)/Binary.cpp \
	$(SRC_DIR)/CameraCalibration.cpp \
	$(SRC_DIR)/Edges.cpp \
	$(SRC_DIR)/Features.cpp \
	$(SRC_DIR)/Geometric.cpp \
	$(SRC_DIR)/Histograms.cpp \
	$(SRC_DIR)/Images.cpp \
	$(SRC_DIR)/main.cpp \
	$(SRC_DIR)/MyApplication.cpp \
	$(SRC_DIR)/Recognition.cpp \
	$(SRC_DIR)/Region.cpp \
	$(SRC_DIR)/Utilities.cpp \
	$(SRC_DIR)/Video.cpp

WATERSHED_SOURCES := \
	$(SRC_DIR)/watershed_main.cpp \
	$(SRC_DIR)/Utilities.cpp

# Object files
OPENCV_EXAMPLE_OBJECTS := $(OPENCV_EXAMPLE_SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
WATERSHED_OBJECTS := $(WATERSHED_SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/watershed_%.o)

# Executables
OPENCV_EXAMPLE_TARGET := $(BUILD_DIR)/opencv-example
WATERSHED_TARGET := $(BUILD_DIR)/watershed

# Build type (default: debug)
BUILD_TYPE ?= debug

ifeq ($(BUILD_TYPE),release)
    CXXFLAGS += $(CXXFLAGS_RELEASE)
else
    CXXFLAGS += $(CXXFLAGS_DEBUG)
endif

# Linker flags
LDFLAGS := -Wl
LIBS := $(OPENCV_LIBS)

# Default target
.PHONY: all
all: $(OPENCV_EXAMPLE_TARGET) $(WATERSHED_TARGET)

# Create directories
$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Build opencv-example executable
$(OPENCV_EXAMPLE_TARGET): $(OPENCV_EXAMPLE_OBJECTS) | $(BUILD_DIR)
	@echo "Linking $@..."
	@$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)
	@echo "Built $@"

# Build watershed executable
$(WATERSHED_TARGET): $(WATERSHED_OBJECTS) | $(BUILD_DIR)
	@echo "Linking $@..."
	@$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)
	@echo "Built $@"

# Compile opencv-example object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@echo "Compiling $<..."
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile watershed object files (with different naming to avoid conflicts)
$(OBJ_DIR)/watershed_%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@echo "Compiling $< (watershed)..."
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)

# Clean and rebuild
.PHONY: rebuild
rebuild: clean all

# Build release version
.PHONY: release
release:
	@$(MAKE) BUILD_TYPE=release

# Build debug version (explicit)
.PHONY: debug
debug:
	@$(MAKE) BUILD_TYPE=debug

# Run opencv-example
.PHONY: run
run: $(OPENCV_EXAMPLE_TARGET)
	@$(OPENCV_EXAMPLE_TARGET)

# Run watershed
.PHONY: run-watershed
run-watershed: $(WATERSHED_TARGET)
	@$(WATERSHED_TARGET)

# Print help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all           - Build both executables (default)"
	@echo "  clean         - Remove build artifacts"
	@echo "  rebuild       - Clean and rebuild"
	@echo "  debug         - Build debug version (default)"
	@echo "  release       - Build release version"
	@echo "  run           - Build and run opencv-example"
	@echo "  run-watershed - Build and run watershed"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make                    # Build debug version"
	@echo "  make release            # Build release version"
	@echo "  make BUILD_TYPE=release # Alternative release build"
	@echo "  make clean              # Clean build directory"

# Dependency tracking (optional but recommended)
-include $(OPENCV_EXAMPLE_OBJECTS:.o=.d)
-include $(WATERSHED_OBJECTS:.o=.d)

$(OBJ_DIR)/%.d: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -MM -MT $(OBJ_DIR)/$*.o $< -MF $@

$(OBJ_DIR)/watershed_%.d: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -MM -MT $(OBJ_DIR)/watershed_$*.o $< -MF $@
