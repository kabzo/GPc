# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/clion-2017.3.4/bin/cmake/bin/cmake

# The command to remove a file.
RM = /opt/clion-2017.3.4/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/juraj/git/master_thesis/gp/GPc

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/juraj/git/master_thesis/gp/GPc/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/testNoise.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/testNoise.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/testNoise.dir/flags.make

CMakeFiles/testNoise.dir/test/testNoise.cpp.o: CMakeFiles/testNoise.dir/flags.make
CMakeFiles/testNoise.dir/test/testNoise.cpp.o: ../test/testNoise.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/juraj/git/master_thesis/gp/GPc/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/testNoise.dir/test/testNoise.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/testNoise.dir/test/testNoise.cpp.o -c /home/juraj/git/master_thesis/gp/GPc/test/testNoise.cpp

CMakeFiles/testNoise.dir/test/testNoise.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testNoise.dir/test/testNoise.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/juraj/git/master_thesis/gp/GPc/test/testNoise.cpp > CMakeFiles/testNoise.dir/test/testNoise.cpp.i

CMakeFiles/testNoise.dir/test/testNoise.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testNoise.dir/test/testNoise.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/juraj/git/master_thesis/gp/GPc/test/testNoise.cpp -o CMakeFiles/testNoise.dir/test/testNoise.cpp.s

CMakeFiles/testNoise.dir/test/testNoise.cpp.o.requires:

.PHONY : CMakeFiles/testNoise.dir/test/testNoise.cpp.o.requires

CMakeFiles/testNoise.dir/test/testNoise.cpp.o.provides: CMakeFiles/testNoise.dir/test/testNoise.cpp.o.requires
	$(MAKE) -f CMakeFiles/testNoise.dir/build.make CMakeFiles/testNoise.dir/test/testNoise.cpp.o.provides.build
.PHONY : CMakeFiles/testNoise.dir/test/testNoise.cpp.o.provides

CMakeFiles/testNoise.dir/test/testNoise.cpp.o.provides.build: CMakeFiles/testNoise.dir/test/testNoise.cpp.o


# Object files for target testNoise
testNoise_OBJECTS = \
"CMakeFiles/testNoise.dir/test/testNoise.cpp.o"

# External object files for target testNoise
testNoise_EXTERNAL_OBJECTS =

testNoise: CMakeFiles/testNoise.dir/test/testNoise.cpp.o
testNoise: CMakeFiles/testNoise.dir/build.make
testNoise: libGPc.a
testNoise: CMakeFiles/testNoise.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/juraj/git/master_thesis/gp/GPc/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable testNoise"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testNoise.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/testNoise.dir/build: testNoise

.PHONY : CMakeFiles/testNoise.dir/build

CMakeFiles/testNoise.dir/requires: CMakeFiles/testNoise.dir/test/testNoise.cpp.o.requires

.PHONY : CMakeFiles/testNoise.dir/requires

CMakeFiles/testNoise.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/testNoise.dir/cmake_clean.cmake
.PHONY : CMakeFiles/testNoise.dir/clean

CMakeFiles/testNoise.dir/depend:
	cd /home/juraj/git/master_thesis/gp/GPc/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/juraj/git/master_thesis/gp/GPc /home/juraj/git/master_thesis/gp/GPc /home/juraj/git/master_thesis/gp/GPc/cmake-build-debug /home/juraj/git/master_thesis/gp/GPc/cmake-build-debug /home/juraj/git/master_thesis/gp/GPc/cmake-build-debug/CMakeFiles/testNoise.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/testNoise.dir/depend
