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
include CMakeFiles/testDist.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/testDist.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/testDist.dir/flags.make

CMakeFiles/testDist.dir/test/testDist.cpp.o: CMakeFiles/testDist.dir/flags.make
CMakeFiles/testDist.dir/test/testDist.cpp.o: ../test/testDist.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/juraj/git/master_thesis/gp/GPc/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/testDist.dir/test/testDist.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/testDist.dir/test/testDist.cpp.o -c /home/juraj/git/master_thesis/gp/GPc/test/testDist.cpp

CMakeFiles/testDist.dir/test/testDist.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testDist.dir/test/testDist.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/juraj/git/master_thesis/gp/GPc/test/testDist.cpp > CMakeFiles/testDist.dir/test/testDist.cpp.i

CMakeFiles/testDist.dir/test/testDist.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testDist.dir/test/testDist.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/juraj/git/master_thesis/gp/GPc/test/testDist.cpp -o CMakeFiles/testDist.dir/test/testDist.cpp.s

CMakeFiles/testDist.dir/test/testDist.cpp.o.requires:

.PHONY : CMakeFiles/testDist.dir/test/testDist.cpp.o.requires

CMakeFiles/testDist.dir/test/testDist.cpp.o.provides: CMakeFiles/testDist.dir/test/testDist.cpp.o.requires
	$(MAKE) -f CMakeFiles/testDist.dir/build.make CMakeFiles/testDist.dir/test/testDist.cpp.o.provides.build
.PHONY : CMakeFiles/testDist.dir/test/testDist.cpp.o.provides

CMakeFiles/testDist.dir/test/testDist.cpp.o.provides.build: CMakeFiles/testDist.dir/test/testDist.cpp.o


# Object files for target testDist
testDist_OBJECTS = \
"CMakeFiles/testDist.dir/test/testDist.cpp.o"

# External object files for target testDist
testDist_EXTERNAL_OBJECTS =

testDist: CMakeFiles/testDist.dir/test/testDist.cpp.o
testDist: CMakeFiles/testDist.dir/build.make
testDist: libGPc.a
testDist: CMakeFiles/testDist.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/juraj/git/master_thesis/gp/GPc/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable testDist"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testDist.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/testDist.dir/build: testDist

.PHONY : CMakeFiles/testDist.dir/build

CMakeFiles/testDist.dir/requires: CMakeFiles/testDist.dir/test/testDist.cpp.o.requires

.PHONY : CMakeFiles/testDist.dir/requires

CMakeFiles/testDist.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/testDist.dir/cmake_clean.cmake
.PHONY : CMakeFiles/testDist.dir/clean

CMakeFiles/testDist.dir/depend:
	cd /home/juraj/git/master_thesis/gp/GPc/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/juraj/git/master_thesis/gp/GPc /home/juraj/git/master_thesis/gp/GPc /home/juraj/git/master_thesis/gp/GPc/cmake-build-debug /home/juraj/git/master_thesis/gp/GPc/cmake-build-debug /home/juraj/git/master_thesis/gp/GPc/cmake-build-debug/CMakeFiles/testDist.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/testDist.dir/depend

