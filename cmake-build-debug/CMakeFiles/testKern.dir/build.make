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
include CMakeFiles/testKern.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/testKern.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/testKern.dir/flags.make

CMakeFiles/testKern.dir/test/testKern.cpp.o: CMakeFiles/testKern.dir/flags.make
CMakeFiles/testKern.dir/test/testKern.cpp.o: ../test/testKern.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/juraj/git/master_thesis/gp/GPc/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/testKern.dir/test/testKern.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/testKern.dir/test/testKern.cpp.o -c /home/juraj/git/master_thesis/gp/GPc/test/testKern.cpp

CMakeFiles/testKern.dir/test/testKern.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testKern.dir/test/testKern.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/juraj/git/master_thesis/gp/GPc/test/testKern.cpp > CMakeFiles/testKern.dir/test/testKern.cpp.i

CMakeFiles/testKern.dir/test/testKern.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testKern.dir/test/testKern.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/juraj/git/master_thesis/gp/GPc/test/testKern.cpp -o CMakeFiles/testKern.dir/test/testKern.cpp.s

CMakeFiles/testKern.dir/test/testKern.cpp.o.requires:

.PHONY : CMakeFiles/testKern.dir/test/testKern.cpp.o.requires

CMakeFiles/testKern.dir/test/testKern.cpp.o.provides: CMakeFiles/testKern.dir/test/testKern.cpp.o.requires
	$(MAKE) -f CMakeFiles/testKern.dir/build.make CMakeFiles/testKern.dir/test/testKern.cpp.o.provides.build
.PHONY : CMakeFiles/testKern.dir/test/testKern.cpp.o.provides

CMakeFiles/testKern.dir/test/testKern.cpp.o.provides.build: CMakeFiles/testKern.dir/test/testKern.cpp.o


# Object files for target testKern
testKern_OBJECTS = \
"CMakeFiles/testKern.dir/test/testKern.cpp.o"

# External object files for target testKern
testKern_EXTERNAL_OBJECTS =

testKern: CMakeFiles/testKern.dir/test/testKern.cpp.o
testKern: CMakeFiles/testKern.dir/build.make
testKern: libGPc.a
testKern: CMakeFiles/testKern.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/juraj/git/master_thesis/gp/GPc/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable testKern"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testKern.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/testKern.dir/build: testKern

.PHONY : CMakeFiles/testKern.dir/build

CMakeFiles/testKern.dir/requires: CMakeFiles/testKern.dir/test/testKern.cpp.o.requires

.PHONY : CMakeFiles/testKern.dir/requires

CMakeFiles/testKern.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/testKern.dir/cmake_clean.cmake
.PHONY : CMakeFiles/testKern.dir/clean

CMakeFiles/testKern.dir/depend:
	cd /home/juraj/git/master_thesis/gp/GPc/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/juraj/git/master_thesis/gp/GPc /home/juraj/git/master_thesis/gp/GPc /home/juraj/git/master_thesis/gp/GPc/cmake-build-debug /home/juraj/git/master_thesis/gp/GPc/cmake-build-debug /home/juraj/git/master_thesis/gp/GPc/cmake-build-debug/CMakeFiles/testKern.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/testKern.dir/depend
