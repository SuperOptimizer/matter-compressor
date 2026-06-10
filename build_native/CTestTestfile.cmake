# CMake generated Testfile for 
# Source directory: /home/forrest/matter-compressor
# Build directory: /home/forrest/matter-compressor/build_native
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(roundtrip "/home/forrest/matter-compressor/build_native/mc_roundtrip")
set_tests_properties(roundtrip PROPERTIES  _BACKTRACE_TRIPLES "/home/forrest/matter-compressor/CMakeLists.txt;42;add_test;/home/forrest/matter-compressor/CMakeLists.txt;0;")
add_test(append_roundtrip "/home/forrest/matter-compressor/build_native/mc_append_roundtrip")
set_tests_properties(append_roundtrip PROPERTIES  _BACKTRACE_TRIPLES "/home/forrest/matter-compressor/CMakeLists.txt;47;add_test;/home/forrest/matter-compressor/CMakeLists.txt;0;")
