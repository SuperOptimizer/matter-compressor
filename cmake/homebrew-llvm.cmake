# Toolchain: Homebrew LLVM end-to-end (clang + lld + llvm-ar/ranlib).
#   cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=cmake/homebrew-llvm.cmake \
#         [-DMC_THINLTO=ON -DMC_PGO_USE=/path/merged.profdata]
set(HOMEBREW_LLVM /opt/homebrew/opt/llvm)
set(CMAKE_C_COMPILER   ${HOMEBREW_LLVM}/bin/clang)
set(CMAKE_CXX_COMPILER ${HOMEBREW_LLVM}/bin/clang++)
set(CMAKE_AR           ${HOMEBREW_LLVM}/bin/llvm-ar)
set(CMAKE_RANLIB       ${HOMEBREW_LLVM}/bin/llvm-ranlib)
set(CMAKE_NM           ${HOMEBREW_LLVM}/bin/llvm-nm)
add_link_options(-fuse-ld=/opt/homebrew/opt/lld/bin/ld64.lld)
