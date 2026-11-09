cmake_minimum_required(VERSION 3.16)
project(ngs_mpi)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(MPI REQUIRED)
find_package(GTest REQUIRED)

include_directories(include ${MPI_CXX_INCLUDE_PATH} ${GTEST_INCLUDE_DIRS})

# main executable
add_executable(ngs_mpi
    src/main.cpp
    src/csr.cpp
    src/local_graph.cpp
    src/delta_stepping.cpp
    src/sssp_reference.cpp
)
target_link_libraries(ngs_mpi ${MPI_CXX_LIBRARIES})

# synthetic graph generator (no MPI)
add_executable(gen_graph
    src/gen_graph.cpp
    src/csr.cpp
)

# unit tests (no MPI)
add_executable(unit_tests
    tests/unit_tests.cpp
    src/csr.cpp
    src/local_graph.cpp
    src/sssp_reference.cpp
)
target_link_libraries(unit_tests ${GTEST_LIBRARIES} pthread)

# mpi tests
add_executable(mpi_tests
    tests/mpi_tests.cpp
    src/csr.cpp
    src/local_graph.cpp
    src/delta_stepping.cpp
    src/sssp_reference.cpp
)
target_link_libraries(mpi_tests ${MPI_CXX_LIBRARIES} ${GTEST_LIBRARIES} pthread)