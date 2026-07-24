set(HNSWLIB_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party/hnswlib)

foreach(bl pre_filter post_filter in_filter)
    add_executable(${bl} baselines/${bl}.cpp)
    target_include_directories(${bl} PRIVATE ${HNSWLIB_INCLUDE_DIR} includes)
    target_link_libraries(${bl} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_features(${bl} PRIVATE cxx_std_17)
endforeach()
