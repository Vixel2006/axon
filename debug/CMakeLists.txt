add_executable(debug_tensor_cuda debug_tensor_cuda.cu)
target_link_libraries(debug_tensor_cuda PRIVATE axon)
target_include_directories(debug_tensor_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)