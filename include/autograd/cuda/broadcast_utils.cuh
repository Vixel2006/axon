#ifndef BROADCAST_UTILS_CUH
#define BROADCAST_UTILS_CUH

#define MAX_DIMS 8 // Assuming max 8 dimensions for tensors

// Helper to get the index in a tensor, considering broadcasting from 'out_shape' to 'in_shape'
// linear_idx_out: linear index into the output tensor (which has the broadcasted shape)
// out_shape: shape of the output tensor
// out_ndim: number of dimensions of the output tensor
// in_shape: shape of the input tensor (the one we are calculating gradient for)
// in_strides: strides of the input tensor
// in_ndim: number of dimensions of the input tensor
__device__ __inline__ int get_broadcasted_input_idx(int linear_idx_out, const int* out_shape, int out_ndim,
                                         const int* in_shape, const int* in_strides, int in_ndim)
{
    int out_coords[MAX_DIMS];
    int in_coords[MAX_DIMS];

    // Convert linear_idx_out to multidimensional coordinates for out_shape
    int temp_idx = linear_idx_out;
    for (int d = out_ndim - 1; d >= 0; --d)
    {
        out_coords[d] = temp_idx % out_shape[d];
        temp_idx /= out_shape[d];
    }

    // Map out_coords to in_coords, handling broadcasting
    // This assumes right-alignment of dimensions for broadcasting
    int in_dim_offset = out_ndim - in_ndim;
    for (int d = 0; d < out_ndim; ++d)
    {
        int current_in_dim = d - in_dim_offset;
        if (current_in_dim >= 0)
        {
            if (in_shape[current_in_dim] == 1 && out_shape[d] > 1)
            {
                in_coords[current_in_dim] = 0; // Broadcasted dimension
            }
            else
            {
                in_coords[current_in_dim] = out_coords[d];
            }
        }
        // If out_ndim > in_ndim, leading dimensions of out_shape are new.
        // These correspond to a single element in the input, so their coordinate in input is 0.
        // This is implicitly handled by current_in_dim < 0.
    }

    // Calculate linear index in prev_grad using in_strides
    int final_in_idx = 0;
    for (int d = 0; d < in_ndim; ++d)
    {
        final_in_idx += in_coords[d] * in_strides[d];
    }
    return final_in_idx;
}

#endif // BROADCAST_UTILS_CUH
