// OpenCL kernel for vector addition.

// Using constant types of memory for read-only operations in order to optimise the global memory

__kernel
void vectorComputation( __constant float *a, __constant float *b, __global float *c )
{
	// The global id tells us the index of the vector for this thread.
	int gradients_id = get_local_id(0);
	int inputs_id = get_local_id(1);
	int global_size_M = get_local_size(1);



	// Perform the addition.

	c[gradients_id * global_size_M + inputs_id] += a[gradients_id] * b[inputs_id];
}
