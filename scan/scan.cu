#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}


// This is the CUDA "kernel" function that is run on the GPU.  You
// know this because it is marked as a __global__ function.
__global__ void
upsweep_kernel(int N, int two_d, int two_dplus1, int* output) {

    // compute overall thread index from position of thread in current
    // block, and given the block we are in (in this example only a 1D
    // calculation is needed so the code only looks at the .x terms of
    // blockDim and threadIdx.
    long long index = ((long long) blockIdx.x * (long long) blockDim.x + (long long) threadIdx.x)*(long long) two_dplus1;
    // if (index == N-1){
    //     printf("last num %d\n", output[index]);
    // }

    if (index < N)
        output[(long long) index+two_dplus1-1] += output[(long long) index+two_d-1];

}

__global__ void
downsweep_kernel(int N, int two_d, int two_dplus1, int* output) {

    // compute overall thread index from position of thread in current
    // block, and given the block we are in (in this example only a 1D
    // calculation is needed so the code only looks at the .x terms of
    // blockDim and threadIdx.
    long long index = ((long long) blockIdx.x * (long long) blockDim.x + (long long) threadIdx.x)* (long long) two_dplus1;


    // this check is necessary to make the code work for values of N
    // that are not a multiple of the thread block size (blockDim.x)
    if (index < N) {
        int t = output[(long long) index+two_d-1];
        output[(long long) index+two_d-1] = output[(long long) index+two_dplus1-1];
        output[(long long) index+two_dplus1-1] += t;
    }
}

__global__ void
single_update_kernel(int index, int* output, int value){
    output[index] = value;
}


void print_array2(int* array, int N) {
    printf("[");
    for(int i = 0; i < N; i++) {
        printf("%d, ", array[i]);
    }
    printf("]\n");
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int* input, int N, int* result)
{

    // CS149 TODO:
    //
    // Implement your exclusive scan implementation here.  Keep in
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.

    // round up N
    // int* awef = new int[10000];
    // cudaMemcpy(awef, result, 10000 * sizeof(int), cudaMemcpyDeviceToHost);
    // print_array2(awef,10000);
    int rounded_N = nextPow2(N);
    // printf("rounded_N: %d\n", nextPow2(N));
    // cudaMemset(result+N, 0, (rounded_N-N)*sizeof(int));
    const int blocks = (rounded_N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // printf("block numbers: %d\n", blocks);
    for (int two_d = 1; two_d <= rounded_N/2; two_d *= 2) {
        int two_dplus1 = 2*two_d;
        upsweep_kernel<<<blocks, THREADS_PER_BLOCK>>>(rounded_N, two_d, two_dplus1, result);
    }
    single_update_kernel<<<1,1>>>(rounded_N-1, result, 0);

    // cudaMemcpy(awef, result, rounded_N * sizeof(int), cudaMemcpyDeviceToHost);
    // print_array2(awef,rounded_N);

    for (int two_d = rounded_N/2; two_d >= 1; two_d /= 2) {
        int two_dplus1 = 2*two_d;
        downsweep_kernel<<<blocks, THREADS_PER_BLOCK>>>(rounded_N, two_d, two_dplus1, result);
    }
    // cudaMemcpy(awef, result, 10000 * sizeof(int), cudaMemcpyDeviceToHost);
    // printf("result");
    // print_array2(awef, 10000);
}



//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}

__global__ void
mark_repeats_kernel(int N, int* input, int* output) {
    int index = (long long) blockIdx.x* (long long) blockDim.x+ (long long) threadIdx.x;
    if (index+1 < N) {
        output[index] = input[index] == input[index+1] ? 1 : 0;
    }
}

__global__ void
move_to_indices_kernel(int N, int* flags, int* indices, int* output, int* d_sharedInteger) {
    int index = (long long) blockIdx.x* (long long) blockDim.x+(long long) threadIdx.x;
    if (index+1 < N && flags[index]) {
        atomicAdd(d_sharedInteger, 1);
        output[indices[index]] = index;
    }
}
// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.

    int rounded_length = nextPow2(length);

    int blocks = (rounded_length+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    int* repeat_flags;
    cudaMalloc((void **)&repeat_flags, rounded_length * sizeof(int));


    mark_repeats_kernel<<<blocks, THREADS_PER_BLOCK>>>(rounded_length, device_input, repeat_flags);
    // int* awef = new int[length];
    // printf("[repeat_flags]");
    // cudaMemcpy(awef, repeat_flags, length * sizeof(int), cudaMemcpyDeviceToHost);
    // print_array2(awef, length);

    int* output_indices;
    cudaMalloc((void **)&output_indices, rounded_length * sizeof(int));
    cudaMemcpy(output_indices, repeat_flags, rounded_length*sizeof(int), cudaMemcpyDeviceToDevice);
    exclusive_scan(device_input, rounded_length, output_indices);
    // printf("[ouput_indices]");
    // cudaMemcpy(awef, output_indices, length * sizeof(int), cudaMemcpyDeviceToHost);
    // print_array2(awef, length);

    // Host variable to store the result
    int h_sharedInteger = 0;

    // Device variable
    int *d_sharedInteger;
    cudaMalloc(&d_sharedInteger, sizeof(int));
    cudaMemcpy(d_sharedInteger, &h_sharedInteger, sizeof(int), cudaMemcpyHostToDevice);


    move_to_indices_kernel<<<blocks, THREADS_PER_BLOCK>>>(length, repeat_flags, output_indices, device_output, d_sharedInteger);
     // Copy the result back to the host
    cudaMemcpy(&h_sharedInteger, d_sharedInteger, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_sharedInteger);

    // printf("Shared Integer after atomic adds: %d\n", h_sharedInteger);

    
    // printf("[device_output]");
    // cudaMemcpy(awef, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);
    // print_array2(awef, length);
    return h_sharedInteger; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
