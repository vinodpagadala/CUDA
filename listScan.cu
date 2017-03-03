#include <stdio.h>
#include <stdlib.h>
#include <helper_timer.h>

// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// + lst[n-1]}

#define BLOCK_SIZE 512 //@@ You can change this


__global__ void setInput(int *inp, int *aux, int length) {
    unsigned int tid = threadIdx.x; 
unsigned int sIndex = 2 * blockIdx.x * BLOCK_SIZE;
    if (blockIdx.x) {
       if (sIndex + tid < length)
          inp[sIndex+ tid] += aux[blockIdx.x ];
       if (sIndex+ BLOCK_SIZE + tid < length)
        inp[sIndex+ BLOCK_SIZE + tid] += aux[blockIdx.x ];
    }
}
__global__ void scan(int *input, int *output, int *aux, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from here
  
  __shared__ float sArray[BLOCK_SIZE << 1];
    unsigned int tid = threadIdx.x;
unsigned int sIndex = 2 * blockIdx.x * BLOCK_SIZE;
    if (sIndex + tid < len)
       sArray[tid] = input[sIndex + tid];
    else
      sArray[tid] = 0;
    if (sIndex + BLOCK_SIZE + tid < len)
       sArray[BLOCK_SIZE + tid] = input[sIndex + BLOCK_SIZE + tid];
    else
       sArray[BLOCK_SIZE + tid] = 0;
    __syncthreads();

    // Reduction
    int str;
    for (str = 1; str <= BLOCK_SIZE; str <<= 1) {
       int index = (tid + 1) * str * 2 - 1;
       if (index < 2 * BLOCK_SIZE)
          sArray[index] += sArray[index - str];
       __syncthreads();
    }

    // Post reduction
    for (str = BLOCK_SIZE >> 1; str; str >>= 1) {
       int index = (tid + 1) * str * 2 - 1;
       if (index + str < 2 * BLOCK_SIZE)
          sArray[index + str] += sArray[index];
       __syncthreads();
    }

    if (sIndex + tid < len)
       output[sIndex + tid] = sArray[tid];
    if (sIndex + BLOCK_SIZE + tid < len)
       output[sIndex + BLOCK_SIZE + tid] = sArray[BLOCK_SIZE + tid];

    if (aux && tid == 0)
       aux[blockIdx.x] = sArray[2 * BLOCK_SIZE - 1];
  
}

int main(int argc, char **argv) {
  int *hostInput;  // The input 1D list
  int *hostOutput; // The output list
  int *expectedOutput;
  int *deviceInput;
  int *deviceOutput;
  int *deviceAuxArray, *deviceAuxScannedArray;
  int numElements; // number of elements in the list
  
  FILE *infile, *outfile;
  int inputLength, outputLength;
  StopWatchLinux stw;
  unsigned int blog = 1;

  // Import host input data
  stw.start();
  if ((infile = fopen("input.raw", "r")) == NULL)
  { printf("Cannot open input.raw.\n"); exit(EXIT_FAILURE); }
  fscanf(infile, "%i", &inputLength);
  hostInput = (int *)malloc(sizeof(int) * inputLength);
  for (int i = 0; i < inputLength; i++)
     fscanf(infile, "%i", &hostInput[i]);
  fclose(infile);
  numElements = inputLength;
  hostOutput = (int *)malloc(numElements * sizeof(int));
  stw.stop();
  printf("Importing data and creating memory on host: %f ms\n", stw.getTime());

  if (blog) printf("*** The number of input elements in the input is %i\n", numElements);

  stw.reset();
  stw.start();
  
  cudaMalloc((void **)&deviceInput, numElements * sizeof(int));
  cudaMalloc((void **)&deviceOutput, numElements * sizeof(int));

  cudaMalloc(&deviceAuxArray, (BLOCK_SIZE << 1) * sizeof(int));
  cudaMalloc(&deviceAuxScannedArray, (BLOCK_SIZE << 1) * sizeof(int));
  
  stw.stop();
  printf("Allocating GPU memory: %f ms\n", stw.getTime());

  stw.reset();
  stw.start();
  
  cudaMemset(deviceOutput, 0, numElements * sizeof(int));
  
  stw.stop();
  printf("Clearing output memory: %f ms\n", stw.getTime());

  stw.reset();
  stw.start();
  
  cudaMemcpy(deviceInput, hostInput, numElements * sizeof(int),
                     cudaMemcpyHostToDevice);

  stw.stop();
  printf("Copying input memory to the GPU: %f ms\n", stw.getTime());

  //@@ Initialize the grid and block dimensions here
  int gridSize = ceil((float)numElements/(BLOCK_SIZE<<1));
  dim3 dimGrid(gridSize, 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  stw.reset();
  stw.start();
  
  //@@ Modify this to complete the functionality of the scan
  //@@ on the device
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, deviceAuxArray, numElements);
  scan<<<dim3(1,1,1), dimBlock>>>(deviceAuxArray, deviceAuxScannedArray, NULL, BLOCK_SIZE << 1);
  setInput<<<dimGrid, dimBlock>>>(deviceOutput, deviceAuxScannedArray, numElements);

  cudaDeviceSynchronize();
 
  stw.stop();
  printf("Performing CUDA computation: %f ms\n", stw.getTime());

  stw.reset();
  stw.start();
  
  cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(int),
                     cudaMemcpyDeviceToHost);
  
  stw.stop();
  printf("Copying output memory to the CPU: %f ms\n", stw.getTime());

  stw.reset();
  stw.start();
  
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceAuxArray);
  cudaFree(deviceAuxScannedArray);

  stw.stop();
  printf("Freeing GPU Memory: %f ms\n", stw.getTime());

  if ((outfile = fopen("output.raw", "r")) == NULL)
  { printf("Cannot open output.raw.\n"); exit(EXIT_FAILURE); }
  fscanf(outfile, "%i", &outputLength);
  expectedOutput = (int *)malloc(sizeof(int) * outputLength);  
  for (int i = 0; i < outputLength; i++)
     fscanf(outfile, "%i", &expectedOutput[i]);	
  fclose(outfile);
  
  int test = 1;
for (int i=0;i<outputLength;i++)
  {
      printf("%i\n",hostOutput[i]);
  }
  for (int i = 0; i < outputLength; i++) {
     if (expectedOutput[i] != hostOutput[i])
        printf("%i %i %i\n", i, expectedOutput[i], hostOutput[i]);
     test = test && (expectedOutput[i] == hostOutput[i]);
  }
  
  if (test) printf("Results correct.\n");
  else printf("Results incorrect.\n");

  free(hostInput);
  cudaFreeHost(hostOutput);
  free(expectedOutput);

  return 0;
}
