/*
CUDA - generate array of random numbers and calculate occurence of odd and even numbers - with streams
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

#define MAX 1000

__host__
void errorexit(const char *s) {
    printf("\n%s",s); 
    exit(EXIT_FAILURE);   
}

__global__ void generate(unsigned int seed, curandState_t* states, int* random) {
  int my_index=blockIdx.x*blockDim.x+threadIdx.x;
  curand_init(seed,my_index,0,&states[my_index]);
  random[my_index]=curand(&states[my_index]) % MAX;
}

__global__ 
void checkPrime(int *random, int *result) {
    int my_index=blockIdx.x*blockDim.x+threadIdx.x;
    int divCount = 0;
    float temp = sqrt((float)random[my_index]);
    for(int i = 2; i < (int)temp;i++)
    {
    	if(random[my_index]%i == 0)
    	{
    		divCount += 1;
    	}
    }
    
   if(divCount == 0)
   {
   atomicAdd(&result[0],1);
   }
   else
   {
   atomicAdd(&result[1],1);
   }
}

__global__ 
void calculateOccurance(int *random, int *hist) {
    int my_index=blockIdx.x*blockDim.x+threadIdx.x;
    int index = random[my_index];
   atomicAdd(&hist[index],1);
}


int main(int argc,char **argv) {
    int threadsinblock=1024;
    int blocksingrid=10000; 

    int size = threadsinblock*blocksingrid;
    
    //how many streams will be used
    int streamCnt = 3;
    
    //memory allocation host
    int *hresults=NULL;
    cudaMallocHost((void **)&hresults, 2*sizeof(int));
    
    int *hhistogram=NULL;
    cudaMallocHost((void **)&hhistogram, MAX*sizeof(int));
    //int *hhistogram=(int*)malloc(MAX*sizeof(int));


    //int *hresults=(int*)malloc(2*sizeof(int));
    int *hrandoms=(int*)malloc(size*sizeof(int));
    
    //create pointer to streams
    cudaStream_t streams[streamCnt];

    curandState_t* states;
    //memory allocation for generator states
    cudaMalloc((void**) &states, size * sizeof(curandState_t));

    //memory allocation for randoms
    int *drandom=NULL;
    if (cudaSuccess!=cudaMalloc((void **)&drandom,size*sizeof(int)))
      errorexit("Error allocating memory on the GPU");
    
    //memory allocation for results 
    int *dresults=NULL;
    if (cudaSuccess!=cudaMalloc((void **)&dresults,2*sizeof(int)))
      errorexit("Error allocating memory on the GPU");
      
    // mem alloc for hist
    int *dhistogram=NULL;
    if (cudaSuccess!=cudaMalloc((void **)&dhistogram,MAX*sizeof(int)))
      errorexit("Error allocating memory on the GPU");

    //results memory initialize with 0
    if (cudaSuccess!=cudaMemset(dresults,0, 2*sizeof(int)))
      errorexit("Error initializing memory on the GPU");

    //hist memory initialize with 0
    if (cudaSuccess!=cudaMemset(dhistogram,0, MAX*sizeof(int)))
      errorexit("Error initializing memory on the GPU");
      
    //create streams
    int i;
    for(i=0;i<streamCnt;i++) {
      if (cudaSuccess!=cudaStreamCreate(&streams[i]))
           errorexit("Error creating stream");
    }

    //generate numbers in stream 0
    generate<<<blocksingrid,threadsinblock, threadsinblock*sizeof(double), streams[0]>>>(time(0),states, drandom);
    
    //oczekiwanie na zakończenie prac przez strumien 0
    cudaStreamSynchronize(streams[0]);

    //calculate prime count - stream 1
    checkPrime<<<blocksingrid,threadsinblock,threadsinblock*sizeof(double), streams[1]>>>(drandom, dresults);
    
    //calculate hist - stream 2
    calculateOccurance<<<blocksingrid,threadsinblock, threadsinblock*sizeof(double), streams[2]>>>(drandom, dhistogram);
    
    //asynchronic copy of random numbers
    cudaMemcpyAsync(hrandoms,drandom,size*sizeof(int),cudaMemcpyDeviceToHost, streams[0]);
    
    //wait for task on streams 1 and 2
    cudaStreamSynchronize(streams[1]);
    cudaStreamSynchronize(streams[2]);
    
    //asynchronic copy of results from device to host
    if (cudaSuccess!=cudaMemcpyAsync(hresults,dresults,2*sizeof(int),cudaMemcpyDeviceToHost, streams[0]))
       errorexit("Error copying results");
       
    //asynchronic copy of hist from device to host
    if (cudaSuccess!=cudaMemcpyAsync(hhistogram,dhistogram,MAX*sizeof(int),cudaMemcpyDeviceToHost, streams[1]))
       errorexit("Error copying results");
    
    //wait for stream 0 to end its task
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    for(int i = 0; i < MAX; i++)
    {
    printf("number of %d: %d\n", i, hhistogram[i]);
    }
    printf("Found %d prime numbers and %d not prime numbers \n",hresults[0],hresults[1]);

    //delete streams
    for(i=0;i<streamCnt;i++) {
      if (cudaSuccess!=cudaStreamDestroy(*(streams+i)))
         errorexit("Error creating stream");
    }

    
    //free memory
    free(hrandoms);
     if (cudaSuccess!=cudaFreeHost(hresults))
      errorexit("Error when deallocating space on the GPU");
     if (cudaSuccess!=cudaFreeHost(hhistogram))
      errorexit("Error when deallocating space on the GPU");
     if (cudaSuccess!=cudaFree(states))
      errorexit("Error when deallocating space on the GPU");
    if (cudaSuccess!=cudaFree(dresults))
      errorexit("Error when deallocating space on the GPU");
    if (cudaSuccess!=cudaFree(drandom))
      errorexit("Error when deallocating space on the GPU");
    if (cudaSuccess!=cudaFree(dhistogram))
      errorexit("Error when deallocating space on the GPU");

}
