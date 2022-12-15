/*
CUDA - generate array of random numbers and calculate occurence of odd and even numbers - no streams
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

    //memory allocation host
    int *hresults=(int*)malloc(2*sizeof(int));
    int *hrandoms=(int*)malloc(size*sizeof(int));
    int *hhistogram=(int*)malloc(MAX*sizeof(int));

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
      
    int *dhistogram=NULL;
    if (cudaSuccess!=cudaMalloc((void **)&dhistogram,MAX*sizeof(int)))
      errorexit("Error allocating memory on the GPU");
   
    //results memory initialize with 0
    if (cudaSuccess!=cudaMemset(dresults,0, 2*sizeof(int)))
      errorexit("Error initializing memory on the GPU");
    
    //hist memory initialize with 0
    if (cudaSuccess!=cudaMemset(dhistogram,0, MAX*sizeof(int)))
      errorexit("Error initializing memory on the GPU");

    //kernel for number generation
    generate<<<blocksingrid,threadsinblock>>>(time(0),states, drandom);
    if (cudaSuccess!=cudaGetLastError())
      errorexit("Error during kernel launch");
  
    //copy random numbers to host - i.e. for future to file export
    if (cudaSuccess!=cudaMemcpy(hrandoms,drandom,size*sizeof(int),cudaMemcpyDeviceToHost))
      errorexit("Error copying randoms");
     

    //calculate odd numbers
    checkPrime<<<blocksingrid,threadsinblock>>>(drandom, dresults);
    if (cudaSuccess!=cudaGetLastError())
      errorexit("Error during kernel launch");

    //calculate even numbers
    calculateOccurance<<<blocksingrid,threadsinblock>>>(drandom, dhistogram);
    
    
        //copy
    if (cudaSuccess!=cudaMemcpy(hhistogram,dhistogram,MAX*sizeof(int),cudaMemcpyDeviceToHost))
      errorexit("Error copying histogram");
      
      
    //copy results to host
    if (cudaSuccess!=cudaMemcpy(hresults,dresults,2*sizeof(int),cudaMemcpyDeviceToHost))
       errorexit("Error copying results");
    
    for(int i = 0; i < MAX; i++)
    {
    printf("number of %d: %d\n", i, hhistogram[i]);
    }
    printf("Found %d prime numbers and %d not prime numbers \n",hresults[0],hresults[1]);

    //free memory
    free(hresults);
    free(hrandoms);
    free(hhistogram);
     if (cudaSuccess!=cudaFree(states))
      errorexit("Error when deallocating space on the GPU");
    if (cudaSuccess!=cudaFree(dresults))
      errorexit("Error when deallocating space on the GPU");
    if (cudaSuccess!=cudaFree(drandom))
      errorexit("Error when deallocating space on the GPU");
    if (cudaSuccess!=cudaFree(dhistogram))
      errorexit("Error when deallocating space on the GPU");

}
