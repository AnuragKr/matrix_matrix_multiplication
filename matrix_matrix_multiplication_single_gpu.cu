#include<stdio.h>
#include <cuda.h>
#include <sys/time.h>

__global__ void matMulGpu(int *a,int *b,int *c,int matrixsize){
        int row = blockIdx.x;
        int col = threadIdx.x;

        for(int k=0;k<blockDim.x;k++)
                c[row*matrixsize+col] += a[row*matrixsize+k]*b[k*matrixsize+col];
}

//Utility function for time calculation
double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d", stat);
  return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printTime(const char *str, double starttime, double endtime) {
	printf("%s%3f seconds\n", str, endtime - starttime);
}

#define N 1024
#define BLOCKSIZE N
int main(){
    //Declaring Variables for Host
    int *a,*b,*c;

    //Declaring Variables for Device
    int *d_a,*d_b,*d_c,*d_d;

    //Allocated memory for variables with array of size N*N  on the Host
    a=(int *)malloc(sizeof(int)*N*N);
    b=(int *)malloc(sizeof(int)*N*N);
    c=(int *)malloc(sizeof(int)*N*N);

    //Allocated memory for variables with array of size N*N on the Device
    if(cudaMalloc((void **)&d_a,sizeof(int)*N*N)!=cudaSuccess) printf("Error in allocation d_a \n");
    if(cudaMalloc((void **)&d_b,sizeof(int)*N*N)!=cudaSuccess) printf("Error in allocation d_b \n");
    if(cudaMalloc((void **)&d_c,sizeof(int)*N*N)!=cudaSuccess) printf("Error in allocation d_c \n");
    if(cudaMalloc((void **)&d_d,sizeof(int)*N*N)!=cudaSuccess) printf("Error in allocation d_c \n");


    //Initialized Host variables
    for(int i=0;i<N*N;i++)a[i]=b[i]=1;

    //Copying Host variables content to Device
    if(cudaMemcpy(d_a,a,N*N*sizeof(int),cudaMemcpyHostToDevice) != cudaSuccess)printf("memcpy a->d_a failed \n");
    if(cudaMemcpy(d_b,b,N*N*sizeof(int),cudaMemcpyHostToDevice) != cudaSuccess)printf("memcpy b->d_b failed \n");

    //Launch the kernel for computation on the device
    int nblocks = ceil((int)N / BLOCKSIZE);
    double starttime,endtime;

    //GPU-2
    starttime = rtclock();
    matMulGpu<<<nblocks * N, BLOCKSIZE>>>(d_a,d_b,d_d,N);
    //Waiting For GPU Device to Finish computation
    cudaDeviceSynchronize();
    endtime = rtclock();
    printTime("GPU time: ", starttime, endtime);
    printf("\n");

    //Copying Result Synchronously from GPU to CPU
    if(cudaMemcpy(c,d_d,sizeof(int)*N*N,cudaMemcpyDeviceToHost)!=cudaSuccess)printf("\nCopy result failed\n");

    //Checking result
    for(int i=0;i<N*N;i++){ 
	    if(c[i]!=N){
		    printf("Error in result\n");
		    break;
	    }
    }

    //Free up the memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);

    return 0;
}
