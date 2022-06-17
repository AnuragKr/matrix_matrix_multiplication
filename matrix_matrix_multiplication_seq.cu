#include<stdio.h>
#include <cuda.h>
#include <sys/time.h>

void matMulCpu(int *a,int *b,int *c, int matrixsize) {
        for (int i = 0; i < matrixsize; ++i) {
	        for (int j = 0; j < matrixsize; ++j) {
		        for (int k = 0; k < matrixsize; ++k) {
			        c[i * matrixsize + j] += a[i * matrixsize + k] * b[k * matrixsize + j];
		        }
	        }
	}
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

    //Allocated memory for variables with array of size N*N  on the Host
    a=(int *)malloc(sizeof(int)*N*N);
    b=(int *)malloc(sizeof(int)*N*N);
    c=(int *)malloc(sizeof(int)*N*N);

    //Initialized Host variables
    for(int i=0;i<N*N;i++)a[i]=b[i]=1;

    //CPU
    double starttime,endtime;
    starttime = rtclock();
    matMulCpu(a,b,c,N);
    endtime = rtclock();
    printTime("CPU time: ", starttime, endtime);

    return 0;
}
