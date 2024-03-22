#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <time.h>
#include <iostream>

#include <sys/time.h>
#define USECPSEC 1000000ULL


//initializes the matrix
void init_matrix(int *d, int N)
{
    for (int i = 0; i < N; i++)
    {
        d[i] = (rand() % 10);
    }
}


//optional Add matrix instead of multiply
__global__ void addMatrix(int *a, int *b, int *c)
{

    int global_index = threadIdx.x + blockDim.x * threadIdx.y;
    c[global_index] = a[global_index] + b[global_index];
}

//GPU matrix multiplication
__global__ void multMatrix(int *a,int *b,int *c,int width){
    int row= threadIdx.y + blockDim.y * blockIdx.y;
    int col= threadIdx.x + blockDim.x * blockIdx.x;
     
    
    if(row<width && col<width){
        for(int i=0; i<width;i++){
        c[row*width+col]+= a[row*width+i] * b[i*width+col];
        }       
    }
 
}

unsigned long long dtime_usec(unsigned long long start=0){

  timeval tv;
  gettimeofday(&tv, 0);
  return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
  //return ((tv.tv_sec)+tv.tv_usec)-start;
  //return ((tv.tv_sec*USECPSEC))-start;
}

__global__ void kwait(unsigned long long duration){
    unsigned long long start=clock64();
    while(clock64()< start + duration);
}


//The benchmark for testing reasons
void testing(int N, int kernelLaunches, FILE *fptr){
    for (int i = 5; i <= N; i += 5)
    {
        //printf("Seg fault is here 1");
        clock_t start, end;
        double duration;
        srand(time(NULL));
        size_t bytes = i * i * sizeof(int);
        int *a, *b, *c;


        int rc=cudaMallocManaged(&a, bytes);
        int rc2=cudaMallocManaged(&b, bytes);
        int rc3=cudaMallocManaged(&c,bytes);

        
      

        //Creates 16 threads per block and makes sure that there are 
        int threads =16;
        int blocks = (i + threads - 1) / threads;
        
        //in a 2D configuration to match the i*i space that we allocated previously
        dim3 THREADS(threads, threads);
        dim3 BLOCKS(blocks, blocks);


        //Initailizes the matrices with random numbers.
        init_matrix(a, i*i);
        init_matrix(b, i*i);

        unsigned long long my_duration = 1000000ULL;

        //timer for the benchmark and begins the benchmark
       //start = clock();
        unsigned long long dt =dtime_usec(0);
       // unsigned long long gpuDT=dt+clock64();


        for (int j = 0; j <= kernelLaunches; j++)
        {
            //printf("Seg fault is here 7");
            kwait<<<1,1>>>((my_duration*1000)*(i/5));
           // multMatrix<<<BLOCKS,THREADS>>>(a,b,c,i);
            cudaDeviceSynchronize();
            //printf("Cuda Return Code: %d", rc);
            //printf("A=%d |B=%d| C=%d   |",a[i*i-1],b[i*i-1],c[i*i-1]);
        }
        unsigned long long dtEND =dtime_usec(dt);
        //unsigned long long gpuEND=dtEND+clock64();
        //end = clock();
       
        unsigned long long expectedDuration=my_duration*(i/5);
        unsigned long long avgTimeTaken=dtEND/kernelLaunches;
        // divided by 1000 to get the number in NANOseconds
         //std::cout << "elapsed time: " << (dt/(float)USECPSEC)-(dtEND/(float)USECPSEC) << "s" << std::endl;
       // std::cout << expectedDuration << "|" << avgTimeTaken<< "|"<<expectedDuration-avgTimeTaken<<std::endl;
        std::cout << i <<"|"<<expectedDuration-avgTimeTaken<<std::endl;
        //ends the benchmark
        
        
        //Presents results
        //duration = dtEND/1000;

        /*
       // printf("Total Duration: %f \n", duration);
        printf("%f :", duration);
        
        double avgDuration = duration / kernelLaunches;
        
        //printf("Average time for each kernel:\n %f", avgDuration);
        printf(" %f :", avgDuration);

        //printf("Size %d\n", i);
        printf("%d\n", i);
        */

        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
        
    }
}

int main(int argc, char **argv)
{
    //printf("Seg fault is here 5");
    FILE *fptr = NULL ;

    
   
    int N = 50;
    int kernelLaunches; 
    bool both=false;


    for (int i = 1; i < argc; i++) //Escape Values
    {
        //printf("Seg fault is here");
        if (strcmp(argv[i], "-size") == 0 && i + 1 < argc)
        {
            N = atoi(argv[i + 1]);
        }

        else if (strcmp(argv[i], "-n") == 0 && i+ 1 < argc)
        {
            kernelLaunches = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-sync") == 0 && i + 1 < argc) {
            if (strcmp(argv[i + 1], "spin") == 0)
            {
                cudaSetDeviceFlags(cudaDeviceScheduleSpin);    
            }
            else if (strcmp(argv[i + 1], "block") == 0)
            {
                cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
            }
            else if(strcmp(argv[i + 1], "auto") == 0){
                cudaSetDeviceFlags(cudaDeviceScheduleAuto);
            }
            else if(strcmp(argv[i + 1], "yield") == 0){
                cudaSetDeviceFlags(cudaDeviceScheduleYield);
            }
            else if(strcmp(argv[i + 1], "oldBlock") == 0){
                cudaSetDeviceFlags(cudaDeviceBlockingSync);
            }
            else if (strcmp(argv[i + 1], "both") == 0)
            {
                both=true;
            }
            else
            {
                printf("\n INVALID SYNC");
            }
        }
    }

   // printf("I AM BREAKING HERE 177");
    
    
    //if you pciked a specific type of spin then just spin
    if(!both){
        //catches if the files doesnt exist
        
        testing(N,kernelLaunches,fptr);
       
    }
    //otherwise initialize and do both;
    else if(both){
        //printf("I AM BENCHMARKING BOTH");
        cudaSetDeviceFlags(cudaDeviceScheduleSpin);
        
        /*
        fptr=fopen("spinResults.txt", "w");
        //catches if the file doesnt exist
        
        if (fptr == NULL)
        {
            printf("Error opening file my g");
            exit(1);
        }
        */
        testing(N,kernelLaunches,fptr);
        


        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
        /*
        fptr=fopen("blockResults.txt", "w");
        
        //catches if the file doesnt exist
        if (fptr == NULL)
        {
            printf("Error opening file my g");
            exit(1);
        }
       */
        testing(N,kernelLaunches,fptr);

        //system("gnuplot -p blockResults.txt,spinResults.txt");
    }
}
