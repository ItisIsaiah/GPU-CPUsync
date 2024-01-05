#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <time.h>



//initializes the matrix
void init_matrix(int *d, int N)
{
    
    printf("Started initializing Matrix with size %d\n",N);
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

__global__ void kwait(unsigned long long duration){
    unsigned long long start=clock64();
    while(clock64()< start + duration);
}


//The benchmark for testing reasons
void testing(int N, int kernelLaunches, FILE *fptr){
    for (int i = 10; i <= N; i += 10)
    {

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
        
        
        //timer for the benchmark and begins the benchmark
        start = clock();

        for (int j = 0; j <= kernelLaunches; j++)
        {
            multMatrix<<<BLOCKS,THREADS>>>(a,b,c,i);
            cudaDeviceSynchronize();
        }
        end = clock();
        //ends the benchmark


        //Presents results
        duration = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Total Duration: %f \n", duration);
        fprintf(fptr, "%f :", duration);

        double avgDuration = duration / kernelLaunches;

        printf("Average time for each kernel:\n %f", avgDuration);
        fprintf(fptr, " %f :", avgDuration);

        printf("Size %d\n", i);
        fprintf(fptr, "%d\n", i);

       // printf("\nFirst 3 of a :%d |%d | %d\n",a[i*i-3],a[i*i-2],a[i*i-1]);
        //printf("\nFirst 3 of b :%d |%d | %d\n",b[i*i-3],b[i*i-2],b[i*i-1]);
       // printf("\nFirst 3 of c :%d |%d | %d\n \n",c[i*i-3],c[i*i-2],c[i*i-1]);  


        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
        
    }
}

int main(int argc, char **argv)
{
    
    FILE *fptr = NULL ;

    
   // fprintf(fptr, "Spin Method Duration Size ");
    int N = 50;
    int kernelLaunches; 
    bool both=false;


    for (int i = 1; i < argc; i++) //Escape Values
    {
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
                fptr=fopen("spinResults.txt", "w");
                //fprintf(fptr, "spin: \n");
            }
            else if (strcmp(argv[i + 1], "block") == 0)
            {
                cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
                fptr=fopen("blockResults.txt", "w");
                //fprintf(fptr, "block: \n");
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

    printf("I AM BREAKING HERE 177");
    
    
    //if you pciked a specific type of spin then just spin
    if(!both){
        //catches if the files doesnt exist
        if (fptr == NULL)
        {
            printf("Error opening file my g");
            exit(1);
        }
        testing(N,kernelLaunches,fptr);
       
    }
    //otherwise initialize and do both;
    else if(both){
        printf("I AM BENCHMARKING BOTH");
        cudaSetDeviceFlags(cudaDeviceScheduleSpin);
        fptr=fopen("spinResults.txt", "w");
        //catches if the file doesnt exist
        if (fptr == NULL)
        {
            printf("Error opening file my g");
            exit(1);
        }
        //fprintf(fptr, "spin: \n");
        testing(N,kernelLaunches,fptr);
        


        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
        fptr=fopen("blockResults.txt", "w");
        
        //catches if the file doesnt exist
        if (fptr == NULL)
        {
            printf("Error opening file my g");
            exit(1);
        }
       // fprintf(fptr, "block: \n");
        testing(N,kernelLaunches,fptr);

        system("gnuplot -p blockResults.txt,spinResults.txt");
    }
}
