#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <time.h>

void init_matrix(int *d, int N)
{
    
    printf("Started initializing Matrix with size %d\n",N);
    for (int i = 0; i < N; i++)
    {
        //printf("initializing: d[%d] \n",i);
        d[i] = (rand() % 10);
        //printf("d[%d] is initialized\n",i);
    }
}

/*
void fill_random(int array[r][c],int max){
    for(int i=0;i <r;i++){
        for(int j=0;j<c;j++){
            array[i][j] =(rand() %max)+1;
        }
    }
}
*/

__global__ void addMatrix(int *a, int *b, int *c)
{

    int global_index = threadIdx.x + blockDim.x * threadIdx.y;
    c[global_index] = a[global_index] + b[global_index];
}

__global__ void multMatrix(int *a,int *b,int *c,int width){
    int row= threadIdx.y + blockDim.y * blockIdx.y;
    int col= threadIdx.x + blockDim.x * blockIdx.x;
     
    //printf("\nrow:%d ||Col:%d\n",row,col);
    if(row<width && col<width){
        for(int i=0; i<width;i++){
        c[row*width+col]+= a[row*width+i] * b[i*width+col];
        }
        //printf("\nrow:%d ||Col:%d|| Cell: %d\n",row,col,c[row*width+col]);
    }
    

    
    
 
}

__global__ void kwait(unsigned long long duration){
    unsigned long long start=clock64();
    while(clock64()< start + duration);
}

int main(int argc, char **argv)
{
    FILE *fptr = fopen("results.txt", "w");
    if (fptr == NULL)
    {
        printf("Error opening file my g");
        exit(1);
    }
    fprintf(fptr, "Spin Method Duration Size ");

    int N = 50;
    int kernelLaunches; 
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
                fprintf(fptr, "spin: \n");
            }
            else if (strcmp(argv[i + 1], "block") == 0)
            {
                cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
                fprintf(fptr, "block: \n");
            }
            else
            {
                printf("\n INVALID SYNC");
            }
        }
    }

    for (int i = 10; i <= N; i += 10)
    {

        clock_t start, end;
        double duration;

     // for input later
        srand(time(NULL));
       // printf("Allocating space\n");
        size_t bytes = i * i * sizeof(int);
       
        int *a, *b, *c;


        int rc=cudaMallocManaged(&a, bytes);
        int rc2=cudaMallocManaged(&b, bytes);
        int rc3=cudaMallocManaged(&c,bytes);

        
       //printf("%d|%d|%d",rc,rc2,rc3);


        int threads =16;//was 16
        int blocks = (i + threads - 1) / threads;
        

        dim3 THREADS(threads, threads);
        dim3 BLOCKS(blocks, blocks);
        // printf("initializing matrix\n");
        init_matrix(a, i*i);
        init_matrix(b, i*i);
        
        const unsigned long long my_duration= 2000000000ULL; // FOR KWAIT
        
        // printf("GOing to run Kernels \n");

        start = clock();

        for (int j = 0; j <= kernelLaunches; j++)
        {
            multMatrix<<<BLOCKS,THREADS>>>(a,b,c,i);
            cudaDeviceSynchronize();
        }
        end = clock();
        

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