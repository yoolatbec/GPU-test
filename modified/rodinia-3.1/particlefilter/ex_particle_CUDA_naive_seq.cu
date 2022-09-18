/**
 * @file ex_particle_OPENMP_seq.c
 * @author Michael Trotter & Matt Goodrum
 * @brief Particle filter implementation in C/OpenMP 
 */
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <float.h>
#define PI 3.1415926535897932
#define BLOCK_X 16
#define BLOCK_Y 16

static float min_time = -1;

/**
@var M value for Linear Congruential Generator (LCG); use GCC's value
*/
long M = INT_MAX;
/**
@var A value for LCG
*/
int A = 1103515245;
/**
@var C value for LCG
*/
int C = 12345;

const int threads_per_block = 128;

void check_error(cudaError e) {
  if (e != cudaSuccess) {
    printf("\nCUDA error: %s\n", cudaGetErrorString(e));
    exit(1);
  }
}
__device__ int findIndexSeq(double * CDF, int lengthCDF, double value)
{
  int index = -1;
  int x;
  for(x = 0; x < lengthCDF; x++)
  {
    if(CDF[x] >= value)
    {
      index = x;
      break;
    }
  }
  if(index == -1)
    return lengthCDF-1;
  return index;
}
__device__ int findIndexBin(double * CDF, int beginIndex, int endIndex, double value)
{
  if(endIndex < beginIndex)
    return -1;
  int middleIndex;
  while(endIndex > beginIndex)
  {
    middleIndex = beginIndex + ((endIndex-beginIndex)/2);
    if(CDF[middleIndex] >= value)
    {
      if(middleIndex == 0)
        return middleIndex;
      else if(CDF[middleIndex-1] < value)
        return middleIndex;
      else if(CDF[middleIndex-1] == value)
      {
        while(CDF[middleIndex] == value && middleIndex >= 0)
          middleIndex--;
        middleIndex++;
        return middleIndex;
      }
    }
    if(CDF[middleIndex] > value)
      endIndex = middleIndex-1;
    else
      beginIndex = middleIndex+1;
  }
  return -1;
}
/*****************************
* CUDA Kernel Function to replace FindIndex
* param1: arrayX
* param2: arrayY
* param3: CDF
* param4: u
* param5: xj
* param6: yj
* param7: Nparticles
*****************************/
//seperate
__global__ void kernel_seperate(double * arrayX, double * arrayY, double * CDF, double * u, double * xj, double * yj, int Nparticles){
  int block_id = blockIdx.x;// + gridDim.x * blockIdx.y;
  int i = blockDim.x * block_id + threadIdx.x;

  if(i < Nparticles){
    int index = -1;
    int x;

    for(x = 0; x < Nparticles; x++){
      if(CDF[x] >= u[i]){
        index = x;
        break;
      }
    }
    if(index == -1){
      index = Nparticles-1;
    }

    xj[i] = arrayX[index];
    yj[i] = arrayY[index];
  }
}

//modified
__global__ void kernel(double * arrayX, double * arrayY, double * CDF, double * u, double * xj, double * yj, int Nparticles){
  int block_id = blockIdx.x;// + gridDim.x * blockIdx.y;
  int i = blockDim.x * block_id + threadIdx.x;

  if(i < Nparticles){
    int index = -1;
    int x;

    for(x = 0; x < Nparticles; x++){
      if(CDF[x] >= u[i]){
        index = x;
        break;
      }
    }
    if(index == -1){
      index = Nparticles-1;
    }

    double tx = arrayX[index];
    double ty = arrayY[index];
    xj[i] = tx;
    yj[i] = ty;
  }
}
/** 
 * Takes in a double and returns an integer that approximates to that double
 * @return if the mantissa < .5 => return value < input value; else return value > input value
 */
double roundDouble(double value){
  int newValue = (int)(value);
  if(value - newValue < .5)
    return newValue;
  else
    return newValue++;
}
/**
* Set values of the 3D array to a newValue if that value is equal to the testValue
* @param testValue The value to be replaced
* @param newValue The value to replace testValue with
* @param array3D The image vector
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
*/
void setIf(int testValue, int newValue, int * array3D, int * dimX, int * dimY, int * dimZ){
  int x, y, z;
  for(x = 0; x < *dimX; x++){
    for(y = 0; y < *dimY; y++){
      for(z = 0; z < *dimZ; z++){
        if(array3D[x * *dimY * *dimZ+y * *dimZ + z] == testValue)
          array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
      }
    }
  }
}
/**
* Generates a uniformly distributed random number using the provided seed and GCC's settings for the Linear Congruential Generator (LCG)
* @see http://en.wikipedia.org/wiki/Linear_congruential_generator
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a uniformly distributed number [0, 1)
*/
double randu(int * seed, int index)
{
  int num = A*seed[index] + C;
  seed[index] = num % M;
  return fabs(seed[index]/((double) M));
}
/**
* Generates a normally distributed random number using the Box-Muller transformation
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a double representing random number generated using the Box-Muller algorithm
* @see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
*/
double randn(int * seed, int index){
  /*Box-Muller algorithm*/
  double u = randu(seed, index);
  double v = randu(seed, index);
  double cosine = cos(2*PI*v);
  double rt = -2*log(u);
  return sqrt(rt)*cosine;
}
/**
* Sets values of 3D matrix using randomly generated numbers from a normal distribution
* @param array3D The video to be modified
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
* @param seed The seed array
*/
void addNoise(int * array3D, int * dimX, int * dimY, int * dimZ, int * seed){
  int x, y, z;
  for(x = 0; x < *dimX; x++){
    for(y = 0; y < *dimY; y++){
      for(z = 0; z < *dimZ; z++){
        array3D[x * *dimY * *dimZ + y * *dimZ + z] = array3D[x * *dimY * *dimZ + y * *dimZ + z] + (int)(5*randn(seed, 0));
      }
    }
  }
}
/**
* Fills a radius x radius matrix representing the disk
* @param disk The pointer to the disk to be made
* @param radius  The radius of the disk to be made
*/
void strelDisk(int * disk, int radius)
{
  int diameter = radius*2 - 1;
  int x, y;
  for(x = 0; x < diameter; x++){
    for(y = 0; y < diameter; y++){
      double distance = sqrt(pow((double)(x-radius+1),2) + pow((double)(y-radius+1),2));
      if(distance < radius)
        disk[x*diameter + y] = 1;
    }
  }
}
/**
* Dilates the provided video
* @param matrix The video to be dilated
* @param posX The x location of the pixel to be dilated
* @param posY The y location of the pixel to be dilated
* @param poxZ The z location of the pixel to be dilated
* @param dimX The x dimension of the frame
* @param dimY The y dimension of the frame
* @param dimZ The number of frames
* @param error The error radius
*/
void dilate_matrix(int * matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error)
{
  int startX = posX - error;
  while(startX < 0)
    startX++;
  int startY = posY - error;
  while(startY < 0)
    startY++;
  int endX = posX + error;
  while(endX > dimX)
    endX--;
  int endY = posY + error;
  while(endY > dimY)
    endY--;
  int x,y;
  for(x = startX; x < endX; x++){
    for(y = startY; y < endY; y++){
      double distance = sqrt( pow((double)(x-posX),2) + pow((double)(y-posY),2) );
      if(distance < error)
        matrix[x*dimY*dimZ + y*dimZ + posZ] = 1;
    }
  }
}

/**
* Dilates the target matrix using the radius as a guide
* @param matrix The reference matrix
* @param dimX The x dimension of the video
* @param dimY The y dimension of the video
* @param dimZ The z dimension of the video
* @param error The error radius to be dilated
* @param newMatrix The target matrix
*/
void imdilate_disk(int * matrix, int dimX, int dimY, int dimZ, int error, int * newMatrix)
{
  int x, y, z;
  for(z = 0; z < dimZ; z++){
    for(x = 0; x < dimX; x++){
      for(y = 0; y < dimY; y++){
        if(matrix[x*dimY*dimZ + y*dimZ + z] == 1){
          dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
        }
      }
    }
  }
}
/**
* Fills a 2D array describing the offsets of the disk object
* @param se The disk object
* @param numOnes The number of ones in the disk
* @param neighbors The array that will contain the offsets
* @param radius The radius used for dilation
*/
void getneighbors(int * se, int numOnes, double * neighbors, int radius){
  int x, y;
  int neighY = 0;
  int center = radius - 1;
  int diameter = radius*2 -1;
  for(x = 0; x < diameter; x++){
    for(y = 0; y < diameter; y++){
      if(se[x*diameter + y]){
        neighbors[neighY*2] = (int)(y - center);
        neighbors[neighY*2 + 1] = (int)(x - center);
        neighY++;
      }
    }
  }
}
/**
* The synthetic video sequence we will work with here is composed of a
* single moving object, circular in shape (fixed radius)
* The motion here is a linear motion
* the foreground intensity and the backgrounf intensity is known
* the image is corrupted with zero mean Gaussian noise
* @param I The video itself
* @param IszX The x dimension of the video
* @param IszY The y dimension of the video
* @param Nfr The number of frames of the video
* @param seed The seed array used for number generation
*/
void videoSequence(int * I, int IszX, int IszY, int Nfr, int * seed){
  int k;
  int max_size = IszX*IszY*Nfr;
  /*get object centers*/
  int x0 = (int)roundDouble(IszY/2.0);
  int y0 = (int)roundDouble(IszX/2.0);
  I[x0 *IszY *Nfr + y0 * Nfr  + 0] = 1;

  /*move point*/
  int xk, yk, pos;
  for(k = 1; k < Nfr; k++){
    xk = abs(x0 + (k-1));
    yk = abs(y0 - 2*(k-1));
    pos = yk * IszY * Nfr + xk *Nfr + k;
    if(pos >= max_size)
      pos = 0;
    I[pos] = 1;
  }

  /*dilate matrix*/
  int * newMatrix = (int *)calloc(IszX*IszY*Nfr, sizeof(int));
  imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
  int x, y;
  for(x = 0; x < IszX; x++){
    for(y = 0; y < IszY; y++){
      for(k = 0; k < Nfr; k++){
        I[x*IszY*Nfr + y*Nfr + k] = newMatrix[x*IszY*Nfr + y*Nfr + k];
      }
    }
  }
  free(newMatrix);

  /*define background, add noise*/
  setIf(0, 100, I, &IszX, &IszY, &Nfr);
  setIf(1, 228, I, &IszX, &IszY, &Nfr);
  /*add noise*/
  addNoise(I, &IszX, &IszY, &Nfr, seed);
}
/**
* Determines the likelihood sum based on the formula: SUM( (IK[IND] - 100)^2 - (IK[IND] - 228)^2)/ 100
* @param I The 3D matrix
* @param ind The current ind array
* @param numOnes The length of ind array
* @return A double representing the sum
*/
double calcLikelihoodSum(int * I, int * ind, int numOnes){
  double likelihoodSum = 0.0;
  int y;
  for(y = 0; y < numOnes; y++)
    likelihoodSum += (pow((double)(I[ind[y]] - 100),2) - pow((double)(I[ind[y]]-228),2))/50.0;
  return likelihoodSum;
}
/**
 * Finds the first element in the CDF that is greater than or equal to the provided value and returns that index
 * @note This function uses sequential search
 * @param CDF The CDF
 * @param lengthCDF The length of CDF
 * @param value The value to be found
 * @return The index of value in the CDF; if value is never found, returns the last index
 */
int findIndex(double * CDF, int lengthCDF, double value){
  int index = -1;
  int x;
  for(x = 0; x < lengthCDF; x++){
    if(CDF[x] >= value){
      index = x;
      break;
    }
  }
  if(index == -1){
    return lengthCDF-1;
  }
  return index;
}
/**
* The implementation of the particle filter using OpenMP for many frames
* @see http://openmp.org/wp/
* @note This function is designed to work with a video of several frames. In addition, it references a provided MATLAB function which takes the video, the objxy matrix and the x and y arrays as arguments and returns the likelihoods
* @param I The video to be run
* @param IszX The x dimension of the video
* @param IszY The y dimension of the video
* @param Nfr The number of frames
* @param seed The seed array used for random number generation
* @param Nparticles The number of particles to be used
*/
void particleFilter(int * I, int IszX, int IszY, int Nfr, int * seed, int Nparticles){
  int max_size = IszX*IszY*Nfr;
  //original particle centroid
  double xe = roundDouble(IszY/2.0);
  double ye = roundDouble(IszX/2.0);
        
  //expected object locations, compared to center
  int radius = 5;
  int diameter = radius*2 - 1;
  int * disk = (int *)calloc(diameter*diameter, sizeof(int));
  strelDisk(disk, radius);
  int countOnes = 0;
  int x, y;
  for(x = 0; x < diameter; x++){
    for(y = 0; y < diameter; y++){
      if(disk[x*diameter + y] == 1)
        countOnes++;
    }
  }
  double * objxy = (double *)calloc(countOnes*2, sizeof(double));
  getneighbors(disk, countOnes, objxy, radius);

  //initial weights are all equal (1/Nparticles)
  double * weights = (double *)calloc(Nparticles, sizeof(double));
  for(x = 0; x < Nparticles; x++){
    weights[x] = 1/((double)(Nparticles));
  }
  //initial likelihood to 0.0
  double * likelihood = (double *)calloc(Nparticles, sizeof(double));
  double * arrayX = (double *)calloc(Nparticles, sizeof(double));
  double * arrayY = (double *)calloc(Nparticles, sizeof(double));
  double * xj = (double *)calloc(Nparticles, sizeof(double));
  double * yj = (double *)calloc(Nparticles, sizeof(double));
  double * CDF = (double *)calloc(Nparticles, sizeof(double));

  //GPU copies of arrays
  double * arrayX_GPU;
  double * arrayY_GPU;
  double * xj_GPU;
  double * yj_GPU;
  double * CDF_GPU;

  int * ind = (int *)calloc(countOnes, sizeof(int));
  double * u = (double *)calloc(Nparticles, sizeof(double));
  double * u_GPU;

  //CUDA memory allocation
  check_error(cudaMalloc((void **) &arrayX_GPU, sizeof(double)*Nparticles));
  check_error(cudaMalloc((void **) &arrayY_GPU, sizeof(double)*Nparticles));
  check_error(cudaMalloc((void **) &xj_GPU, sizeof(double)*Nparticles));
  check_error(cudaMalloc((void **) &yj_GPU, sizeof(double)*Nparticles));
  check_error(cudaMalloc((void **) &CDF_GPU, sizeof(double)*Nparticles));
  check_error(cudaMalloc((void **) &u_GPU, sizeof(double)*Nparticles));

  for(x = 0; x < Nparticles; x++){
    arrayX[x] = xe;
    arrayY[x] = ye;
  }
  int k;
  //double * Ik = (double *)calloc(IszX*IszY, sizeof(double));
  int indX, indY;
  float elapsed_time_total = 0;
  for(k = 1; k < Nfr; k++){
    //printf("TIME TO SET ARRAYS TOOK: %f\n", elapsed_time(get_weights, set_arrays));
    //apply motion model
    //draws sample from motion model (random walk). The only prior information
    //is that the object moves 2x as fast as in the y direction

    for(x = 0; x < Nparticles; x++){
      arrayX[x] = arrayX[x] + 1.0 + 5.0*randn(seed, x);
      arrayY[x] = arrayY[x] - 2.0 + 2.0*randn(seed, x);
    }
    //particle filter likelihood
    for(x = 0; x < Nparticles; x++){

      //compute the likelihood: remember our assumption is that you know
      // foreground and the background image intensity distribution.
      // Notice that we consider here a likelihood ratio, instead of
      // p(z|x). It is possible in this case. why? a hometask for you.
      //calc ind
      for(y = 0; y < countOnes; y++){
        indX = roundDouble(arrayX[x]) + objxy[y*2 + 1];
        indY = roundDouble(arrayY[x]) + objxy[y*2];
        ind[y] = fabs(indX*IszY*Nfr + indY*Nfr + k);
        if(ind[y] >= max_size)
          ind[y] = 0;
      }
      likelihood[x] = calcLikelihoodSum(I, ind, countOnes);
      likelihood[x] = likelihood[x]/countOnes;
    }
    // update & normalize weights
    // using equation (63) of Arulampalam Tutorial
    for(x = 0; x < Nparticles; x++){
      weights[x] = weights[x] * exp(likelihood[x]);
    }
    double sumWeights = 0;
    for(x = 0; x < Nparticles; x++){
      sumWeights += weights[x];
    }
    for(x = 0; x < Nparticles; x++){
      weights[x] = weights[x]/sumWeights;
    }
    xe = 0;
    ye = 0;
    // estimate the object location by expected values
    for(x = 0; x < Nparticles; x++){
      xe += arrayX[x] * weights[x];
      ye += arrayY[x] * weights[x];
    }
    // printf("XE: %lf\n", xe);
    // printf("YE: %lf\n", ye);
    // double distance = sqrt( pow((double)(xe-(int)roundDouble(IszY/2.0)),2) + pow((double)(ye-(int)roundDouble(IszX/2.0)),2) );
    // printf("%lf\n", distance);
    //display(hold off for now)

    //pause(hold off for now)

    //resampling

    CDF[0] = weights[0];
    for(x = 1; x < Nparticles; x++){
      CDF[x] = weights[x] + CDF[x-1];
    }
    double u1 = (1/((double)(Nparticles)))*randu(seed, 0);
    for(x = 0; x < Nparticles; x++){
      u[x] = u1 + x/((double)(Nparticles));
    }
    //CUDA memory copying from CPU memory to GPU memory
    cudaMemcpy(arrayX_GPU, arrayX, sizeof(double)*Nparticles, cudaMemcpyHostToDevice);
    cudaMemcpy(arrayY_GPU, arrayY, sizeof(double)*Nparticles, cudaMemcpyHostToDevice);
    cudaMemcpy(xj_GPU, xj, sizeof(double)*Nparticles, cudaMemcpyHostToDevice);
    cudaMemcpy(yj_GPU, yj, sizeof(double)*Nparticles, cudaMemcpyHostToDevice);
    cudaMemcpy(CDF_GPU, CDF, sizeof(double)*Nparticles, cudaMemcpyHostToDevice);
    cudaMemcpy(u_GPU, u, sizeof(double)*Nparticles, cudaMemcpyHostToDevice);
    //Set number of threads
    int num_blocks = ceil((double) Nparticles/(double) threads_per_block);

    cudaEvent_t begin, end;
    float elapsed_time;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    cudaEventRecord(begin);

    //KERNEL FUNCTION CALL
    #ifdef SEPERATE
    kernel_seperate <<< num_blocks, threads_per_block >>> (arrayX_GPU, arrayY_GPU, CDF_GPU, u_GPU, xj_GPU, yj_GPU, Nparticles);
    #else
    kernel <<< num_blocks, threads_per_block >>> (arrayX_GPU, arrayY_GPU, CDF_GPU, u_GPU, xj_GPU, yj_GPU, Nparticles);
    #endif
    cudaEventRecord(end);
    cudaEventSynchronize(begin);
    cudaEventSynchronize(end);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time, begin, end);
    elapsed_time_total += elapsed_time;
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    //CUDA memory copying back from GPU to CPU memory
    cudaMemcpy(yj, yj_GPU, sizeof(double)*Nparticles, cudaMemcpyDeviceToHost);
    cudaMemcpy(xj, xj_GPU, sizeof(double)*Nparticles, cudaMemcpyDeviceToHost);

    for(x = 0; x < Nparticles; x++){
      //reassign arrayX and arrayY
      arrayX[x] = xj[x];
      arrayY[x] = yj[x];
      weights[x] = 1/((double)(Nparticles));
    }
  }

    if(min_time < 0 || elapsed_time_total < min_time){
        min_time = elapsed_time_total;
    }

#ifdef SEPERATE
    printf("Seperate version.");
#endif

    printf("elapsed time: %f ms, min time: %f ms\n", elapsed_time_total, min_time);

  //CUDA freeing of memory
  cudaFree(u_GPU);
  cudaFree(CDF_GPU);
  cudaFree(yj_GPU);
  cudaFree(xj_GPU);
  cudaFree(arrayY_GPU);
  cudaFree(arrayX_GPU);

  //free memory
  free(disk);
  free(objxy);
  free(weights);
  free(likelihood);
  free(arrayX);
  free(arrayY);
  free(xj);
  free(yj);
  free(CDF);
  free(u);
  free(ind);
}

#ifndef TEST_ROUND
#define TEST_ROUND 9
#endif

int main(int argc, char * argv[]){
  char* usage = "naive.out -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles>";
  //check number of arguments
  if(argc != 9)
  {
    printf("%s\n", usage);
    return 0;
  }
  //check args deliminators
  if( strcmp( argv[1], "-x" ) ||  strcmp( argv[3], "-y" ) || strcmp( argv[5], "-z" ) || strcmp( argv[7], "-np" ) ) {
    printf( "%s\n",usage );
    return 0;
  }

  int IszX, IszY, Nfr, Nparticles;

  //converting a string to a integer
  if( sscanf( argv[2], "%d", &IszX ) == EOF ) {
    printf("ERROR: dimX input is incorrect");
    return 0;
  }

  if( IszX <= 0 ) {
    printf("dimX must be > 0\n");
    return 0;
  }

  //converting a string to a integer
  if( sscanf( argv[4], "%d", &IszY ) == EOF ) {
    printf("ERROR: dimY input is incorrect");
    return 0;
  }

  if( IszY <= 0 ) {
    printf("dimY must be > 0\n");
    return 0;
  }

  //converting a string to a integer
  if( sscanf( argv[6], "%d", &Nfr ) == EOF ) {
    printf("ERROR: Number of frames input is incorrect");
    return 0;
  }

  if( Nfr <= 0 ) {
    printf("number of frames must be > 0\n");
    return 0;
  }

  //converting a string to a integer
  if( sscanf( argv[8], "%d", &Nparticles ) == EOF ) {
    printf("ERROR: Number of particles input is incorrect");
    return 0;
  }

  if( Nparticles <= 0 ) {
    printf("Number of particles must be > 0\n");
    return 0;
  }

    for(int r = 0; r < TEST_ROUND; r++){
        //establish seed
        int * seed = (int *)calloc(Nparticles, sizeof(int));
        int i;
        for(i = 0; i < Nparticles; i++)
            seed[i] = time(0)*i;
        //malloc matrix
        int * I = (int *)calloc(IszX*IszY*Nfr, sizeof(int));
        //call video sequence
        videoSequence(I, IszX, IszY, Nfr, seed);
        //call particle filter
        particleFilter(I, IszX, IszY, Nfr, seed, Nparticles);

        free(seed);
        free(I);
    }

  return 0;
}
