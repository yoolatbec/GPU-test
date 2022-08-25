//====================================================================================================100
//		UPDATE
//====================================================================================================100

//    2006.03   Rob Janiczek
//        --creation of prototype version
//    2006.03   Drew Gilliam
//        --rewriting of prototype version into current version
//        --got rid of multiple function calls, all code in a  
//         single function (for speed)
//        --code cleanup & commenting
//        --code optimization efforts   
//    2006.04   Drew Gilliam
//        --added diffusion coefficent saturation on [0,1]
//		2009.12 Lukasz G. Szafaryn
//		-- reading from image, command line inputs
//		2010.01 Lukasz G. Szafaryn
//		--comments

//====================================================================================================100
//	DEFINE / INCLUDE
//====================================================================================================100

#define TEST_ROUND 9

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>

#include "define.c"
#include "extract_kernel.cu"
#include "prepare_kernel.cu"
#include "reduce_kernel.cu"
#include "srad_kernel.cu"
#include "srad2_kernel.cu"
#include "compress_kernel.cu"
#include "graphics.c"
#include "resize.c"

#include "device.c"				// (in library path specified to compiler)	needed by for device functions

//====================================================================================================100
//	MAIN FUNCTION
//====================================================================================================100

int main(int argc, char *argv []){

	//================================================================================80
	// 	VARIABLES
	//================================================================================80

    // inputs image, input paramenters
    fp* image_ori;																// originalinput image
	int image_ori_rows;
	int image_ori_cols;
	long image_ori_elem;

    // inputs image, input paramenters
    fp* image;															// input image
    int Nr,Nc;													// IMAGE nbr of rows/cols/elements
	long Ne;

	// algorithm parameters
    int niter;																// nbr of iterations
    fp lambda;															// update step size

    // size of IMAGE
	int r1,r2,c1,c2;												// row/col coordinates of uniform ROI
	long NeROI;														// ROI nbr of elements

    // surrounding pixel indicies
    int *iN,*iS,*jE,*jW;    

    // counters
    int iter;   // primary loop
    long i,j;    // image row/col

	// memory sizes
	int mem_size_i;
	int mem_size_j;
	int mem_size_single;

	//================================================================================80
	// 	GPU VARIABLES
	//================================================================================80

	// CUDA kernel execution parameters
	dim3 threads;
	int blocks_x;
	dim3 blocks;
	dim3 blocks2;
	dim3 blocks3;

	// memory sizes
	int mem_size;															// matrix memory size

	// HOST
	int no;
	int mul;
	fp total;
	fp total2;
	fp meanROI;
	fp meanROI2;
	fp varROI;
	fp q0sqr;

	// DEVICE
	fp* d_sums;															// partial sum
	fp* d_sums2;
	int* d_iN;
	int* d_iS;
	int* d_jE;
	int* d_jW;
	fp* d_dN; 
	fp* d_dS; 
	fp* d_dW; 
	fp* d_dE;
	fp* d_I;																// input IMAGE on DEVICE
	fp* d_c;

	//================================================================================80
	// 	GET INPUT PARAMETERS
	//================================================================================80

	if(argc != 5){
		printf("ERROR: wrong number of arguments\n");
		return 0;
	}
	else{
		niter = atoi(argv[1]);
		lambda = atof(argv[2]);
		Nr = atoi(argv[3]);						// it is 502 in the original image
		Nc = atoi(argv[4]);						// it is 458 in the original image
	}

	//================================================================================80
	// 	READ IMAGE (SIZE OF IMAGE HAS TO BE KNOWN)
	//================================================================================80

    // read image
	image_ori_rows = 502;
	image_ori_cols = 458;
	image_ori_elem = image_ori_rows * image_ori_cols;
    cudaEvent_t begin, end;
    float elapsed_time, min_time = -1;

    for(int r = 0; r < TEST_ROUND; r++){
        image_ori = (fp*)malloc(sizeof(fp) * image_ori_elem);

        read_graphics(	"image.pgm",
                                    image_ori,
                                    image_ori_rows,
                                    image_ori_cols,
                                    1);

        //================================================================================80
        // 	RESIZE IMAGE (ASSUMING COLUMN MAJOR STORAGE OF image_orig)
        //================================================================================80

        Ne = Nr*Nc;

        image = (fp*)malloc(sizeof(fp) * Ne);

        resize(	image_ori,
                    image_ori_rows,
                    image_ori_cols,
                    image,
                    Nr,
                    Nc,
                    1);

        //================================================================================80
        // 	SETUP
        //================================================================================80

        r1     = 0;											// top row index of ROI
        r2     = Nr - 1;									// bottom row index of ROI
        c1     = 0;											// left column index of ROI
        c2     = Nc - 1;									// right column index of ROI

        // ROI image size
        NeROI = (r2-r1+1)*(c2-c1+1);											// number of elements in ROI, ROI size

        // allocate variables for surrounding pixels
        mem_size_i = sizeof(int) * Nr;											//
        iN = (int *)malloc(mem_size_i) ;										// north surrounding element
        iS = (int *)malloc(mem_size_i) ;										// south surrounding element
        mem_size_j = sizeof(int) * Nc;											//
        jW = (int *)malloc(mem_size_j) ;										// west surrounding element
        jE = (int *)malloc(mem_size_j) ;										// east surrounding element

        // N/S/W/E indices of surrounding pixels (every element of IMAGE)
        for (i=0; i<Nr; i++) {
            iN[i] = i-1;														// holds index of IMAGE row above
            iS[i] = i+1;														// holds index of IMAGE row below
        }
        for (j=0; j<Nc; j++) {
            jW[j] = j-1;														// holds index of IMAGE column on the left
            jE[j] = j+1;														// holds index of IMAGE column on the right
        }

        // N/S/W/E boundary conditions, fix surrounding indices outside boundary of image
        iN[0]    = 0;															// changes IMAGE top row index from -1 to 0
        iS[Nr-1] = Nr-1;														// changes IMAGE bottom row index from Nr to Nr-1 
        jW[0]    = 0;															// changes IMAGE leftmost column index from -1 to 0
        jE[Nc-1] = Nc-1;														// changes IMAGE rightmost column index from Nc to Nc-1

        //================================================================================80
        // 	GPU SETUP
        //================================================================================80

        // allocate memory for entire IMAGE on DEVICE
        mem_size = sizeof(fp) * Ne;																		// get the size of float representation of input IMAGE
        cudaMalloc((void **)&d_I, mem_size);														//

        // allocate memory for coordinates on DEVICE
        cudaMalloc((void **)&d_iN, mem_size_i);													//
        cudaMemcpy(d_iN, iN, mem_size_i, cudaMemcpyHostToDevice);				//
        cudaMalloc((void **)&d_iS, mem_size_i);													// 
        cudaMemcpy(d_iS, iS, mem_size_i, cudaMemcpyHostToDevice);				//
        cudaMalloc((void **)&d_jE, mem_size_j);													//
        cudaMemcpy(d_jE, jE, mem_size_j, cudaMemcpyHostToDevice);				//
        cudaMalloc((void **)&d_jW, mem_size_j);													// 
        cudaMemcpy(d_jW, jW, mem_size_j, cudaMemcpyHostToDevice);			//

        // allocate memory for partial sums on DEVICE
        cudaMalloc((void **)&d_sums, mem_size);													//
        cudaMalloc((void **)&d_sums2, mem_size);												//

        // allocate memory for derivatives
        cudaMalloc((void **)&d_dN, mem_size);														// 
        cudaMalloc((void **)&d_dS, mem_size);														// 
        cudaMalloc((void **)&d_dW, mem_size);													// 
        cudaMalloc((void **)&d_dE, mem_size);														// 

        // allocate memory for coefficient on DEVICE
        cudaMalloc((void **)&d_c, mem_size);														// 

        checkCUDAError("setup");

        cudaEventCreate(&begin);
        cudaEventCreate(&end);

        //================================================================================80
        // 	KERNEL EXECUTION PARAMETERS
        //================================================================================80

        // all kernels operating on entire matrix
        threads.x = NUMBER_THREADS;												// define the number of threads in the block
        threads.y = 1;
        blocks_x = Ne/threads.x;
        if (Ne % threads.x != 0){												// compensate for division remainder above by adding one grid
            blocks_x = blocks_x + 1;																	
        }
        blocks.x = blocks_x;													// define the number of blocks in the grid
        blocks.y = 1;

        //================================================================================80
        // 	COPY INPUT TO CPU
        //================================================================================80

        cudaMemcpy(d_I, image, mem_size, cudaMemcpyHostToDevice);

        //================================================================================80
        // 	SCALE IMAGE DOWN FROM 0-255 TO 0-1 AND EXTRACT
        //================================================================================80

        cudaEventRecord(begin);

        extract<<<blocks, threads>>>(	Ne,
                                        d_I);

        checkCUDAError("extract");

        //================================================================================80
        // 	COMPUTATION
        //================================================================================80

        // printf("iterations: ");

        // execute main loop
        for (iter=0; iter<niter; iter++){										// do for the number of iterations input parameter

        // printf("%d ", iter);
        // fflush(NULL);

    #ifdef SEPERATE
            prepare_seperate<<<blocks, threads>>>(	Ne,
                                            d_I,
                                            d_sums,
                                            d_sums2);
    #else
            // execute square kernel
            prepare<<<blocks, threads>>>(	Ne,
                                            d_I,
                                            d_sums,
                                            d_sums2);
    #endif
            checkCUDAError("prepare");

            // performs subsequent reductions of sums
            blocks2.x = blocks.x;												// original number of blocks
            blocks2.y = blocks.y;												
            no = Ne;														// original number of sum elements
            mul = 1;														// original multiplier

            while(blocks2.x != 0){

                checkCUDAError("before reduce");

    #ifdef SEPERATE
            reduce_seperate<<<blocks2, threads>>>(	Ne,
                                                no,
                                                mul,
                                                d_sums, 
                                                d_sums2);
    #else
                // run kernel
                reduce<<<blocks2, threads>>>(	Ne,
                                                no,
                                                mul,
                                                d_sums, 
                                                d_sums2);
    #endif

                checkCUDAError("reduce");

                // update execution parameters
                no = blocks2.x;												// get current number of elements
                if(blocks2.x == 1){
                    blocks2.x = 0;
                }
                else{
                    mul = mul * NUMBER_THREADS;									// update the increment
                    blocks_x = blocks2.x/threads.x;								// number of blocks
                    if (blocks2.x % threads.x != 0){							// compensate for division remainder above by adding one grid
                        blocks_x = blocks_x + 1;
                    }
                    blocks2.x = blocks_x;
                    blocks2.y = 1;
                }

                checkCUDAError("after reduce");

            }

            checkCUDAError("before copy sum");

            // copy total sums to device
            // mem_size_single = sizeof(fp) * 1;
            // cudaMemcpy(&total, d_sums, mem_size_single, cudaMemcpyDeviceToHost);
            // cudaMemcpy(&total2, d_sums2, mem_size_single, cudaMemcpyDeviceToHost);

            // checkCUDAError("copy sum");

            // // calculate statistics
            // meanROI	= total / fp(NeROI);										// gets mean (average) value of element in ROI
            // meanROI2 = meanROI * meanROI;										//
            // varROI = (total2 / fp(NeROI)) - meanROI2;						// gets variance of ROI								
            // q0sqr = varROI / meanROI2;											// gets standard deviation of ROI

            // execute srad kernel

    #ifdef SEPERATE

            srad_seperate<<<blocks, threads>>>(	lambda,									// SRAD coefficient 
                                        Nr,										// # of rows in input image
                                        Nc,										// # of columns in input image
                                        Ne,										// # of elements in input image
                                        d_iN,									// indices of North surrounding pixels
                                        d_iS,									// indices of South surrounding pixels
                                        d_jE,									// indices of East surrounding pixels
                                        d_jW,									// indices of West surrounding pixels
                                        d_dN,									// North derivative
                                        d_dS,									// South derivative
                                        d_dW,									// West derivative
                                        d_dE,									// East derivative
                                        q0sqr,									// standard deviation of ROI 
                                        d_c,									// diffusion coefficient
                                        d_I);

    #else

            srad<<<blocks, threads>>>(	lambda,									// SRAD coefficient 
                                        Nr,										// # of rows in input image
                                        Nc,										// # of columns in input image
                                        Ne,										// # of elements in input image
                                        d_iN,									// indices of North surrounding pixels
                                        d_iS,									// indices of South surrounding pixels
                                        d_jE,									// indices of East surrounding pixels
                                        d_jW,									// indices of West surrounding pixels
                                        d_dN,									// North derivative
                                        d_dS,									// South derivative
                                        d_dW,									// West derivative
                                        d_dE,									// East derivative
                                        q0sqr,									// standard deviation of ROI 
                                        d_c,									// diffusion coefficient
                                        d_I);									// output image

    #endif

            checkCUDAError("srad");

            // execute srad2 kernel
            srad2<<<blocks, threads>>>(	lambda,									// SRAD coefficient 
                                        Nr,										// # of rows in input image
                                        Nc,										// # of columns in input image
                                        Ne,										// # of elements in input image
                                        d_iN,									// indices of North surrounding pixels
                                        d_iS,									// indices of South surrounding pixels
                                        d_jE,									// indices of East surrounding pixels
                                        d_jW,									// indices of West surrounding pixels
                                        d_dN,									// North derivative
                                        d_dS,									// South derivative
                                        d_dW,									// West derivative
                                        d_dE,									// East derivative
                                        d_c,									// diffusion coefficient
                                        d_I);									// output image

            checkCUDAError("srad2");

        }

        cudaEventRecord(end);
        cudaEventSynchronize(begin);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, begin, end);

    #ifdef SEPERATE
        printf("Seperate version. ");
    #endif

        if(min_time < 0 || elapsed_time < min_time){
            min_time = elapsed_time;
        }

        printf("elapsed time: %f ms, min time: %f ms\n", elapsed_time, min_time);

        cudaEventDestroy(begin);
        cudaEventDestroy(end);


        //================================================================================80
        // 	SCALE IMAGE UP FROM 0-1 TO 0-255 AND COMPRESS
        //================================================================================80

        compress<<<blocks, threads>>>(	Ne,
                                        d_I);

        checkCUDAError("compress");
        //================================================================================80
        // 	COPY RESULTS BACK TO CPU
        //================================================================================80

        cudaMemcpy(image, d_I, mem_size, cudaMemcpyDeviceToHost);

        checkCUDAError("copy back");

        //================================================================================80
        // 	WRITE IMAGE AFTER PROCESSING
        //================================================================================80

        // write_graphics(	"image_out.pgm",
        //                 image,
        //                 Nr,
        //                 Nc,
        //                 1,
        //                 255);

        //================================================================================80
        //	DEALLOCATE
        //================================================================================80

        free(image_ori);
        free(image);
        free(iN); 
        free(iS); 
        free(jW); 
        free(jE);

        cudaFree(d_I);
        cudaFree(d_c);
        cudaFree(d_iN);
        cudaFree(d_iS);
        cudaFree(d_jE);
        cudaFree(d_jW);
        cudaFree(d_dN);
        cudaFree(d_dS);
        cudaFree(d_dE);
        cudaFree(d_dW);
        cudaFree(d_sums);
        cudaFree(d_sums2);

    }


	//================================================================================80
	//	DISPLAY TIMING
	//================================================================================80

}

//====================================================================================================100
//	END OF FILE
//====================================================================================================100
