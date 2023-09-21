/**
 * DPC++ application for high-order exhaustive epistasis detection using K2 scoring.
 */

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>
#include <gsl/gsl_combination.h>
#include <omp.h>
#include <algorithm>
#include "combination.h"

#define MAX_LINE_SIZE 134217728	
#define MAX_NUM_LINES 1000000

#define NUM_COMB_PER_KERNEL_INSTANCE 131072
#define WORKGROUP_SIZE 32


unsigned long long * d_datasetCases;
unsigned long long * d_datasetControls;
uint * d_combinationArray;
float * d_lgammaPrecalc;
float * d_output;
int * d_output_index;


__inline__ float shfl_xor_32(float scalarValue, const int n,
		sycl::nd_item<3> item_ct1) {
	return item_ct1.get_sub_group().shuffle_xor(scalarValue, n);
}


__inline__ float warpReduceMin(float val, sycl::nd_item<3> item_ct1) {
	for(int i=1; i < item_ct1.get_sub_group().get_local_range().get(0); i = i * 2) {
		val = sycl::min(val, shfl_xor_32(val, i, item_ct1));
	}	
	return val;
}


__inline__ float blockReduceMin(float val, sycl::nd_item<3> item_ct1, float *shared) {

	int lane = item_ct1.get_local_id(2) %
		item_ct1.get_sub_group().get_local_range().get(0);
	int wid = item_ct1.get_local_id(2) /
		item_ct1.get_sub_group().get_local_range().get(0);

	val = warpReduceMin(val, item_ct1); 
	if (lane==0) shared[wid]=val; 

	item_ct1.barrier();

	val = (item_ct1.get_local_id(2) <
			(item_ct1.get_local_range().get(2) /
			 item_ct1.get_sub_group().get_local_range().get(0)))
		? shared[lane]
		: FLT_MAX;
	if (wid == 0) shared[0] =
		warpReduceMin(val, item_ct1);

	item_ct1.barrier();

	val = shared[0];
	return val;
}


/* Saves best cadidate score
 * Based on: https://streamhpc.com/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
 */
__inline__ void atomicMin_g_f(float *addr, float val) {
	
	union {
		unsigned int u32;
		float        f32;
	} next, expected, current;
	current.f32    = *addr;
	do {
		expected.f32 = current.f32;
		next.f32 = sycl::min(expected.f32, val);
		current.u32 = dpct::atomic_compare_exchange_strong((unsigned int *)addr, expected.u32, next.u32);
	} while( current.u32 != expected.u32 );
}


/* Saves indexes of set of SNPs with best cadidate score */
__inline__ void atomicGetIndex(float *addr, float val, int *index,
		sycl::nd_item<3> item_ct1) {

	int next, expected, current;
	current = *index;

	do {
		expected = current;
		float global_minVal = *addr;
		if(val <= global_minVal) {
			next = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);
			current = dpct::atomic_compare_exchange_strong(index, expected, next);
		}
	} while( current != expected );
}


/* Evaluates a set of SNPs from combinationArray */
void epistasis(unsigned long long *datasetCases, unsigned long long *datasetControls, uint * combinationArray, float *lgammaPrecalc, 
		float *output, int *output_index, int epistasisSize, int numSNPs, int casesSize, int controlsSize, ulong numCombinations, int comb,
		sycl::nd_item<3> item_ct1, uint8_t *dpct_local, float *shared) {

	int combination_i = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2); 

	int local_id = item_ct1.get_local_id(2); 
	int wid = item_ct1.get_local_id(2) / item_ct1.get_sub_group().get_local_range().get(0);

	int i, j;
	int sample_i;
	int cases_i, controls_i;
	uint index[16];
	uint dataset_i;

	float score = FLT_MAX;

	auto smem = (uint *)dpct_local;
	uint * observedValues_shared = smem;

	const int pow_table[10] = {1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683}; 

	for(i=0;i<(WORKGROUP_SIZE * COMB_SIZE);i=i+WORKGROUP_SIZE) {
		observedValues_shared[i + local_id] = 0;
		observedValues_shared[WORKGROUP_SIZE * COMB_SIZE + i + local_id] = 0;
	}
	item_ct1.barrier();

	if(combination_i < numCombinations) { 

		score = 0;

		uint combination_table[EPISTASIS_SIZE];
		for(i=0; i < EPISTASIS_SIZE; i++) {
			combination_table[i] = combinationArray[numCombinations * i + combination_i];
		}

		unsigned long long casesArr[3 * EPISTASIS_SIZE];		
		unsigned long long controlsArr[3 * EPISTASIS_SIZE];        	

		for(cases_i = 0; cases_i < casesSize; cases_i++) {

			casesArr[0] = datasetCases[0 * numSNPs * casesSize + cases_i * numSNPs + combination_table[0]];
			casesArr[1] = datasetCases[1 * numSNPs * casesSize + cases_i * numSNPs + combination_table[0]];
			casesArr[2] = datasetCases[2 * numSNPs * casesSize + cases_i * numSNPs + combination_table[0]];

			unsigned long long mask = (casesArr[0] | casesArr[1] | casesArr[2]);

			for(int epistasis_i=1; epistasis_i < EPISTASIS_SIZE; epistasis_i++) {

				casesArr[epistasis_i * 3 + 0] = datasetCases[0 * numSNPs * casesSize + cases_i * numSNPs + combination_table[epistasis_i]];
				casesArr[epistasis_i * 3 + 1] = datasetCases[1 * numSNPs * casesSize + cases_i * numSNPs + combination_table[epistasis_i]];
				casesArr[epistasis_i * 3 + 2] = mask & ~(casesArr[epistasis_i * 3 + 0] | casesArr[epistasis_i * 3 + 1]); 
			}

                        for(int a_i = 0; a_i < 3; a_i++) {
                                for(int b_i = 0; b_i < 3; b_i++) {
                                        #if defined(PAIRWISE)
                                        uint comb_i = a_i * 3 + b_i;
                                        observedValues_shared[comb_i * 2 * WORKGROUP_SIZE + 1 * WORKGROUP_SIZE + local_id] += sycl::popcount(casesArr[0 * 3 + a_i] & casesArr[1 * 3 + b_i]);
                                        #else
                                        for(int c_i = 0; c_i < 3; c_i++) {
                                                uint comb_i = a_i * 9 + b_i * 3 + c_i;
                                                observedValues_shared[comb_i * 2 * WORKGROUP_SIZE + 1 * WORKGROUP_SIZE + local_id] += sycl::popcount(casesArr[0 * 3 + a_i] & casesArr[1 * 3 + b_i] & casesArr[2 * 3 + c_i]);
                                        }
                                        #endif
                                }
                        }
		}

		for(controls_i = 0; controls_i < controlsSize; controls_i++) {

			controlsArr[0] = datasetControls[0 * numSNPs * controlsSize + controls_i * numSNPs + combination_table[0]];
			controlsArr[1] = datasetControls[1 * numSNPs * controlsSize + controls_i * numSNPs + combination_table[0]];
			controlsArr[2] = datasetControls[2 * numSNPs * controlsSize + controls_i * numSNPs + combination_table[0]];

			unsigned long long mask = (controlsArr[0] | controlsArr[1] | controlsArr[2]);

			for(int epistasis_i=1; epistasis_i < EPISTASIS_SIZE; epistasis_i++) {

				controlsArr[epistasis_i * 3 + 0] = datasetControls[0 * numSNPs * controlsSize + controls_i * numSNPs + combination_table[epistasis_i]];
				controlsArr[epistasis_i * 3 + 1] = datasetControls[1 * numSNPs * controlsSize + controls_i * numSNPs + combination_table[epistasis_i]];
				controlsArr[epistasis_i * 3 + 2] = mask & ~(controlsArr[epistasis_i * 3 + 0] | controlsArr[epistasis_i * 3 + 1]); 
			}

                        for(int a_i = 0; a_i < 3; a_i++) {
                                for(int b_i = 0; b_i < 3; b_i++) {
                                        #if defined(PAIRWISE)
                                        uint comb_i = a_i * 3 + b_i;
                                        observedValues_shared[comb_i * 2 * WORKGROUP_SIZE + 0 * WORKGROUP_SIZE + local_id] += sycl::popcount(controlsArr[0 * 3 + a_i] & controlsArr[1 * 3 + b_i]);
                                        #else
                                        for(int c_i = 0; c_i < 3; c_i++) {
                                                uint comb_i = a_i * 9 + b_i * 3 + c_i;
                                                observedValues_shared[comb_i * 2 * WORKGROUP_SIZE + 0 * WORKGROUP_SIZE + local_id] += sycl::popcount(controlsArr[0 * 3 + a_i] & controlsArr[1 * 3 + b_i] & controlsArr[2 * 3 + c_i]);
                                        }
                                        #endif
                                }
                        }
		}

		for (i=0; i< COMB_SIZE; i++) { 
			ushort zerosCount = observedValues_shared[i * 2 * WORKGROUP_SIZE + 0 * WORKGROUP_SIZE + local_id];
			ushort onesCount = observedValues_shared[i * 2 * WORKGROUP_SIZE + 1 * WORKGROUP_SIZE + local_id];

			score = score + lgammaPrecalc[zerosCount] + lgammaPrecalc[onesCount] -
				lgammaPrecalc[zerosCount + onesCount + 1];

		}
		score = sycl::fabs(score);
	}

	float min_score = blockReduceMin(score, item_ct1, shared); 

	if(local_id == 0) {
		atomicMin_g_f(output, min_score);
	}

	if(score == min_score) {    
		atomicGetIndex(output, min_score, output_index, item_ct1);
	}

}


void cuda_mem_init(int datasetCases_size, int datasetControls_size, int combinationArray_size, int lgammaPrecalc_size, int output_size) try {
	
	dpct::device_ext &dev_ct1 = dpct::get_current_device();
	sycl::queue &q_ct1 = dev_ct1.default_queue();

	d_datasetCases = sycl::malloc_device<unsigned long long>(datasetCases_size, q_ct1);
	d_datasetControls = sycl::malloc_device<unsigned long long>(datasetControls_size, q_ct1);
	d_combinationArray = sycl::malloc_device<uint>(combinationArray_size, q_ct1);
	d_lgammaPrecalc = sycl::malloc_device<float>(lgammaPrecalc_size, q_ct1);
	d_output = sycl::malloc_device<float>(output_size, q_ct1);
	d_output_index = sycl::malloc_device<int>(output_size, q_ct1);
}
catch (sycl::exception const &exc) {
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}


void cuda_mem_copy(unsigned long long *datasetCases, int datasetCases_size,
		unsigned long long *datasetControls,
		int datasetControls_size, uint *combinationArray,
		int combinationArray_size, float *lgammaPrecalc,
		int lgammaPrecalc_size, float *output, int output_size) try {
	
	dpct::device_ext &dev_ct1 = dpct::get_current_device();
	sycl::queue &q_ct1 = dev_ct1.default_queue();

	if(datasetCases != NULL)
		q_ct1.memcpy(d_datasetCases, datasetCases, datasetCases_size * sizeof(unsigned long long)).wait();

	if(datasetControls != NULL)
		q_ct1.memcpy(d_datasetControls, datasetControls, datasetControls_size * sizeof(unsigned long long)).wait();

	if(combinationArray != NULL)
		q_ct1.memcpy(d_combinationArray, combinationArray, combinationArray_size * sizeof(uint)).wait();

	if(lgammaPrecalc != NULL)
		q_ct1.memcpy(d_lgammaPrecalc, lgammaPrecalc, lgammaPrecalc_size * sizeof(float)).wait();

	if(output != NULL)
		q_ct1.memcpy(d_output, output, output_size * sizeof(float)).wait();

}
catch (sycl::exception const &exc) {
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__
		<< ", line:" << __LINE__ << std::endl;
	std::exit(1);
}


void cuda_launch_kernel(int numSNPs, int controlsSize, int casesSize, int numCombinationsPerKernelInstance, int CpuThreadID, sycl::queue *stream_id)
{
	int blocksPerGrid = (size_t)ceil(
			((float)numCombinationsPerKernelInstance) /
			((float)WORKGROUP_SIZE)); 

	int comb = (int)pow(3.0, EPISTASIS_SIZE);

	dpct::get_default_queue().submit([&](sycl::handler &cgh) {
			sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
			sycl::access::target::local>
			dpct_local_acc_ct1(
					sycl::range<1>(2 * WORKGROUP_SIZE * comb * sizeof(uint)), cgh);
			sycl::accessor<float, 1, sycl::access::mode::read_write,
			sycl::access::target::local>
			shared_acc_ct1(sycl::range<1>(WORKGROUP_SIZE), cgh);

			auto d_datasetCases_ct0 = d_datasetCases;
			auto d_datasetControls_ct1 = d_datasetControls;
			auto
			d_combinationArray_CpuThreadID_NUM_COMB_PER_KERNEL_INSTANCE_EPISTASIS_SIZE_ct2 =
			d_combinationArray +
			(CpuThreadID * NUM_COMB_PER_KERNEL_INSTANCE * EPISTASIS_SIZE);
			auto d_lgammaPrecalc_ct3 = d_lgammaPrecalc;
			auto d_output_CpuThreadID_ct4 = d_output + CpuThreadID;
			auto d_output_index_CpuThreadID_ct5 = d_output_index + CpuThreadID;
			auto EPISTASIS_SIZE_ct6 = EPISTASIS_SIZE;
			auto numSNPs_ct7 = numSNPs;
			auto ceil_casesSize_ct8 = (int)ceil(casesSize / 64.0f);
			auto ceil_controlsSize_ct9 = (int)ceil(controlsSize / 64.0f);

			cgh.parallel_for(
					sycl::nd_range<3>(sycl::range<3>(1, 1, blocksPerGrid) *
						sycl::range<3>(1, 1, WORKGROUP_SIZE),
						sycl::range<3>(1, 1, WORKGROUP_SIZE)),
					[=](sycl::nd_item<3> item_ct1) {
					epistasis(
							d_datasetCases_ct0, d_datasetControls_ct1,
							d_combinationArray_CpuThreadID_NUM_COMB_PER_KERNEL_INSTANCE_EPISTASIS_SIZE_ct2,
							d_lgammaPrecalc_ct3, d_output_CpuThreadID_ct4,
							d_output_index_CpuThreadID_ct5, EPISTASIS_SIZE_ct6, numSNPs_ct7,
							ceil_casesSize_ct8, ceil_controlsSize_ct9,
							numCombinationsPerKernelInstance, comb, item_ct1,
							dpct_local_acc_ct1.get_pointer(), shared_acc_ct1.get_pointer());
					});
	});

}


void cuda_clean_up() {

	dpct::device_ext &dev_ct1 = dpct::get_current_device();
	sycl::queue &q_ct1 = dev_ct1.default_queue();

	sycl::free(d_datasetCases, q_ct1);
	sycl::free(d_datasetControls, q_ct1);
	sycl::free(d_combinationArray, q_ct1);
	sycl::free(d_output, q_ct1);
	sycl::free(d_output_index, q_ct1);
}



/* Loads a sample from dataset in 0's, 1's and 2's format */
int getValues(char* line, u_char* data, u_char* data_target) {

        int num = 0;
        const char* tok;
        for (tok = strtok(line, ",");
                        tok && *tok;
                        tok = strtok(NULL, ",\n"))
        {
                if(data != NULL) {
                        data[num] = atoi(tok);
                }
                num++;
        }
        if(data_target != NULL) {
                data_target[0] = data[num - 1];
        }
        return num;
}



int main(int argc, char *argv[]) try {

	int numSNPs;
	int sampleSize;
	int casesSize;
	int controlsSize;
	unsigned long long numCombinations;

	dpct::device_ext &dev_ct1 = dpct::get_current_device();
	sycl::queue &q_ct1 = dev_ct1.default_queue();

	std::cout << "Device : " << q_ct1.get_device().get_info<sycl::info::device::name>() << std::endl;

	if(argc < 2) {
		printf("USE: infile\n");
		return 1;
	}

	struct timespec t_start, t_end;
	clock_gettime(CLOCK_MONOTONIC, &t_start);


	FILE* stream = fopen(argv[1], "r");	// file with dataset

	char * line = (char *) malloc(MAX_LINE_SIZE * sizeof(char));
	fgets(line, MAX_LINE_SIZE, stream);
	int numCols = getValues(line, NULL, NULL);

	u_char * dataset = (u_char*) malloc(sizeof(u_char) * MAX_NUM_LINES * numCols);
	u_char * dataset_target = (u_char*) malloc(sizeof(u_char) * MAX_NUM_LINES);

	if(dataset == NULL) {
		printf("\nMemory allocation for dataset (genotype) failed.\n");
	}
	if(dataset_target == NULL) {
		printf("\nMemory allocation for dataset (phenotype) failed.\n");
	}

	sampleSize = 0;
	while (fgets(line, MAX_LINE_SIZE, stream))
	{
		getValues(line, dataset + (numCols * sampleSize), dataset_target + sampleSize);
		sampleSize++;
	}


	/* Counts the number of controls (0s) and cases (1s) */
	controlsSize = 0;
	casesSize = 0;
	for(int i=0; i < sampleSize; i++) {
		if(dataset_target[i] == 1) {

			casesSize++;
		}
		else {
			controlsSize++;
		}
	}

        printf("sample size: %d\n#cases: %d, #controls:%d\n", sampleSize, casesSize, controlsSize);

	int datasetOnes_64packed_size =
		ceil(((float)casesSize) / 64.0f) * (numCols - 1) * 3;
	unsigned long long * datasetOnes_64packed = (unsigned long long*) calloc(datasetOnes_64packed_size, sizeof(unsigned long long));

	int datasetZeros_64packed_size =
		ceil(((float)controlsSize) / 64.0f) * (numCols - 1) * 3;
	unsigned long long * datasetZeros_64packed = (unsigned long long*) calloc(datasetZeros_64packed_size, sizeof(unsigned long long));

        if(datasetOnes_64packed == NULL) {
                printf("\nMemory allocation for internal representation (cases) failed.\n");
        }
        if(datasetOnes_64packed == NULL) {
                printf("\nMemory allocation for internal representation (controls) failed.\n");
        }


	int numSamplesOnes_64packed = (int)ceil(((float)casesSize) / 64.0f);
	int numSamplesZeros_64packed = (int)ceil(((float)controlsSize) / 64.0f);

	/* Binarizes dataset */
	for(int column_j=0; column_j < (numCols - 1); column_j++) {  
		int numSamples0Found = 0;
		int numSamples1Found = 0;
		for(int line_i=0; line_i < sampleSize; line_i++) {

			int datasetElement = dataset[line_i * numCols + column_j];
			if(dataset_target[line_i] == 1) {
				int Ones_index = datasetElement * numSamplesOnes_64packed * (numCols - 1) + ((int)(numSamples1Found / 64.0f)) * (numCols - 1) + column_j;
				datasetOnes_64packed[Ones_index] = datasetOnes_64packed[Ones_index] | (((unsigned long long) 1) << (numSamples1Found % 64));
				numSamples1Found++;
			}

			else {
				int Zeros_index = datasetElement * numSamplesZeros_64packed * (numCols - 1) + ((int)(numSamples0Found / 64.0f)) * (numCols - 1) + column_j;
				datasetZeros_64packed[Zeros_index] = datasetZeros_64packed[Zeros_index] | (((unsigned long long) 1) << (numSamples0Found % 64));
				numSamples0Found++;
			}

		}
	}

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        double timing_duration_loaddata = ((t_end.tv_sec + ((double) t_end.tv_nsec / 1000000000)) - (t_start.tv_sec + ((double) t_start.tv_nsec / 1000000000)));

	numSNPs = numCols - 1;
	printf("#SNPs: %d\n", numSNPs);
	numCombinations = mCk(numSNPs);
	printf("#combinations: %llu\n", numCombinations);


	int output_size = NUM_CPU_THREADS;

	cuda_mem_init(datasetOnes_64packed_size, datasetZeros_64packed_size, NUM_COMB_PER_KERNEL_INSTANCE * NUM_CPU_THREADS * EPISTASIS_SIZE, sampleSize, output_size);	

	float * outputFromGpu;
	outputFromGpu = sycl::malloc_host<float>(NUM_CPU_THREADS, q_ct1); 
	int * output_indexFromGpu;
	output_indexFromGpu = sycl::malloc_host<int>(NUM_CPU_THREADS, q_ct1);

	for(int i = 0; i < NUM_CPU_THREADS; i++) {
		outputFromGpu[i] = FLT_MAX;
	}

	float minScorePerCpuThread[NUM_CPU_THREADS];
	for(int i = 0; i < NUM_CPU_THREADS; i++) {
		minScorePerCpuThread[i] = FLT_MAX;
	}

	unsigned long long indexMinScorePerCpuThread[NUM_CPU_THREADS];

	float * h_lgammaPrecalc = (float*) malloc(sampleSize * sizeof(float));
        if(h_lgammaPrecalc == NULL) {
                printf("\nMemory allocation for internal representation (cases) failed.\n");
        }

	/* Precalculates lgamma() values */
	for(int i=1; i < (sampleSize + 1); i++) {
		h_lgammaPrecalc[i - 1] = lgamma((double)i);
	}

	cuda_mem_copy(datasetOnes_64packed, datasetOnes_64packed_size, datasetZeros_64packed, datasetZeros_64packed_size, NULL, 0, h_lgammaPrecalc, sampleSize, outputFromGpu, output_size);	

	uint * combinationArray;
	combinationArray = (uint *)sycl::malloc_host(sizeof(uint) * NUM_COMB_PER_KERNEL_INSTANCE * NUM_CPU_THREADS * EPISTASIS_SIZE, q_ct1);

	#pragma omp parallel for num_threads(NUM_CPU_THREADS) schedule(dynamic) 
	for (unsigned long long j = 0; j < (unsigned long long)ceil(numCombinations / (double)NUM_COMB_PER_KERNEL_INSTANCE); j++) {

		int omp_thread_id = omp_get_thread_num();
		gsl_combination * c = gsl_combination_calloc (numSNPs, EPISTASIS_SIZE);

		int numCombToGenerate = std::min((unsigned long long)NUM_COMB_PER_KERNEL_INSTANCE, numCombinations - (j * NUM_COMB_PER_KERNEL_INSTANCE));

		int startingCombination[EPISTASIS_SIZE];	

		int retCombGen = combination(startingCombination, numSNPs, 1 + ( j * NUM_COMB_PER_KERNEL_INSTANCE ));
		if(retCombGen != 0) {
			printf("Problem in iteration: %llu, starting combination index: %lld\n", j, 1 + ( j * NUM_COMB_PER_KERNEL_INSTANCE ));
			continue;
		}

		/* Sets combination to the one that the cpu thread must start with */
		size_t * combData = gsl_combination_data(c);
		for(int z=0; z<EPISTASIS_SIZE; z++) {
			combData[z] = startingCombination[z];
		}

		for(int comb_i = 0; comb_i < numCombToGenerate; comb_i++) {
			for(int z=0; z<EPISTASIS_SIZE; z++) {
				combinationArray[(omp_thread_id * NUM_COMB_PER_KERNEL_INSTANCE * EPISTASIS_SIZE) + z*numCombToGenerate + comb_i] = combData[z];
			}
			gsl_combination_next (c);
		}
		gsl_combination_free(c);

		q_ct1.memcpy( d_combinationArray + (omp_thread_id * NUM_COMB_PER_KERNEL_INSTANCE * EPISTASIS_SIZE),
			      combinationArray + (omp_thread_id * NUM_COMB_PER_KERNEL_INSTANCE * EPISTASIS_SIZE),
			      numCombToGenerate * EPISTASIS_SIZE * sizeof(uint) );
			 
		outputFromGpu[omp_thread_id] = FLT_MAX;
		q_ct1.memcpy(d_output + omp_thread_id, outputFromGpu + omp_thread_id, sizeof(float));

		cuda_launch_kernel(numSNPs, controlsSize, casesSize, numCombToGenerate, omp_thread_id, 0);
		
		q_ct1.memcpy(outputFromGpu + omp_thread_id, d_output + omp_thread_id, sizeof(float));
		q_ct1.memcpy(output_indexFromGpu + omp_thread_id, d_output_index + omp_thread_id, sizeof(int));

		q_ct1.wait();

		if(outputFromGpu[omp_thread_id] < minScorePerCpuThread[omp_thread_id]) {
			minScorePerCpuThread[omp_thread_id] = outputFromGpu[omp_thread_id];
			indexMinScorePerCpuThread[omp_thread_id] = j * NUM_COMB_PER_KERNEL_INSTANCE + output_indexFromGpu[omp_thread_id];
		}

	}

	float minScore = FLT_MAX;
	unsigned long long indexOfMinScore;
	for(int i = 0; i < NUM_CPU_THREADS; i++) {
		if(minScorePerCpuThread[i] < minScore) {
			minScore = minScorePerCpuThread[i];
			indexOfMinScore = indexMinScorePerCpuThread[i];
		}
	}

	printf("Solution with best score\n");
	int bestFoundCombination[EPISTASIS_SIZE];	
	combination(bestFoundCombination, numSNPs, 1 + indexOfMinScore);
	for(int comb_index=0; comb_index < EPISTASIS_SIZE; comb_index++) {
		printf("%d ", bestFoundCombination[comb_index]);
	}
	printf("%.2lf \n", minScore);

	cuda_clean_up();

	free(dataset);  
	free(dataset_target);
	sycl::free(outputFromGpu, q_ct1);
	sycl::free(output_indexFromGpu, q_ct1);
	sycl::free(combinationArray, q_ct1);

	clock_gettime(CLOCK_MONOTONIC, &t_end);

	double timing_duration_app = ((t_end.tv_sec + ((double) t_end.tv_nsec / 1000000000)) - (t_start.tv_sec + ((double) t_start.tv_nsec / 1000000000)));
        printf("Load+preprocess data:\t%0.3lf seconds\n", timing_duration_loaddata);
	printf("Total execution time:\t%0.3lf seconds\n", timing_duration_app);

	return 0;
}
catch (sycl::exception const &exc) {
	std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
	std::exit(1);
}
