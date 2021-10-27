# crossarch-episdet
> High-order exhaustive epistasis detection using K2 Bayesian scoring

High-order epistasis detection is a bioinformatics application with very high computational complexity that searches for correlations between genetic markers such as single-nucleotide polymorphisms (SNPs), single base changes in a DNA sequence that occur in at least 1% of a population, and phenotype (e.g. disease state).
Finding new associations between genotype and phenotype can contribute to improved preventive care, personalized treatments and to the development of better drugs for more conditions. 
This implementation has been obtained through translation of CUDA code using the Intel DPC++ Compatibility Tool.

Showcased during the [OneAPI Dev Summit 2020](https://www.oneapi.com/events/devcon2020/) keynote: Joe Curley, with guest appearance of Aleksandar Ilic. “oneAPI Vision for Heterogeneous Compute”, Intel oneAPI Developer Summit, keynote, virtual event, November 2020

## Installation

Downloading and installing dependencies:
```bash
        sh setup.sh
```

Compiling cross-architecture application binary targeting high-order (k=3) epistasis detection searches:
```bash
        make
```

## Usage example

Running in DevCloud for synthetic example dataset with 1024 SNPs (178,433,024 triplets of SNPs to evaluate) and 4096 samples:
```bash
        make run        # run on any type of DevCloud node
        make run_cpu    # run on DevCloud node with Intel Xeon Gold 6128 (CPU)
        make run_gpu    # run on DevCloud node with Intel Iris Xe MAX (GPU)
```

Use the `qstat` command to check if the search terminated on DevCloud, and when completed, access the output of the search with:
```bash
cat run.sh.o*
```

## Short Term Goals

Exploring techniques for achieving as high as possible throughput and energy-efficiency in high-order epistasis searches on challenging datasets, through:
* Developing software that makes full use in a collaborative way of the different types of available accelerator architectures and devices, including state-of-the-art Intel CPUs and GPUs;
* High scalability using heterogeneous and homogeneous cluster configurations with different combinations of types of computing devices;
* Using technologies, APIs and programming languages (OneAPI, DPC++) built around open standards (C++, SYCL) as a way to avoid proprietary lock-in.
