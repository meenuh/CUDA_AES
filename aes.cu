
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aes.h"

__constant__ unsigned char D_SBOX[16][16];
__constant__ unsigned char D_INV_SBOX[16][16];

__device__ void keyExpansion() {

}

__global__ void AESKernel(char *x)
{
	printf("%02x %d %d\n", D_INV_SBOX[threadIdx.x][threadIdx.y], threadIdx.x, threadIdx.y);
	//printf("%02x", D_SBOX[threadIdx.x]);
}

int main(int argc, char **argv)
{
	AES_INFO *data = get_args(argv, argc);
	char *h = (char *)malloc(256);
	for (int i = 0; i < 256; i++)
		h[i] = i;

	char *arr;
	cudaMalloc((void**)&arr, sizeof(char) * 256);
	cudaMemset(arr, 0, sizeof(char)*256);

	if (cudaMemcpyToSymbol(D_SBOX, SBOX, sizeof(char) * 256) != cudaSuccess) {
		printf("error on copy");
	}
	if (cudaMemcpyToSymbol(D_INV_SBOX, INVERSE_SBOX, sizeof(char) * 256) != cudaSuccess) {
		printf("error on copy");
	}

	cudaMemcpy(arr, h, 256, cudaMemcpyHostToDevice);

	dim3 block(16, 16);

	AESKernel << <1, block>> > (arr);
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess)
		fprintf(stderr, "err %s", cudaGetLastError());

	cudaMemcpy(h, arr, 256, cudaMemcpyDeviceToHost);
    return 0;
}

//arg[0] - program name - not used
//arg[1] - Encrypt or decrypt(e for encrypt, d for decrypt)
//arg[2] - Filename containing our key
//arg[3] - Input file to encrypt / decrypt
//arg[4] - Output file name
AES_INFO *get_args(char **argv, int argc) {
	AES_INFO *info = (AES_INFO *)malloc(sizeof(AES_INFO));
	if (NULL == info) {
		printf("Error on malloc.\n");
		exit(EXIT_FAILURE);
	}

	if (argc != 5) {
		printf("Incorrect number of arguments. Found %d\n", argc);
		exit(EXIT_FAILURE);
	}

	switch (*argv[1]) {
	case 'e':
		info->mode = ENCRYPT;
		break;
	case 'd':
		info->mode = DECRYPT;
		break;
	default:
		printf("ERROR. First argument should be e for Encrypt or d for Decrypt.\n");
		exit(EXIT_FAILURE);
	}

	//get the key
	char buffer[KEYSIZE];

	FILE *key_fp = fopen(argv[2], "rb");
	if (NULL == key_fp) {
		printf("Error opening keyfile: %s\n", argv[2]);
		exit(EXIT_FAILURE);
	}

	fread(buffer, 1, KEYSIZE, key_fp);
	strncpy(info->key, buffer, KEYSIZE);
	
	printf("Using key: ");
	for (int i = 0; i < KEYSIZE; i++) {
		printf("%02x ", info->key[i]);
	}

	//get FILE * to file to encrypt/decrypt
	info->fin = fopen(argv[3], "rb");
	if (NULL == info->fin) {
		printf("Error opening input file %s\n", argv[3]);
		exit(EXIT_FAILURE);
	}
	printf("\nUsing input file %s\n", argv[3]);

	//get output file name to be used later
	info->output_filename = (char *)malloc(strlen(argv[4]) + 1);
	strcpy(info->output_filename, argv[4]);
	printf("Output file is %s\n", argv[4]);

	return info;
}
