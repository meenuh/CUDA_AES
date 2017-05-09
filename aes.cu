
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aes.h"

__constant__ unsigned char D_SBOX[16][16];
__constant__ unsigned char D_INV_SBOX[16][16];

//Global expanded key array - 44 words (4 words per round * 11 round keys needed)
unsigned char ExpandKey[44][4];

//Global state matrix which will store the data on each AES round
unsigned char State[4][4];

__host__ void keyExpansion(unsigned char * Key) {
	//Temp word to store the previous word and to rotate word
	unsigned char temp[4], rtemp[4];
	
	//Word iteration (0-3). 4 words is 1 round key.
    int word_it;
	
	//Copies the first 8 words. First 4 is first round key. Second 4 used to generate next round key
    memcpy(ExpandKey, Key, 32);

	//Calculate the rest of the words for the expanded key
	//44 rounds to generate 11 round keys, first already created
	for (int Round_it = 4; Round_it < 44; Round_it++) { 
        
		//Get the last created word
        memcpy(temp, ExpandKey[Round_it - 1], 4);
		
		//This only occurs in every 4 rounds iterations (last word in a round)
        if ((Round_it % 4) == 0) { 
			
			//Every 4th word, rotate by 1 byte, apply SBOX, and XOR RCON
			rtemp[4] = {temp[0], temp[1], temp[2], temp[3]};
			for (word_it = 0; word_it < 4; word_it++) {
				temp[word_it] = rtemp[(word_it + 1) % 4];
			}
			
			//Apply SBOX (using host const SBOX, not D_SBOX)
            for (word_it = 0; word_it < 4; ++word_it){
				temp[word_it] = SBOX[(temp[word_it]&0xf0) >> 4][temp[word_it]&0x0f];
			}
        
			//XOR of the first element with the corresponding RCON value (elements 2-4 are always 0 in RCON)
            temp[0] ^= RCon[Round_it / 4];
        }
        
		//For all the words, apply an XOR with the block 4 bytes before the current one
        for (word_it = 0; word_it < 4; word_it++) {
			temp[word_it] ^= ExpandKey[Round_it - 4][word_it];
        }
		
		// Store the new word on th expanded key
        memcpy(ExpandKey[Round_it], temp, 4);
	}
}

__global__ void AESKernel(char *x)
{
	printf("%02x %d %d\n", D_INV_SBOX[threadIdx.x][threadIdx.y], threadIdx.x, threadIdx.y);
	//printf("%02x", D_SBOX[threadIdx.x]);
}

int main(int argc, char **argv)
{
	//CUDA error variable
    cudaError_t cuda_error;
	
	AES_INFO *data = get_args(argv, argc);
	char *h = (char *)malloc(256);
	for (int i = 0; i < 256; i++)
		h[i] = i;

	char *arr;
	cudaMalloc((void**)&arr, sizeof(char) * 256);
	cudaMemset(arr, 0, sizeof(char)*256);

	cuda_error = cudaMemcpyToSymbol(D_SBOX, SBOX, sizeof(char) * 256);
	if (cuda_error != cudaSuccess) {
		printf("error on copy SBOX");
	}
	cuda_error = cudaMemcpyToSymbol(D_INV_SBOX, INVERSE_SBOX, sizeof(char) * 256)
	if (cuda_error != cudaSuccess) {
		printf("error on copy INV_SBOX");
	}

	cudaMemcpy(arr, h, 256, cudaMemcpyHostToDevice);

	dim3 block(16, 16);
	
	keyExpansion(data->key);
	AESKernel <<<1, block>>> (arr);
	cuda_error = cudaGetLastError();
	if(cuda_error != cudaSuccess){
		fprintf(stderr, "err %s", cuda_error);
	}
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

	//get the key file
	//char buffer[KEYSIZE]; //Why do you need a buffer for key file read?

	FILE *key_fp = fopen(argv[2], "rb");
	if (NULL == key_fp) {
		printf("Error opening keyfile: %s\n", argv[2]);
		exit(EXIT_FAILURE);
	}

	fread(info->key, 1, KEYSIZE, key_fp);
	//strncpy(info->key, buffer, KEYSIZE);
	
	printf("Using key: ");
	for (int i = 0; i < KEYSIZE; i++) {
		if(info->key[1] == NULL){
			printf("Error: Invalid key file, missing value at %d\n", i);
			exit(EXIT_FAILURE);
		}
		printf("%02x ", info->key[i]);
	}

	//get the input FILE * to encrypt/decrypt
	info->fin = fopen(argv[3], "rb");
	if (NULL == info->fin) {
		printf("Error opening input file %s\n", argv[3]);
		exit(EXIT_FAILURE);
	}
	printf("\nUsing input file %s\n", argv[3]);

	//get the output file name to be used later
	info->output_filename = (char *)malloc(strlen(argv[4]) + 1);
	strcpy(info->output_filename, argv[4]);
	printf("Output file is %s\n", argv[4]);

	return info;
}
