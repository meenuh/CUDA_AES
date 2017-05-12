COMMAND LINE OPTIONS FOR OUR PROGRAM (AES.cu & AESO.cu)
<mode> <inputfile> <outputfile> <keyfile>

mode: d or e
	- d for decrypt
	- e for encrypt

inputfile: file to encrypt/decrypt
outputfile: file to write encrypted/decrypted data to
keyfile: file containing key (NOTE: MUST BE 16 bytes)

COMMAND LINE OPTIONS FOR OTHER PROGRAM (AESG.cu & AESG.c)
<mode> <keylength> <inputfile> <outputfile> <keyfile>

mode: d or e
	- d for decrypt
	- e for encrypt
keylength: length of key
	- 1 for 128 bit
	- 2 for 192 bit
	- 3 for 256 bit

inputfile: file to encrypt/decrypt
outputfile: file to write encrypted/decrypted data to
keyfile: file containing key (NOTE: MUST BE 16/24/32 bytes based on keylength)