#include "layer.h"

/* Defualt Layer Constructor */
Layer::Layer(int x, int y, int z){
	this->x = x;
	this->y = y;
	this->z = z;
	// printf("Set1\n");
	float bias_values[y];
	float kernel_values[y][x];

	output = NULL;
	layer_out = NULL;
	bias   = NULL;
	kernel = NULL;
	

	for (int i = 0; i < y; ++i) {
		/* Random Initialization of Bias terms */
		bias_values[i] = 0.5f - float(rand()) / float(RAND_MAX);
		/* Random Initialization of Filter Windows */
		for (int j = 0; j < x; ++j) {
			kernel_values[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
		}
	}

	/* Allocate memory on GPU */
	cudaMalloc(&output, sizeof(float) * z);
	cudaMalloc(&layer_out, sizeof(float) * z);
	cudaMalloc(&bias, sizeof(float) * y);
	cudaMalloc(&kernel, sizeof(float) * x * y);
	cudaMalloc(&d_output, sizeof(float) * z);
	cudaMalloc(&d_layer_out, sizeof(float) * z);
	cudaMalloc(&d_kernel, sizeof(float) * x * y);
	cudaMemcpy(bias, bias_values, sizeof(float) * y, cudaMemcpyHostToDevice);
	cudaMemcpy(kernel, kernel_values, sizeof(float) * x * y, cudaMemcpyHostToDevice);
}

/* Layer Destructor function */
Layer::~Layer(){
	cudaFree(output);
	cudaFree(layer_out);
	cudaFree(bias);
	cudaFree(kernel);
	cudaFree(d_output);
	cudaFree(d_layer_out);
	cudaFree(d_kernel);
}

/* Send data i at a time */
void Layer::set_dims(float *data){
	cudaMemcpy(output, data, sizeof(float) * z, cudaMemcpyHostToDevice);
}

/* Clear the GPU memory after each iteration */
void Layer::clear(){
	cudaMemset(output, 0x00, sizeof(float) * z);
	cudaMemset(layer_out, 0x00, sizeof(float) * z);
}

/* Clear the GPU memory after Back prop */
void Layer::clear_backprop(){
	cudaMemset(d_output, 0x00, sizeof(float) * z);
	cudaMemset(d_layer_out, 0x00, sizeof(float) * z);
	cudaMemset(d_kernel, 0x00, sizeof(float) * x * y);
}

/* 
	Activation function used for activation of CNN layers and Flatten layer
	@param float v: Input float for activation.
	Sigmoid activation function
*/
__device__ float activation_function(float v){
	return 1 / (1 + exp(-v));
}

/*
	activate
	@param float input: Pointer to input layer
	@param float output: Pointer to out put layer
	@param cont N: Dim of the current visual field
	Passes the values from the input window through the activation function (sigmoid)
	Stores the results in the output field
*/
__global__ void activate(float *input, float *output, const int N){
	
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] = activation_function(input[idx]);
	}
}
/*
	error function 
	@param float err: Pointer to current temp error of epoch
	@param float output: Output of the current layer.
	@param unsighed int Y: If Y is equal to index number
		subtract the output value at index number
	@param const int N: Dim of field of view 
*/
__global__ void error(float *err, float *output, unsigned int index, const int N){
	
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		err[idx] = ((index == idx ? 1.0f : 0.0f) - output[idx]);
	}
}
/*
	gradient function
	@param float output: Output results
	@param float grad: Pointer to the saved gradients
	@param const int N: Dim of field of view
	Gradient * the derivative added to the output arrays index value
*/
__global__ void gradient(float *output, float *grad, const int N){
	
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] += 1.0E-01f * grad[idx];
	}
}


/*........:::::::: Convolution Layer 1 Kernel Functions Forward/Back Prop ::::::::........*/
/* FORWARD */
/*
	forward_prop_CONV
	@param float input: Input 2darray for image (250x250x1)
	@param float layer_out: Results of the filter window stride (225x225x9)
	@param float kernel: The kernel window (26*26*9)
*/
__global__ void forward_prop_CONV(float input[250][250], float layer_out[9][225][225], float kernel[9][26][26]){
	
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 26*26*9*225*225;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 26);
		const int i2 = ((idx /= 26	) % 26);
		const int i3 = ((idx /= 26	) % 9);
		const int i4 = ((idx /= 9	) % 225);
		const int i5 = ((idx /= 225	) % 225);
		/*
			atomicADD:
				Reads the old value and adds it with the new value and stores it in the old values address
				atomicAdd(*address, value);
		*/
		atomicAdd(&layer_out[i3][i4][i5], kernel[i3][i1][i2] * input[i4 + i1][i5 + i2]);
	}
}

/*
	forward_bias_CONV
	@param float layer_out: The output layer after the first convolution 
	@param float bias: Bias terms to add to the layer_outu
	Adds the bias terms to the layer_output before activation
*/
__global__ void forward_bias_CONV(float layer_out[9][225][225], float bias[9]){
	
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 9*225*225;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 9);
		const int i2 = ((idx /= 9	) % 225);
		const int i3 = ((idx /= 225	) % 225);

		layer_out[i1][i2][i3] += bias[i1];
	}
}

/* BACK */
/*
	back_out_CONV
	@param float d_output: Derivative of the output layer of CONV
	@param float n_kernel: The filter window
	@ param float nd_layer_out: The layer out from CONV
	Updates the output layer during back propagation
*/
__global__ void back_out_CONV(float d_output[9][225][225], float n_kernel[1][25][25], float nd_layer_out[9][9][9]){
	
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1*25*25*9*9*9;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 1);
		const int i2 = ((idx /= 1	) % 25);
		const int i3 = ((idx /= 25	) % 25);
		const int i4 = ((idx /= 25	) % 9);
		const int i5 = ((idx /= 9	) % 9);
		const int i6 = ((idx /= 9	) % 9);

		atomicAdd(&d_output[i4][i5 * 4 + i2][i6 * 4 + i3], n_kernel[i1][i2][i3] * nd_layer_out[i4][i5][i6]);
	}
}

/*
	back_prop_CONV
	@param float d_layer_out: The derivative of the layer_out from forward prop
	@param float d_output: The derivative output function after the layer_output activation
	@param float layer_out: The original output from first Layer
*/
__global__ void back_prop_CONV(float d_layer_out[9][225][225], float d_output[9][225][225], float layer_out[9][225][225]){
	
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 9*225*225;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 9);
		const int i2 = ((idx /= 9	) % 225);
		const int i3 = ((idx /= 225	) % 225);

		const float sig = activation_function(layer_out[i1][i2][i3]);

		d_layer_out[i1][i2][i3] = d_output[i1][i2][i3] * sig * (1 - sig);
	}
}
/*
	back_kernel_CONV
	@param float d_kernel: derivative of the filter window back prop
	@param float d_layer_output: The derivative of the layer_out from forward prop
	@param float p_output: The input derivative 
	Update the kernel window back prop
*/
__global__ void back_kernel_CONV(float d_kernel[9][26][26], float d_layer_out[9][225][225], float p_output[250][250]){
	
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 9*26*26*225*225;
	const float d = pow(225.0f, 2.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 9);
		const int i2 = ((idx /= 9	) % 26);
		const int i3 = ((idx /= 26	) % 26);
		const int i4 = ((idx /= 26	) % 225);
		const int i5 = ((idx /= 225	) % 225);

		atomicAdd(&d_kernel[i1][i2][i3], d_layer_out[i1][i4][i5] * p_output[i4 + i2][i5 + i3] / d);
	}
}

/*
	back_bias_CONV
	@param float bias: Bias values
	@param d_layer_out: derivative layer_out
	Back prop update bias terms for CONV layer
*/
__global__ void back_bias_CONV(float bias[9], float d_layer_out[9][225][225]){
	
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 9*225*225;
	const float d = pow(225.0f, 2.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 9);
		const int i2 = ((idx /= 9	) % 225);
		const int i3 = ((idx /= 225	) % 225);

		atomicAdd(&bias[i1], 1.0E-01f * d_layer_out[i1][i2][i3] / d);
	}
}
/*........:::::::: END Convolution Layer 1 Kernel Functions Forward/Back Prop ::::::::........*/





/*........:::::::: Convolution Layer 2 Kernel Functions Forward/Back Prop ::::::::........*/
/* FORWARD */
/*
	forward_prop_CONV2
	@param float input: Input 2darray for image (225x225x9)
	@param float layer_out: Results of the filter window stride (9x9x9)
	@param float kernel: The kernel window (25*25*1)
*/
__global__ void forward_prop_CONV2(float input[9][225][225], float layer_out[9][9][9], float kernel[1][25][25]){

	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 25*25*9*9*9;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 25);
		const int i2 = ((idx /= 25	) % 25);
		const int i3 = ((idx /= 25	) % 9);
		const int i4 = ((idx /= 9	) % 9);
		const int i5 = ((idx /= 9	) % 9);

		atomicAdd(&layer_out[i3][i4][i5], kernel[0][i1][i2] * input[i3][i4 * 4 + i1][i5 * 4 + i2]);
	}
}

/*
	forward_bias_CONV2
	@param float layer_out: The output layer after the first convolution 
	@param float bias: Bias terms to add to the layer_outu
	Adds the bias terms to the layer_output before activation
*/
__global__ void forward_bias_CONV2(float layer_out[9][9][9], float bias[1]){

	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 9*9*9;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 9);
		const int i2 = ((idx /= 9	) % 9);
		const int i3 = ((idx /= 9	) % 9);

		layer_out[i1][i2][i3] += bias[0];
	}
}

/* BACK */
/*
	back_out_CONV2
	@param float d_output: Derivative of the output layer of CONV2
	@param float n_kernel: The filter window
	@ param float nd_layer_out: The layer out from CONV2
	Updates the output layer during back propagation
*/
__global__ void back_out_CONV2(float d_output[9][9][9], float n_kernel[610][9][9][9], float nd_layer_out[610]){
	
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 610*9*9*9;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 610);
		const int i2 = ((idx /= 610	) % 9);
		const int i3 = ((idx /= 9	) % 9);
		const int i4 = ((idx /= 9	) % 9);

		atomicAdd(&d_output[i2][i3][i4], n_kernel[i1][i2][i3][i4] * nd_layer_out[i1]);
	}
}

/*
	back_prop_CONV2
	@param float d_layer_out: The derivative of the layer_out from forward prop
	@param float d_output: The derivative output function after the layer_output activation
	@param float layer_out: The original output from first Layer
*/
__global__ void back_prop_CONV2(float d_layer_out[9][9][9], float d_output[9][9][9], float layer_out[9][9][9]){
	
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 9*9*9;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 9);
		const int i2 = ((idx /= 9	) % 9);
		const int i3 = ((idx /= 9	) % 9);

		const float sig = activation_function(layer_out[i1][i2][i3]);

		d_layer_out[i1][i2][i3] = d_output[i1][i2][i3] * sig * (1 - sig);
	}
}

/*
	back_kernel_CONV
	@param float d_kernel: derivative of the filter window back prop
	@param float d_layer_output: The derivative of the layer_out from forward prop
	@param float p_output: The input derivative 
	Update the kernel window back prop
*/
__global__ void back_kernel_CONV2(float d_kernel[1][25][25], float d_layer_out[9][9][9], float p_output[9][225][225]){
	
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1*25*25*9*9*9;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 1);
		const int i2 = ((idx /= 1	) % 25);
		const int i3 = ((idx /= 25	) % 25);
		const int i4 = ((idx /= 25	) % 9);
		const int i5 = ((idx /= 9	) % 9);
		const int i6 = ((idx /= 9	) % 9);

		atomicAdd(&d_kernel[i1][i2][i3], d_layer_out[i4][i5][i6] * p_output[i4][i5 * 4 + i2][i6 * 4 + i3]);
	}
}

/*
	back_bias_CONV
	@param float bias: Bias values
	@param d_layer_out: derivative layer_out
	Back prop update bias terms for CONV layer
*/
__global__ void back_bias_CONV2(float bias[1], float d_layer_out[9][9][9]){
	
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 9*9*9;
	const float d = pow(9.0f, 3.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 9);
		const int i2 = ((idx /= 9	) % 9);
		const int i3 = ((idx /= 9	) % 9);

		atomicAdd(&bias[0], 1.0E-01f * d_layer_out[i1][i2][i3] / d);
	}
}

/*........:::::::: END Convolution Layer 2 Kernel Functions Forward/Back Prop ::::::::........*/





/*........:::::::: Flatten Layer Kernel Functions Forward/Back Prop :::::::........*/
/* FORWARD */
/*
	forward_prop_FLATTEN
	@param float input: input layer from the CONV2 layer
	@param float layer_out: ouput layer before activation functions
	@param float kernel: The kernel filter: filter takes the 9x9x9 input and outputs to 610 neurons
*/
__global__ void forward_prop_FLATTEN(float input[9][9][9], float layer_out[610], float kernel[610][9][9][9]){
	
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 610*9*9*9;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 610);
		const int i2 = ((idx /= 610	) % 9);
		const int i3 = ((idx /= 9	) % 9);
		const int i4 = ((idx /= 9	) % 9);

		atomicAdd(&layer_out[i1], kernel[i1][i2][i3][i4] * input[i2][i3][i4]);
	}
}

/*
	forward_bias_FLATTEN
	@param float layer_out: Number of neurons in last layer
	@param float bias: Bias terms to add to the neurons
*/
__global__ void forward_bias_FLATTEN(float layer_out[610], float bias[610]){
	
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 610;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		layer_out[idx] += bias[idx];
	}
}

/* BACK */
/*
	back_kernel_FLATTEN
	@param float d_kernel: derivative of the kernel
	@param float d_layer_out: derivative of the output
	@param output: Prevous layers values
*/
__global__ void back_kernel_FLATTEN(float d_kernel[610][9][9][9], float d_layer_out[610], float p_output[9][9][9]){
	
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 610*9*9*9;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 610);
		const int i2 = ((idx /= 610	) % 9);
		const int i3 = ((idx /= 9	) % 9);
		const int i4 = ((idx /= 9	) % 9);

		d_kernel[i1][i2][i3][i4] = d_layer_out[i1] * p_output[i2][i3][i4];
	}
}

/*
	back_bias_FLATTEN
	@param float bias: Flatten layers bias terms
	@param float d_layer_out: Flatten layers derivative outputs
*/
__global__ void back_bias_FLATTEN(float bias[610], float d_layer_out[610]){
	
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 610;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		bias[idx] += 1.0E-01f * d_layer_out[idx];
	}
}
/*........::::::::END Flatten Layer Kernel Functions Forward/Back Prop ::::::::........*/