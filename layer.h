#include <cstdlib>
#include <vector>
#include <memory>
//#include <cublas_v2.h>
#include <cuda.h>

#ifndef LAYER_H
#define LAYER_H
#endif


class Layer {
	public:
	/* Layer input params x y z H w D*/
	int x;
	int y;
	int z;
	/* Forward prop params */
	float *output;		// Ouput that comes after the activation of a layer
	float *layer_out;	// Layer that comes after a convolution
	float *bias;		// Bias term added to the layer_out before activation
	float *kernel;		// Filter window
	/* Backprop params */
	float *d_output;	// Derivative output backprop
	float *d_layer_out;	// Derivative layer out, backprop
	float *d_kernel;	// Derivative kernel, backprop

	/* Default layer constructor */
	Layer(int x, int y, int z);
	/* Destructor */
	~Layer();

	void set_dims(float *data);
	/* Zero out the memory on the GPU */
	void clear();
	/* Zero out the GPU memory after back prop */
	void clear_backprop();
};

/* Activation Functions */
__device__ float activation_function(float v);
__global__ void activate(float *input, float *output, const int N);

/* Calculate error */
__global__ void error(float *err, float *output, unsigned int index, const int N);
__global__ void gradient(float *output, float *grad, const int y);


/*........:::::::: Forward Propagation Kernel functions ::::::::........*/
/* Forward Convolution Layer 1 Kernel Functions */
__global__ void forward_prop_CONV(float input[250][250], float layer_out[9][225][225], float kernel[9][26][26]);
__global__ void forward_bias_CONV(float layer_out[9][225][225], float bias[9]);

/* Forward Convolution Layer 2 Kernel Functions*/
__global__ void forward_prop_CONV2(float input[9][225][225], float layer_out[9][9][9], float kernel[1][25][25]);
__global__ void forward_bias_CONV2(float layer_out[9][9][9], float bias[1]);

/* Forward Flatten Layer Kernel Functions*/
__global__ void forward_prop_FLATTEN(float input[9][9][9], float layer_out[610], float kernel[610][9][9][9]);
__global__ void forward_bias_FLATTEN(float layer_out[610], float bias[610]);


/*........:::::::: Back Propagation Kernel Functions ::::::::........*/
/* Back Flatten Layer Kernel Functions */
__global__ void back_kernel_FLATTEN(float d_kernel[610][9][9][9], float d_layer_out[610], float p_output[9][9][9]);
__global__ void back_bias_FLATTEN(float bias[610], float d_layer_out[610]);

/* Back Convolution Layer 2 Kernel Functions */
__global__ void back_out_CONV2(float d_output[9][9][9], float n_kernel[610][9][9][9], float nd_layer_out[610]);
__global__ void back_prop_CONV2(float d_layer_out[9][9][9], float d_output[9][9][9], float layer_out[9][9][9]);
__global__ void back_kernel_CONV2(float d_kernel[1][25][25], float d_layer_out[9][9][9], float p_output[9][225][225]);
__global__ void back_bias_CONV2(float bias[1], float d_layer_out[9][9][9]);

/* Back Convolution Layer 1 Kernel Functions */
__global__ void back_out_CONV(float d_output[9][225][225], float n_kernel[1][25][25], float nd_layer_out[9][9][9]);
__global__ void back_prop_CONV(float d_layer_out[9][225][225], float d_output[9][225][225], float layer_out[9][225][225]);
__global__ void back_kernel_CONV(float d_kernel[9][26][26], float d_layer_out[9][225][225], float p_output[250][250]);
__global__ void back_bias_CONV(float bias[9], float d_layer_out[9][225][225]);
