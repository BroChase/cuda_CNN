#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "layer.h"

#include <cuda.h>
#include <cstdio>
#include <time.h>


/* Functions */
static void training();
static inline void loaddata();
/* Vars */
static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

static Layer INPUT_layer = Layer(0, 0, 250*250); 		// InputLayer 250 x 250 x 1
static Layer CONV_layer = Layer(26*26, 9, 225*225*9); 	// Conv Layer 250x250x1 (26x26 filter window stride 1) 9 Kernels --> 225*225*9 output
static Layer CONV2_layer = Layer(25*25, 1, 9*9*9); 		// Maxpooling Layer (225x225x9 input) 25x25 filter --> 9x9x9 output
static Layer FLATTEN_layer = Layer(9*9*9, 610, 610);	// Output Flatten Layer (9x9x9 input) --> 610 output


int main(int argc, const  char **argv)
{
	printf("Start load \n");
	loaddata();

	printf("Learning Phase\n");
	training();

	return 0;
}


static inline void loaddata()
{
	mnist_load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte",
		&test_set, &test_cnt);
}

static double forward_prop(double data[250][250]){
	float input[250][250];

	for(int i = 0; i < 250; i++){
		for(int j = 0; j < 250; j++){
			input[i][j] = data[i][j];
		}
	}
	INPUT_layer.clear();
	CONV_layer.clear();
	CONV2_layer.clear();
	FLATTEN_layer.clear();

	clock_t start, end;

	start = clock();
	INPUT_layer.set_dims((float *)input);
	
	/* Convolutional layer 1 Kernel Functions */
	forward_prop_CONV<<<256, 256>>>((float (*)[250])INPUT_layer.output, (float (*)[225][225])CONV_layer.layer_out, (float (*)[26][26])CONV_layer.kernel);
	forward_bias_CONV<<<256, 256>>>((float (*)[225][225])CONV_layer.layer_out, CONV_layer.bias);
	activate<<<256, 256>>>(CONV_layer.layer_out, CONV_layer.output,CONV_layer.z);

	/* Convolutional layer 2 Kernel Functions */
	forward_prop_CONV2<<<256, 256>>>((float (*)[225][225])CONV_layer.output, (float (*)[9][9])CONV2_layer.layer_out, (float (*)[25][25])CONV2_layer.kernel);
	forward_bias_CONV2<<<256, 256>>>((float (*)[9][9])CONV2_layer.layer_out, CONV2_layer.bias);
	activate<<<256, 256>>>(CONV2_layer.layer_out, CONV2_layer.output, CONV2_layer.z);

	/* Flatten Layer Kernel Functions */
	forward_prop_FLATTEN<<<256, 256>>>((float (*)[9][9])CONV2_layer.output, FLATTEN_layer.layer_out, (float (*)[9][9][9])FLATTEN_layer.kernel);
	forward_bias_FLATTEN<<<256, 256>>>(FLATTEN_layer.layer_out, FLATTEN_layer.bias);
	activate<<<256, 256>>>(FLATTEN_layer.layer_out, FLATTEN_layer.output, FLATTEN_layer.z);
	
	
	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

static double back_prop(){
	clock_t start, end;
	start = clock();
	
	/* Back Propagation Flatten Layer Kernel Functions */
	back_kernel_FLATTEN<<<256, 256>>>((float (*)[9][9][9])FLATTEN_layer.d_kernel, FLATTEN_layer.d_layer_out, (float (*)[9][9])CONV2_layer.output);
	back_bias_FLATTEN<<<256, 256>>>(FLATTEN_layer.bias, FLATTEN_layer.d_layer_out);
	
	/* Back Propagation CONV2 Layer Kernel Functions */
	back_out_CONV2<<<256, 256>>>((float (*)[9][9])CONV2_layer.d_output, (float (*)[9][9][9])FLATTEN_layer.kernel, FLATTEN_layer.d_layer_out);
	back_prop_CONV2<<<256, 256>>>((float (*)[9][9])CONV2_layer.d_layer_out, (float (*)[9][9])CONV2_layer.d_output, (float (*)[9][9])CONV2_layer.layer_out);
	back_kernel_CONV2<<<256, 256>>>((float (*)[25][25])CONV2_layer.d_kernel, (float (*)[9][9])CONV2_layer.d_layer_out, (float (*)[225][225])CONV_layer.output);
	back_bias_CONV2<<<256, 256>>>(CONV2_layer.bias, (float (*)[9][9])CONV2_layer.d_layer_out);
	/* Back Propagation CONV1 Layer Kernel Functions */
	back_out_CONV<<<256, 256>>>((float (*)[225][225])CONV_layer.d_output, (float (*)[25][25])CONV2_layer.kernel, (float (*)[9][9])CONV2_layer.layer_out);
	back_prop_CONV<<<256, 256>>>((float (*)[225][225])CONV_layer.d_layer_out, (float (*)[225][225])CONV_layer.d_output, (float (*)[225][225])CONV_layer.layer_out);
	back_kernel_CONV<<<256, 256>>>((float (*)[26][26])CONV_layer.d_kernel, (float (*)[225][225])CONV_layer.d_layer_out, (float (*)[250])INPUT_layer.output);
	back_bias_CONV<<<256, 256>>>(CONV_layer.bias, (float (*)[225][225])CONV_layer.layer_out);

	/* Apply the gradienst from the derivative kernel weights to the layer weights for the flatten conv2 and conv layers */
	gradient<<<256, 256>>>(FLATTEN_layer.kernel, FLATTEN_layer.d_kernel, FLATTEN_layer.x * FLATTEN_layer.y);
	gradient<<<256, 256>>>(CONV2_layer.kernel, CONV2_layer.d_kernel, CONV2_layer.x * CONV2_layer.y);
	gradient<<<256, 256>>>(CONV_layer.kernel, CONV_layer.d_kernel, CONV_layer.x * CONV_layer.y);

	end = clock();
	return ((double)(end - start)) / CLOCKS_PER_SEC;
}


static void training(){
	//static cublasHandle_t blas;
	//cublasCreate(&blas);

	float training_error;
	int epochs = 50;

	float runtime = 0.0;
	while(epochs < 0 || epochs-- > 0){
		training_error = 0.0f;
		for(int i=0; i < train_cnt; i++){
			float err;

			runtime += forward_prop(train_set[i].data);
			
			CONV_layer.clear_backprop();
			CONV2_layer.clear_backprop();
			FLATTEN_layer.clear_backprop();

			error<<<610, 1>>>(FLATTEN_layer.d_layer_out, FLATTEN_layer.output, train_set[i].label, 610);
			//cublasSnrm2(blas, 610, FLATTEN_layer.layer_out, 1, &err);
			training_error += err;

			runtime += back_prop();
		}

		training_error /= train_cnt;
		//printf("error: %e\n", training_error);
		printf("Epoch: %d, Runtime: %f\n", epochs, runtime*100);
	}

}
