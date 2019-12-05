all:
	ssh node18 nvcc /home/chase.brown/csc5551/Final_Project/CNN_CUDA/main.cu /home/chase.brown/csc5551/Final_Project/CNN_CUDA/layer.cu -o /home/chase.brown/csc5551/Final_Project/CNN_CUDA/CNN
clean:
	rm CNN
