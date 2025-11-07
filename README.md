### Run Instructions

(Optional) Train a new CNN model
python train_mnist_fhe_cnn.py

Run the encrypted inference
python export_mnist_samples.py
cd build
cmake ..
make

Set library path (required before running):
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../FHE_TEST/lib' >> ~/.bashrc
source ~/.bashrc

After that, run the executables:
./test          # Inference for a single MNIST sample
./run_all.sh    # Run inference for 1000 MNIST samples (divided into 10 batches)
                 # Results will be saved to accuracy_log.txt

