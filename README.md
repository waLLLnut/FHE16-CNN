# FHE16-CNN on MNIST 

A small CNN trained on MNIST and executed over encrypted data using the **FHE16** homomorphic encryption scheme.

- **Dataset:** MNIST (handwritten digits)
- **Architecture:** Conv(3×3, stride=3) → ReLU → SumPool(3×3) → FC(27→10)

---

### Run Instructions

#### (Optional) Train a new CNN model
```bash
python train_mnist_fhe_cnn.py --epochs 15
```

#### build essentials
```bash
sudo apt install libjemalloc2
```

#### Download MNIST samples
```bash
python export_mnist_samples.py
```

#### Build
```bash
mkdir build
mv run_all.sh build/
cd build
cmake ..
make
```

#### Set library path (required before running)
```bash
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../FHE_TEST/lib' >> ~/.bashrc
source ~/.bashrc
```

#### Run the executables
```bash
./test          # Inference for a single MNIST sample
./run_all.sh    # Run inference for 1000 MNIST samples (divided into 10 batches)
                 # Results will be saved to accuracy_log.txt
```


