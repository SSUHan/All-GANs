mkdir -p common/data/mnist
mkdir -p common/data/fashion_mnist

wget -P common/data/fashion_mnist http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget -P common/data/fashion_mnist http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget -P common/data/fashion_mnist http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget -P common/data/fashion_mnist http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

wget -P common/data/mnist http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget -P common/data/mnist http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget -P common/data/mnist http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget -P common/data/mnist http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz