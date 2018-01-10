from six.moves import cPickle
f = open("cifar-10-batches-py/test_batch", 'rb')
datadict = cPickle.load(f,encoding='latin1')
f.close()
X = datadict["data"]
Y = datadict['labels']
print(Y)