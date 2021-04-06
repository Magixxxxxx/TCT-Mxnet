import mxnet as mx
a = mx.nd.array([[1,2,3],[4,5,6]])

print (a.reshape((-4,1,2,-2))) # fail
a.reshape((1,2,-2)) # fail
