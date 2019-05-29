## Tensorflow 实战Google深度学习框架 学习笔记

### 5.29

#### 第三章

##### 计算图的概念：

 Tensorflow程序一般分为两个阶段：

1. 定义计算图，申明张量（tensor），这里的张量即代表多维数组，声明张量时为定义其三个属性，名字，维度，类型（进行计算时，类型要匹配，否则会报错）。

   `Tensor("a:0", shape=(2,), dtype=float32)`

   `a.get_shape()`可以获得张量的维度信息

   关于生成计算图，我们可以使用：

   `G = tf.Graph()`

   以生成新的计算图，不同的计算图不仅可以隔离张量和计算，还能够对其进行管理。（`tf.Graph.device()`）可以用于管理设备。

   

2. 计算图里用张量搭建整个结构，然后当使用：

   `with tf.Session(graph = g1) as sess:`时（不用写close()，上下文退出时，自动关闭会话并释放资源）

   or `td.Session().run(res)`时

   会将数字带入并进行计算。

   完毕后要写`sess.close()`来关闭对话，释放资源。

   在默认会话中可以使用：

   ```python
   sess = tf.Session()
   with sess.as_default():
       print(res.eval())
   ```

   可以计算某一具体张量的值。

   或者：

   ```python
    sess = td.InteractiveSession()
    print(res.eval())
    sess.close
   ```

   也可以完成类似功能，

   `InteractiveSession()`会自动将生成的会话注册为默认会话。

   使用tf.ConfigProto可以配置会话：

   他可以配置类似并行的线程数，GPU分配策略，运算超时时间等参数。常用的有：

   allow_soft_placement and log_device_placement

   allow_soft_placement为True时可以将GPU运算放到CPU上：

   	1.	运算无法在GPU上执行
    	2.	没有GPU资源
    	3.	运算输入包含对GPU计算结果的引用。

   log_device_placement为True时，日志中将会记录每个节点被安排在哪个设备上以方便调试。



##### 神经网络

`tf.matmul(a,b)`实现了矩阵乘法的功能。

神经网络参数：

在tensorflow中，tf.Variable的作用就是保存和更新神经网络中的参数，他需要初始化：

`weights = tf.Variable(tf.random_normal([2,3],stddev=2))`

这个代码会产生一个2*3的矩阵，其中的元素是均值为0，标准差为2 的正态分布随机数，可用mean来设置均值。

总的有：

```python
tf.random_normal(size,mean,stddev,dtype,seed)
tf.truncated_normal(size,mean,stddev,dtype,seed)
tf.random_uniform(size,min,max,dtype,seed)
tf.random_gamma(size,alpha,beta,dtype,seed)
tf.zeros(size,dtype)
tf.ones(size,dtype)
tf.fill(size,number)
tf.constant(具体的tensor值)
```

在神经网络中，weights通常用随机数来初始化，而biase用常数来进行初始化。

在初始化时，用使用代码段 `sess.run(w.initializer)`来实现初始化。

`tf.global_variables_initializer()`可以初始化所有变量。

张量和变量的关系：

变量声明是一种运算，其输出结果就是一个张量，所以变量只是一种特殊的张量。



Tensorflow中有集合的概念，所有变量都会被自动加入到tf.GraphKeys.VARIABLES这个集合中，通过tf.global_variables()函数可以拿到当前计算图上所有的变量。

如果声明变量时，参数的trainable属性为True，那么这个变量将会被加入到GraphKeys.TRAINABLE_VARIABLES集合。在Tensorflow中可以通过tf.trainable_variables()活得所有需要优化的参数，Tensorflow中提供的神经网络优化算法会将该集合中的变量作为默认优化对象。

注意点：

进行张量计算时，两个属性必须关注：维度与数据类型。只有匹配才可进行张量运算。但是当

w1 为[2,2]，w2 为[2,3]时，

```python
w1.assign(w2) #报错
w1.assign(w2,validate_shape = False) #不报错
```



##### Tensorflow 训练神经网络

​	流程：初始化变量（initializer）-> 选区一部分训练数据 ->通过前向传播获得预测值 ->通过反向传播更新预测值--->...

​	我们通常把输入设为常数，即用tf.constant来进行设置，但是这样会导致每一次输入都要生成一个新的constant，这样就会生出一个新的节点，从而导致计算图非常大，利用率也低。所以我们使用tf.placeholder 和 free_dict来解决这个问题。

设置代码（前向传播样例）：

```python
import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2,3],stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([3,1],stddev = 1, seed = 1))

x = tf.placeholder(tf.float32, shape = (1,2), name = "input")
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

#print(sess.run(y))# 报错：InvalidArgumentError (see above for traceback): You must feed a value for placeholder tensor 'input' with dtype float and shape [1,2]

print(sess.run(y, feed_dict = {x: [[0.7,0.9]]}))
#output: [[3.957578]]
```

placeholder中设定的类型是不能更改的，维度信息可以通过推算得出。

placeholder的取值必须用feed_dict来赋予。

反向传播算法样例

```python
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))+(1-y_)*tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))# y_为label,clip_by_value 是把数控制在1e-10与1.0之间
learning_rate = 0.001
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
```

优化器有三种最常用的：

```python
tf.train.AdamOptimizer
tf.train.GradientDescentOptimizer
tf.train.MomentumOptimizer
```



#### 第四章 深层神经网络

##### 线性模型的局限性

线性模型只能解决线性可分问题，具有一定的局限性，使用Relu等激活子可以起到去线性化的作用。

常用activation：

```python
tf.nn.relu
tf.sigmoid
tf.tanh
```

单层网络没有办法解决异或问题，加入隐藏层后的多层网络可以。

##### 损失函数

监督学习分为：分类问题与回归问题。

###### 分类问题

cross entropy反应两个概率分布之间的距离，用于分类问题

多分类问题用softmax将输出变换为概率分布

`v1*v2` tensorflow中 * 表示元素直接相乘， `tf.matmul`才是矩阵乘法。

tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。

```python

reduce_mean(input_tensor,
                axis=None,
                keep_dims=False,
                name=None)
```

第一个参数input_tensor： 输入的待降维的tensor;
第二个参数axis： 指定的轴，如果不指定，则计算所有元素的均值;
第三个参数keep_dims：是否降维度，设置为True，输出的结果保持输入tensor的形状，设置为False，输出结果会降低维度;
第四个参数name： 操作的名称;



对取softmax并用cross entropy作为损失函数的操作，tensorflow进行了合并使用库函数：

`tf.nn.softmax_cross_entropy_with_logits( labels=y_ , logits=y )`

###### 回归问题

MSE使用较多

MSE的定义为 `mse = tf.reduce_mean(tf.square(y_ - y))`



##### 自定义损失函数

 `tf.greater(v1,v2): v1>v2->true`

`tf.where(tf.greater(v1,v2),v1,v2):v1>v2->v1`

设立好自己的loss后

在optimizer上使用minimize/maximize，example：

`train_step = tf.train.AdamOptimizer(0.001).minimize(loss)`

`sess.run(train_step)`



##### 优化算法

该书神经网络的训练框架遵循以下原则：

```python
import tensorflow as tf

batch_size = n
#每次读取一部分数据作为当前训练数据来执行反向传播算法
x = tf.placeholder(tf.float32, shape = (batch_size,2), name = "x-input")
y_ = tf.placeholder(tf.float32, shape = (batch_size,1), name = "y-input")

learning_rate = 0.001
loss = ...
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#参数初始化
sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
#迭代的更新参数
STEPS = 5000
for i in range(STEPS):
    start = (i * batch_size) % dataset_size
    end = min(start + batch_size, dataset_size)
	#get current_X, current_Y
    sess.run(train_step,feed_dict = {x: X[start:end], y_: Y[start:end]})
	
    if i % 1000 == 0 :
        total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y_:Y})

    
    print("After %d training step(s), cross entropy on all data is %g" %(i, total_cross_entropy))


 
```

###### Learning Rate

可以使用 `tf.train.exponential_decay`可以指数级地减小学习率。example：

```python
global_step = tf.Variable(0)
learing_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96,staircase = Ture)
learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
```

`tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=True/False)`

decay_steps：衰减速度

decay_rate: 衰减系数

###### 正则项

》在实践中，L1正则项和L2正则项可以同时使用，见P88

`tf.contrib.layers.l1_regularizer(.5)(weights)`0.5为正则化项的权重。

`tf.contrib.layers.l2_regularizer`

样例代码：

```python
import tensorflow as tf
def get_weights(shape, lamb) :
    var = tf.Variable(tf.random_normal(shape), dtype = tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lamb)(var))
    return var

x = tf.placeholder(tf.float32, shape = (None, 2))
y_ = tf.placeholder(tf.float32, shape = (None, 1))

batch_size = 8
layer_dimension = [2, 10, 10, 10, 1]
n_layers = len(layer_dimension)
cur_layer = x
in_dimension = layer_dimension[0]

for i in range(1, n_layers):
    out_dimension = layer_dimension[i]
    weights = get_weights([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape = [out_dimension]))
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weights) + bias)
    in_dimension = layer_dimension[i]

mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

tf.add_to_collection('losses', mse_loss)

loss = tf.add_n(tf.get_collection('losses'))
```

###### 滑动平均模型

使模型在测试数据上更健壮的方法，滑动平均模型。

对每一个变量会维护一个shadow variable

shadow_variable = decay*shadow_variable + (1-decay) * variable

decay 越大模型越趋于稳定，通常设为0.999或者0.9999。

tf提供了 `tf.train.ExpoentialMovingAverage`来实现滑动平均模型。

衰减率 = min{decay, (1+num_updates)/(10+num_updates)}

样例代码：

```python
import tensorflow as tf

v1 = tf.Variable(0, dtype = tf.float32)

step = tf.Variable(0, trainable = False)

ema = tf.train.ExponentialMovingAverage(0.99, step)

maintain_average_op = ema.apply([v1])

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run([v1, ema.average(v1)]))
	#初始化均为零
    sess.run(tf.assign(v1, 5))
    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))
	#第一次，v1更新至5，decay= 1+0/10+0=0.1，小于0.99， v1被更新为4.5
    sess.run(tf.assign(step, 1000))
	#step = 1000 -》 decay = main（0.99，0.999）=0.99
    sess.run(tf.assign(v1,10))

    sess.run(maintain_average_op)
	
    print(sess.run([v1,ema.average(v1)]))
	#v1更新至10，shadow更新至4.555
    sess.run(maintain_average_op)

    print(sess.run([v1,ema.average(v1)]))
```

输出：

```
[0.0, 0.0]
[5.0, 4.5]
[10.0, 4.555]
[10.0, 4.60945]
```

