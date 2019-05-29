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

