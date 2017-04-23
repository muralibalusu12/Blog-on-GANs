### Project 1: Blog Series on Generative Adversarial Networks
You can see it here: [https://bmurali1994.github.io/Blog-on-GANs/](https://bmurali1994.github.io/Blog-on-GANs/)
```
Name: Murali Raghu Babu Balusu
GaTech id: 903241955
GaTech username: mbalusu3
Email: b.murali@gatech.edu (or) muraliraghubabu1994@gmail.com
Phone: 470-338-1473.
```
- In machine learning and statistics, [Classification](https://en.wikipedia.org/wiki/Classification) is the problem of identifying to which of a set of categories (sub-populations) a new observation belongs, on the basis of a training set of data containing observations (or instances) whose category membership is known.

### Generative vs. Discriminative Models
Classification can be done learning the underlying distribution in two different ways:-
1. Generative Models: Generative Models learn a model of the joint probability distribution, p(x,y) of the inputs x and the label y and then make predictions using the Bayes rule to calculate ```p(y|x)``` and then picking the label with the highest likelihood. 
2. Discriminative Models: Discriminative Models directly learn the posterior distribution ```p(y|x)```, a mapping from the input x to the likely label y.


Here's a really simple example. Suppose you have the following data in the form (x,y):

(1,0), (1,0), (2,0), (2, 1)


Now, calculating p(x,y) would be as follows:-
```
      y=0   y=1
     -----------
x=1 | 1/2   0
x=2 | 1/4   1/4
```

Similarly, calculating ```p(y|x)``` from the data would be as follows:- 

```
      y=0   y=1
     -----------
x=1 | 1     0
x=2 | 1/2   1/2
```
Now, observe these two distributions and you will understand the difference as to how they are calculated.
```
- Generative algorithms model p(x,y) by looking at the whole data and then transform that 
into the probability P(y|x) by applying Bayes rule and then used for classification tasks. 

- Discriminative algorithms directly learn the distribution p(y|x) and use this to 
classify a given input x into a class y.
```

### Advantages of Generative Models
However, there are several advantages to learning more about generative modeling:
- They better understand the underlying distribution and will be able to understand the data even when there are no labels.
- They are an excellent test of our ability to use high-dimensional, complicated probability distributions
- They also have extremely important applications in Re-inforcement learning. 
- Missing data: Generative models can be trained with missing data and can provide predictions on inputs that are missing data, like Semi-supervised learning when many labels are actually missing for the data.
- Realistic generation tasks: Since the generative model learn the distribution p(x,y), then I can use the same distribution to randomly sample pairs and generate the likely (x,y) data points from my distribution. This is the main motivation behind the entire theory of GANs.

### History of Generative Models
The image below shows a hirearchical structure of various generative models.
![Image](https://bmurali1994.github.io/Blog-on-GANs/images/s6.png) 

Deep generative models that can learn via the principle of maximum likelihood differ with respect to how they represent or approximate the likelihood. On the left branch of this taxonomic tree, models construct an explicit density, pmodel(x; θ), and thus an explicit likelihood which can be maximized. Among these explicit density models, the density may be computationally tractable, or it may be intractable, meaning that to maximize the likelihood it is necessary to make either variational approximations or Monte Carlo approximations (or both). 

On the right branch of the tree, the model does not explicitly represent a probability distribution over the space where the data lies. Instead, the model provides some way of interacting less directly with this probability distribution. Typically the indirect means of interacting with the probability distribution is the ability to draw samples from it. Some of these implicit models that offer the ability to sample from the distribution do so using a Markov Chain; the model defines a way to stochastically transform an existing sample in order to obtain another sample from the same distribution. Others are able to generate a sample in a single step, starting without any input. 

While the models used for GANs can sometimes be constructed to define an explicit density, the training algorithm for GANs makes use only of the model’s ability to generate samples. GANs are thus trained using the strategy from the rightmost leaf of the tree: using an implicit model that samples directly from the distribution represented by the model. 

There are two basic things we can do with generative models:
- Density Estimation and 
- Sample Generation
GANs fall in the Sample Generation set and they give interesting results when used to generate images.

##### The intuitive idea behind GANs is that in order to be able to better generate the data which is quite similar to the training data, GANs would be learning a better latent representation of the images and this could be extremely useful for computer vision research.

### Deep Learning techniques
So, now Deep Learning techniques have been extensively used in building discriminative models typically used for classification or recognition. Various Deep Learning models with different architectures, loss function, learning techniques etc.. have proved to be extremley powerful in discovering rich models that present the posterior probability distributions over different kinds of data encountered in the real world like images in computer vision, audio in speech processing, text in natural language processing. 

Now, the next question is: Can we use deep learning techniques to learn generative models? If so, how? 
So, if this is done? Can we generate realistic data using that genrative deep learning models? Since, we have seen plenty of application of deep learning in vision, can we generate realistic images?

The answer to all these fascinating questions is YES, we can by using Generative Adverserial Neural Networks (GANs).

### Well, Initially let's look at some amazing results by GANs...
(Images Source: [“Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks”]( https://arxiv.org/abs/1511.06434v2))

The image below shows the images of bedrooms generated by GANs.
![Image](https://bmurali1994.github.io/Blog-on-GANs/images/s1.png) 

The image above show a set of human faces generated by Generative Adverserial Neural Networks.

![Image](https://bmurali1994.github.io/Blog-on-GANs/images/s2.png) 

### Introduction to GANs
I have explained the basic idea behind generative models above, however GANs work in a slightly different way.

#### Generative Adverserial Networks: 
Generative Adversarial Networks (GANs) are a class of neural networks which have gained popularity in the past few years. As explained above, they allow a neural network model to generate data with the same structure as the training data. GANs are extremely powerful and flexible tools. A GAN model can learn to generate pictures of digits, various scenes likes bedrooms, various animals like cats and dogs etc, if given enough training data. Furthermore, it is unsuprvised learning since we do not need any labels about the data. Recently, GANs have also been shown to generate videos as well. 

The basic idea of Generative Adverserial Netural Networks is formulated as a game between two players. One of the player is called the Generator and the other player is called the Discriminator. 
The generator creates samples that are intended to be more realistic and to be from the same distribution as that of the training data. The discriminator examines the given samples and determines if they are fake or real.

So, basically the generator is trying to fool the discrimnator here by generating plausible examples that are likely to be thought as real by the discriminator. The discriminator can be seen as a traditional supervised machine learning techniques and it is being trained to classify the given input sample into two classes (real or fake). The generator is trained to fool the discriminator or in other words, the generator is trained to create samples that are likely to be similar to the ones the drawn from the same distribution as the training data.

see the figure below:
![Image](https://bmurali1994.github.io/Blog-on-GANs/images/gan.png) 

- We can let the entire GAN model learn by itself, by using two networks that one tries to fool the other, and the other one tries not to be fooled by the first one. 

#### Why this works? 
Let's use image generation as an example. Intuitively, whenever the first network(the "generator") produces some fake images that can fool the second network(the "discriminator") to think they are real, the discriminator will receive a loss for not being able to distinguish the fake images from the real images, and update itself to better distinguish fake images next time. And for the generator, if it failed to fool the discriminator, it also received a loss and will update itself to fool the discriminator better. By doing this iteratively, finally we will get a generator that can produce images that are really close to the actual images in the training set.

#### Do we need any labels?
One of the largest advantage of GANs is that it is unsupervised learning. This is to say, we do not need to manually label the image from the generator to be fake/real to make it learn. We just give the network a bunch of real images, and it will learn the rest by itself. It really makes training large-scale generative models possible.

### Let's get Mathematical:

Now, we need to get to the more formal version of this in order to be able to implement this:

GANs are a structured probabilistic model containing latent variables ```z``` and observed variables ```x```. As mentioned before, GANs is setup as a game between two players: Generator and Discriminator. 
These two players are actually two mathematical functions denoted by ```G``` and ```D``` for Generator and Discrimnator respectively. These functions are to be differentiable with respect to its inputs and parameters so that the parameters can be learned. The discriminator function D takes in ```x``` as input and uses θ_d as parameters. The generator function ```G``` takes in ```z``` as input and uses ```θ_g``` as parameters.

Both the generator (G) and the discriminator (D) are neural networks which play a min-max game between one another. 
The generator takes in random noise as input (z) and transforms it into the form of data that we are interested in generating. The discriminator then takes in a set of data, either real (x) or generated (G(z)) and produces a probability of that data being real and fake.

Observe the figure below.
![Image](https://bmurali1994.github.io/Blog-on-GANs/images/GANmodel.png) 

### So to sum it up
```
- GANs is set up as a mini-max game between the generator and the discriminator
- The generator is a neural network learning to transform z (input is noise) and generate 
realistic data (G(z)) similar to the data drawn from the training data's distribution. 
- The discriminator is also a neural network learning to determine if the given input 
(x or G(z)) is real or fake.
```

### Cost functions
Now, both the players generator G and discriminator D have cost functions that are defined in terms of both the players paramaters. 
The discriminator wishes to minimze the cost function ```Jd(θ_d,θ_g)``` and does so by only changing its own ```θ_d``` parameters. Similarly, The generator wishes to minimize the cost function ```Jg(θ_d,θ_g)``` and does this by changing its own parameters ```θ_g```.
Clearly, each player cannot control the other players parameters, therefore both of them are trying to get better at the game. This scenario is straight forward to describe this game as an optimization problem. The cost functions for generator and the discriminator are shown below. These are the formulas extracted from the paper on GANs.
![Image](https://bmurali1994.github.io/Blog-on-GANs/images/s3.png)

![Image](https://bmurali1994.github.io/Blog-on-GANs/images/s4.png)

![Image](https://bmurali1994.github.io/Blog-on-GANs/images/s5.png)

These two neural networks are trained to optimize the functions given above. So, in order to train the parameters ```θ_d``` and ```θ_g```, you would use the following formulas.
### Training the Discriminator

The following image shows the functions that is maximized for the discriminator:
![Image](https://bmurali1994.github.io/Blog-on-GANs/images/disc.png) 
Observe the function carefully as I explain each term below:
```
- m is just the number of training samples in that batch.
- D(x) denotes the probability given by the discriminator D that the input x is real.
- log(D(x)) We take a log of that probability here, because probabilities are typically very 
very small and hence very difficult to observe differences in inputs and for the parameters to 
learn through back propogation.
- z is the random noise given to the generator G.
- G(z) is the output of the generator G, i.e. it is a generated input
- D(G(z)) indicates the probability that the discriminator D think that the generated input 
G(z) is real
- So, 1 - D(G(z)) indicates the probability that the discriminator D thinks that the 
generated input G(Z) is fake.
- log(1-D(G(z)) indicates the log of the probability that the discriminator D thinks that the 
generated input G(Z) is fake.
```
Clearly, according to our set up of min-max game, the discriminator wants both of these log probabilites to be as high as possible.

The discriminator D is trained to MAXIMIZE this function: i.e. to increase the likelihood of giving a high probability to the real data and a low probability to the generated data. (Gradient Ascent technique is used, since we want to maximize the value)


### Training the Generator

The following image shows the function that is minimized for the generator:
![Image](https://bmurali1994.github.io/Blog-on-GANs/images/gene.png) 

Observe the function carefully, 
```
- z is the random noise given to the generator G.
- G(z) is the output of the generator G, i.e. it is a generated input
- D(G(z)) indicates the probability that the discriminator D think that the generated input 
G(z) is real
- So, 1 - D(G(z)) indicates the probability that the discriminator D thinks that the 
generated input G(Z) is fake.
- log(1-D(G(z)) indicates the log of that probability that the discriminator D thinks that 
the generated input G(z) is fake.
```
Clearly, the generator want this to be as low as possible. The Generator minimizes the log-probability of the discriminator being correct. Thus, the generator is trained to MINIMIZE this function. Thus, the generator is then optimized in order to increase the probability of the generated data being rated real highly by the discriminator. (Gradient descent technique is used, since we want to minimize the value)


### Training Generator and Discriminator
We train these two neural networks Generator G and Discriminator D by using graident optimization between these two neural networks using the above two functions on batches of real and generated data for several epochs. 

We expect that over time, the discriminator becomes better at identifying the real and fake data input samples. And that the generator becomes good at generating realistic like data from random noise and fool the discriminator.
![Image](https://bmurali1994.github.io/Blog-on-GANs/images/example.png)

### Application of GANs to problems

Now, we shall apply GAN to a few interesting problems and analyse their results. The code is present in the code folder and the instructions to run it are present in the readme.md file.

### Experiment 1: Approximating a unimodal 1d Gaussian
We'll train a GAN to solve this toy problem - on learning to approximate a unimodal 1d Gaussian distribution. The code is available in the source code folder shared along with relevant comments and the result plots. The code is easily understandable, with the comments in it. Here, I will explain few of the interesting parts of the code.

Below, we create the real data distribution below:
```
class DataDistribution(object):
    def __init__(self):
        self.mu = 4
        self.sigma = 0.4

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples
```
Now, we create the random gaussian noise through [stratified sampling](https://en.wikipedia.org/wiki/Stratified_sampling) to be sent as input to the generator:
```
class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + \
            np.random.random(N) * 0.01
```

Now, we create the neural network architecture for the generator to generate the realistic like data from the gaussian noise. It is basically a two layer neural architecture as shown below:
```
def generator(input, hidden_size):
    h0 = tf.nn.softplus(linear(input, hidden_size, 'g0'))
    h1 = linear(h0, 1, 'g2')
    return h1
```
Below, I mentioned the neural network architecture for the discriminator in order to detect if the input samples are fake or real.
```
def discriminator(input, hidden_size):
    h0 = tf.tanh(linear(input, hidden_size * 2, 'd0'))
    h1 = tf.tanh(linear(h0, hidden_size * 2, 'd1'))
    h2 = tf.tanh(linear(h1, hidden_size * 2, 'd2'))
    h3 = tf.sigmoid(linear(h2, 1, 'd3'))
    return h3
```
The model is trained as shown below using a similar code. We randomly sample an input from the training data. Then, we send in some random noise to the generator and take the generated input and send both the original input from the training data and the generated input from the generator to the discriminator and then based on its outputs, we update the parameters of both the generator and the discriminator according to the cost functions described above. 
```
with tf.Session() as session:
    tf.initialize_all_variables().run()

    for step in xrange(num_steps):
        # update discriminator
        x = data.sample(batch_size)
        z = gen.sample(batch_size)
        session.run([loss_d, opt_d], {
            x: np.reshape(x, (batch_size, 1)),
            z: np.reshape(z, (batch_size, 1))
        })

        # update generator
        z = gen.sample(batch_size)
        session.run([loss_g, opt_g], {
            z: np.reshape(z, (batch_size, 1))
        })
```
So, these are the main components in describing the GAN framework. Apart from the above architecture, I used the following details for training the GAN on the 1d Gaussian:
```
The initial learning rate is set to be 0.005. This learning rate is decayed by 95% 
every 150 steps.
initial_learning_rate = 0.005
decay = 0.95
num_decay_steps = 150
I trained the model for 1500 iterations.
```
Look at the result below:
![Image](https://bmurali1994.github.io/Blog-on-GANs/images/plot1.png)
Clearly, the result looks good. The generator is able to generate realstic data similar to the data from the original training data distribution. Intuitivly as well, this makes sense. Observe that the discriminator always looks at a sample input and tries to detect if its real or fake. Clearly, if the generator keeps generating inputs close to the mean of the training data distribution, then the discriminator will be likely to get fooled and that is what happens above when you are training this.

#### Improving Sample diversity
One of the main problems of GANs is the model collapse. The model at times gets stuck at certain parameter values and it is extremely hard for the model to improve from then on... Here, observe that the generator generates inputs which have a very narrow distribution compared to that of the training data distribution. This is due to the model collapse issue. There have been many techniques or tips and hacks researched recently and one of them is using mini batch discrimination. In this case, we basically send in multiple samples to the discriminator so that it can make a better judgement of the data points being real or fake, hence making the discriminator powerful. Hence, ultimately we would hope that the generator would be able to generate better points as well.

Using this technique, we get better results as shown below.
![Image](https://bmurali1994.github.io/Blog-on-GANs/images/plot2.png)

### Experiment 2: Changing the unimodal distribution mean and variances
I have tried varying mean and variance of the unimodal distribution and here are the important conclusions.
1. Using the exact same architecture did not work that well even on changing the mean. I had to add in some extra power to the generator through another linear layer to make sure that the results are good enough.
2. Using the exact same architecture worked well for the unimodal distribution on changing the variances, except that it took a bit longer to converge for too small or too large variances. Thus, it was taking more no iterations to converge as I increased the variance to a high value like 0.9 or decreased it to a low value like 0.1
3. Here are some interesting plots

![Image](https://bmurali1994.github.io/Blog-on-GANs/images/plot7.png)

### Experiment 3: Approximating a Bimodal distribution
I have experimented with different generator and discriminator architectures, by adding in more layers, changing the activation functions etc.. to train a GAN to learn a bimodal distribution and here are a few ```INTERESTING OBSERVATIONS``` that I found.
1. Activation Function:- Using relu layers in the generator did not help initially for a smaller neural network architecture of the GAN. However, on increasing the number of hidden layers, relu layer definitely helped improve the performance over softplus and tanh.
2. No of hidden layers: Increasing the hidden layers in the generator increased the complexity of the model and I did not observe any significant improvement in the performance after adding a few layers. But, as I increased the hidden layers, the GAN model took longer time to converge. I did not want the model to overfit on the data, hence I also tried making the discriminator powerful by adding in more layers. This certainly helped improve the performance. However, intuitively, we always need to make sure that our discriminator is more powerful, then this helps in the generator also learning to be better.
3. No of hidden units: Increasing the no. of hidden units in the initial hidden layers improved the performance of the GAN. However, increasing it too much beyond a limit deteriorated the performance of the GAN
4. No of iterations: Initially, I found the GAN model to be underfitting for a smaller number of iterations, but on increasing the no of iterations I felt that the GAN model was oscillating between few parameters and the results were almost similar.
Here are a few interesting plots from the experiments.

![Image](https://bmurali1994.github.io/Blog-on-GANs/images/plot8.png)

### Experiment 4: Approximating a Trimodal distribution
I have tried approximating a trimodal distribution as well, but I could not obtain any good results even after experiments with various hidden layer architectures for both generator and the discriminator, I have also tested relu, tanh and softplus activation function for the generator.
Here are a few results, I obtained for the trimodal distributions. Yes, they are not very good.

![Image](https://bmurali1994.github.io/Blog-on-GANs/images/plot9.png)

### Experiment 5: Approximating a Poisson distribution
Poisson distribution in general is quite tough to approximate using GANs using smaller no of training samples. I am sure, increasing the number of training samples and using that with minibatch discrimination would give better results.
Observe the plot below obtained to approximate a poisson distribution. I have tried varying the architectures but could not get good results for the poisson distribution. But, observe the decision function, it aligns quite well with the poisson distribution graph.

![Image](https://bmurali1994.github.io/Blog-on-GANs/images/plot10.png)

### Problems with GANs
GANs are powerful generative models, but they are not easy to train. 
There are a lot of limitations like the discriminator could not be trained too well, hence another paper on [DCGAN](https://arxiv.org/pdf/1511.06434.pdf) was proposed. DCGAN are a variant of GANS with some tricks about the architecture to be used, the loss functions and the pooling or activation functions to be used so that the model remains stable. So, yes GANs are extremely difficult to train.  
A lot of such engineering hacks has to be done in order to train a good GAN, and they are discussed in detail in this paper: [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf).
The instability in GANs is mainly due to the loss function. Here is a very simple version of what is going on: because the loss function can be interpolated as JS divergence (and plus a KL divergence term for the improved version), during training there will be a large chance that the JS divergence will give out close to zero gradient, and that is essentially a disaster for our gradient descent algorithm. And also because the KL divergence is a non-symmetric measure, it leads to "mode collapse", which is another big problem of GANs.

### DCGANs
The following is an image of DCGANs from their paper.

![Image](https://bmurali1994.github.io/Blog-on-GANs/images/example.png)

DCGANs are very famous and a widely used architecture and stable version of GANs. I used DCGANs for the experiments with the images below. Here are the main important ideas in the architecture of DCGANs and the tips and tricks used for stabilised version of training.
1. Use a Convolutional Neural Network instead of plain feedforward neural networks and Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator): This lets the discriminator in GAN model to learn its own spatial downsampling and the generator to learn its own spatial upsampling. 
2. Avoid using fully connected hidden layers for deeper architectures of the generator and discriminator in the GAN as it was  shown to hurt the convergence of the GAN model. 
3. Using Batch Normalization in both the generator and the discriminator stabilizes learning by normalizing the input to each unit having zero mean and unit variance. This clearly eliminates problems that arise with poor initialization or with poor gradient flow in deeper architectures. This helps in preventing the generating from collapsing all the generated points to a signle point in space. This is typically a common problem with GANs. 
4. Using ReLU activation function for the generator and tanh for the output layer fo the generator and LeakyRelU for the inner layers of the discriminator and sigmoid for the output layer of the discriminator would give better results for DCGAN in the experiments. The authors mention to have observed that using a bounded activation allowed the model to learn more quickly to saturate and cover the color space of the training distribution. 

#### More tips and Tricks:
- Use a one sided label smoothing cost for the discriminator meaning: Typically labels for the discriminator are 0 fake data and 1 for real data. Now, Label smoothing is a technique from the 1980s which smooths the targets from 0 and 1 to 0.1 and 0.9 respectively. One useful trick is to smooth the label only on the real side, i.e. smooth the labels for the discriminator to be 0 and 0.9. Do Not smooth the labels for the fake data, because it might enfore the same generator behavior again. The advantages of using this trick are: It prevents discriminator from giving very large gradient signal to generator and is a good regularizer
- Use Virtual Batch normalization in which each example x is normalized based on the statistics collected on a reference batch of examples that are chosen once and fixed at the start of training, and on x itself. The reference batch is normalized using only its own statistics. Batch normalization greatly improves optimization of neural networks, and was shown to be highly effective for DCGANs. However, it causes the output of a neural network for an input example to be highly dependent on several other inputs in the same minibatch. To avoid this problem, VBN is introduced. VBN is computationally expensive because it requires running forward propagation on two minibatches of data, so we use it only in the generator network.
- Typically the discriminator D is more deeper and powerful than the Generator G. This in practice has shown to be better at image tasks. 
- Run the discriminator more often than the the generator. This helps the discriminator to be strong and to avoid the generator from learning something wrong. Then as we train for longer duration, we ultimately hope that the generator becomes good at generating better samples.
- Usually the discriminator's loss is smaller than the generator's loss: meaning, that the discriminator does a better job.



### Experiment 6: MNIST dataset
Now, it would be cool to experiment with images datasets like MNIST using DCGANs architectures and their tips and tricks.
The goal is to train the GAN model using the MNIST digits and to generate MNIST digits using GAN. I have shared the DCGAN source code used and the instructions to run it in the readme file.
Here are the architecture details used for training and testing DCGANs:
- 60,000 MNIST images for training, 10,000 images for testing
- Trained for 25 epochs
- Used learning_rate=0.0002 for Adam optimizer for stochastic optimization and momentum parameter=0.5
- Generator: 4 hidden de-convolutional layers(5x5 kernels) with relu as activation function and tanh as an activation function for the output layer
- Discriminator: 4 hidden convolutional layers(4x4 kernels) , 1 hidden fully connected layers, leaky relu as the activation function for all the inner layers and sigmoid activation for the output layer.
- Trained it for about 12 hours on my laptop.
- I have also sent a few checkpoint models to test the trained models.
- I have also attached the source code and instructions to run it.

Attached below are some interesting good results:

![Image](https://bmurali1994.github.io/Blog-on-GANs/images/mnist2.png) ![Image](https://bmurali1994.github.io/Blog-on-GANs/images/mnist3.png)

### Experiment 7: CelebA Faces dataset
I have also tested the same model with the same DCGAN architecture and parameters mentioned above on the celebA faces dataset. The goal here is to generate faces using a GAN model trained on the celebA dataset.

Here are some results obtained using the DCGAN model:

![Image](https://bmurali1994.github.io/Blog-on-GANs/images/celeb1.png)

![Image](https://bmurali1994.github.io/Blog-on-GANs/images/celeb2.png)

### Experiment 8: Implemented my own GAN architecture and tested it on MNIST images
The goal is to train a GAN model using MNIST images and learn to generate them. 
Now, that we have seen other working implementations of GANs, I went ahead and implemented my own GAN. You can find the source code sent in the code folder.
I have used the same DCGAN architecture, but coded it myself referring to other useful blogs. I have trained the model for about 20,000 iterations, sending in about 128 images each batch, so that rounds upto 46 epochs. I used 4 deconvolutional layers (5x5 kernel size throughout), batchnorm and relu for all the hidden layers and tanh for the output layers. However, I have changed the architecture slightly for the discriminator used 3 convolutional layers(4x4 filters throughout) and 1 fully connected at the end. I tried to plot the generated images after every iteration, so as to observe the progress over time.

#### Generator:
```
def generator(z):
    zP = slim.fully_connected(z,4*4*256,normalizer_fn=slim.batch_norm,\
                              activation_fn=tf.nn.relu,scope='g_project',weights_initializer=initializer)
    zCon = tf.reshape(zP,[-1,4,4,256])
    gen1 = slim.convolution2d_transpose(zCon,num_outputs=64,kernel_size=[5,5],stride=[2,2],\
                                        padding="SAME",normalizer_fn=slim.batch_norm,\
                                        activation_fn=tf.nn.relu,scope='g_conv1', weights_initializer=initializer)
    gen2 = slim.convolution2d_transpose(gen1,num_outputs=32,kernel_size=[5,5],stride=[2,2],\
                                        padding="SAME",normalizer_fn=slim.batch_norm,\
                                        activation_fn=tf.nn.relu,scope='g_conv2', weights_initializer=initializer)
    gen3 = slim.convolution2d_transpose(gen2,num_outputs=16,kernel_size=[5,5],stride=[2,2],\
                                        padding="SAME",normalizer_fn=slim.batch_norm,\
                                        activation_fn=tf.nn.relu,scope='g_conv3', weights_initializer=initializer)
    g_out = slim.convolution2d_transpose(gen3,num_outputs=1,kernel_size=[32,32],padding="SAME",
                                         biases_initializer=None,activation_fn=tf.nn.tanh,\
                                         scope='g_out', weights_initializer=initializer)
    return g_out
# The generator has 4 convolutional layers as described in the DCGAN paper, 
# we also perform batch normalization for each layer and use ReLU as the activation function for hidden layer
# and tanh for the output layer.
```

#### Discriminator:
```
def discriminator(bottom, reuse=False):
    dis1 = slim.convolution2d(bottom,16,[4,4],stride=[2,2],padding="SAME",\
                              biases_initializer=None,activation_fn=lrelu,\
                              reuse=reuse,scope='d_conv1',weights_initializer=initializer)
    dis2 = slim.convolution2d(dis1,32,[4,4],stride=[2,2],padding="SAME",\
                              normalizer_fn=slim.batch_norm,activation_fn=lrelu,\
                              reuse=reuse,scope='d_conv2', weights_initializer=initializer)
    dis3 = slim.convolution2d(dis2,64,[4,4],stride=[2,2],padding="SAME",\
                              normalizer_fn=slim.batch_norm,activation_fn=lrelu,\
                              reuse=reuse,scope='d_conv3',weights_initializer=initializer)
    d_out = slim.fully_connected(slim.flatten(dis3),1,activation_fn=tf.nn.sigmoid,\
                                 reuse=reuse,scope='d_out', weights_initializer=initializer)
    return d_out
# The discriminator has 4 hidden layers, 3 convolutional and 1 fully connected, 
# we perform batchnorm and use leaky RelU for all the hidden layers
# and use sigmoid for the output layer
```

#### Connecting them on the graph:
```
tf.reset_default_graph()

z_size = 100 # input size of noise for the generator 

#initializes all the weights of the network.
initializer = tf.truncated_normal_initializer(stddev=0.02)

#placeholders are used for input into the generator and discriminator, respectively.
z_in = tf.placeholder(shape=[None,z_size],dtype=tf.float32) #Random vector
real_in = tf.placeholder(shape=[None,32,32,1],dtype=tf.float32) #Real images

Gz = generator(z_in) #Generator generates images from random noise z vectors
Dx = discriminator(real_in) #Discriminator determines the probability for the real data
Dg = discriminator(Gz,reuse=True) #Discriminator determines the probability for the generated data

# Defining the Cost functions for both the Generator and the Giscriminator.
d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1.-Dg)) #This optimizes the discriminator.
g_loss = -tf.reduce_mean(tf.log(Dg)) #This optimizes the generator.

tvars = tf.trainable_variables()

#We apply Adam optimizer for the gradient descent to update the parameters in the GAN.
trainerD = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)
trainerG = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)
d_grads = trainerD.compute_gradients(d_loss,tvars[9:]) #calculating the gradients for the weights of discriminator
g_grads = trainerG.compute_gradients(g_loss,tvars[0:9]) #calculating the gradients for the weights of generator

update_D = trainerD.apply_gradients(d_grads) #Updating the weights for the discriminator network.
update_G = trainerG.apply_gradients(g_grads) #Updating the weights for the generator network.
```

#### Training the model
```
batch_size = 128 
# Size of image batch to apply at each iteration.
iterations = 20000   
# Total number of iterations to use.
sample_directory = './figs'  
# Directory to save sample images from generator in.
model_directory = './models' 
# Directory to save trained model to.

init = tf.initialize_all_variables()
saver = tf.train.Saver()
with tf.Session() as sess:  
    sess.run(init)
    for i in range(iterations):
        zs = np.random.uniform(-1.0,1.0,size=[batch_size,z_size]).astype(np.float32) 
        # Generate a random z batch
        xs,_ = mnist.train.next_batch(batch_size) 
        # Draw a sample batch from MNIST dataset.
        xs = (np.reshape(xs,[batch_size,28,28,1]) - 0.5) * 2.0 
        # Transform it to be between -1 and 1
        xs = np.lib.pad(xs, ((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=(-1, -1)) 
        # Pad the images so the are 32x32
        _,dLoss = sess.run([update_D,d_loss],feed_dict={z_in:zs,real_in:xs}) 
        # Update the discriminator
        _,gLoss = sess.run([update_G,g_loss],feed_dict={z_in:zs}) 
        # Update the generator, twice for good measure.
        _,gLoss = sess.run([update_G,g_loss],feed_dict={z_in:zs})
        
        if i % 1000 == 0:
            print "Gen Loss: " + str(gLoss) + " Disc Loss: " + str(dLoss)
            z2 = np.random.uniform(-1.0,1.0,size=[batch_size,z_size]).astype(np.float32) # Generate another z batch
            newZ = sess.run(Gz,feed_dict={z_in:z2}) 
            # Use new z to get sample images from generator.
            if not os.path.exists(sample_directory):
                os.makedirs(sample_directory)
            # Saving sample generator images for viewing training progress.
            save_images(np.reshape(newZ[0:36],[36,32,32]),[6,6],sample_directory+'/fig'+str(i)+'.png')
        if i % 1000 == 0 and i != 0:
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)
            saver.save(sess,model_directory+'/model-'+str(i)+'.cptk')
            print "Saved Model"
```


Observe the images below. They clearly show how the GAN model is learning over iterations. Also, observing the loss, we get that generator loss decreases at first and then later on.. at times keep increasing and decreases later on frquently throughout the training and the discriminator loss as well keeps fluctuating slightly. 
  
![Image](https://bmurali1994.github.io/Blog-on-GANs/images/plot11.png)
![Image](https://bmurali1994.github.io/Blog-on-GANs/images/plot12.png)

### Experiment 9: Conditional GANs
Very Recently Conditional Generative Models have been proposed. You can condition GANs on text to generate content described from the text and so on. In the paper, they send in gaussian noise and also an embedding of the text and train the GAN model to obtain specific images related to the context described in the text. More details can be obtained in the paper: [Generative Adverserial Text to Image synthesis ](https://arxiv.org/abs/1605.05396).
I have run their [code given here](https://github.com/reedscot/icml2016) and the results obtained are mentioned below.

Text: this flower has white petals and a yellow stamen

Generated Image:
![Image](https://bmurali1994.github.io/Blog-on-GANs/images/plot13.png)

Text: the center is yellow surrounded by wavy dark purple petals

Generated Image:
![Image](https://bmurali1994.github.io/Blog-on-GANs/images/plot14.png)

Text: the flower has red and white petals with yellow stamen

Generated Image: 
![Image](https://bmurali1994.github.io/Blog-on-GANs/images/plot15.png)


### Conclusion and Future Directions
- GANs are generative models that use supervised learning to approximate an intractable cost function. They can be used to simulate various cost function including maximum likelihood. 
- Training GANs is extremely tough and heavily dependent on the loss function. A lot of such engineering hacks has to be done in order to train a good GAN. Tips and Tricks are discussed in detail above. Analysis of adding layers or changing activation functions is also described above in the experiments.
- It is extremely interesting to see the wonderful results and applications of GANs and this is very motivating. 
- Future directions of research would be to find out more about the problems of GANs and several ways in order to train a stable model of GAN. Also to explore more about Conditional Generative Adverserial Neural Networks.
- Another important direction of research is to train generative models to understand motion and activities in the videos. This is right now a heavily pursued research area.
- In general this way of Adverserial training through a mini-max game could be used in several other applications of machine learning to learn better models for various tasks. This is an exciting direction to explore as well.


I have explained everthing that I did above and the code has been provided in the code folder along with the instructions to run in the readme file. Below is the various list of references that I referred to throughout while writing the blog.

### References
1. [Generative Adverserial Nets](https://arxiv.org/pdf/1406.2661.pdf)
2. [Unsupervised representation learning with deep convolutional generative adverserial networks](https://arxiv.org/pdf/1511.06434.pdf)
3. [Generative Adverserial Text to Image synthesis ](https://arxiv.org/abs/1605.05396)
4. [Generative Adversarial Networks (GANs) NIPS 2016 tutorial ](http://www.iangoodfellow.com/slides/2016-12-04-NIPS.pdf)
5. [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf)
6. [http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/)
7. [https://medium.com/@awjuliani/generative-adversarial-networks-explained-with-a-classic-spongebob-squarepants-episode-54deab2fce39](https://medium.com/@awjuliani/generative-adversarial-networks-explained-with-a-classic-spongebob-squarepants-episode-54deab2fce39)
8. [https://github.com/carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)
9. [http://blog.evjang.com/2016/06/generative-adversarial-nets-in.html](http://blog.evjang.com/2016/06/generative-adversarial-nets-in.html)
10. [https://github.com/reedscot/icml2016](https://github.com/reedscot/icml2016)

#### That's it from me!
Please feel free to contact me if you have any further questions. Thank you!
```
Name: Murali Raghu Babu Balusu
GaTech id: 903241955
GaTech username: mbalusu3
Email: b.murali@gatech.edu (or) muraliraghubabu1994@gmail.com
Phone: 470-338-1473.
```
