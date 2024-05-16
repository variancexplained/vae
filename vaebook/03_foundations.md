# Variational Autoencoders: Mathematical Framework
Having understood the high-level intuition of how VAEs work, we now delve into the underlying principles and mathematical framework that make VAEs effective. This includes exploring probabilistic models, conditional models, parameterizing distributions with neural networks, and the role of the evidence lower bound (ELBO) in optimizing VAEs. By connecting these elements, we can see how VAEs transform complex data into meaningful representations and back.

In this section, we explore the core principles underlying Variational Autoencoders (VAEs), which combine probabilistic modeling with neural networks to learn complex data distributions efficiently.

* **Probabilistic Models**
* **Conditional Models**
* **Neural Networks and Parameterizing Conditional Distributions**
* **Directed Graphical Models and Neural Networks**
* **Learning in Fully Observed Models with Neural Networks**
* **Learning and Inference in Deep Latent Variable Models**
* **Intractabilities**

## Probability Models

In machine learning, probabilistic models are essential for describing various phenomena observed in data. These models formalize knowledge and skills and are crucial for prediction and decision-making tasks. As data rarely provides a complete picture of the underlying phenomena, probabilistic models incorporate uncertainty by specifying probability distributions over model parameters.

Probabilistic models can include both continuous and discrete variables, and the most comprehensive forms capture all correlations and dependencies between variables in a joint probability distribution. In notation, the observed variables are represented as a vector, denoted as $\mathbf{x}$, which is a random sample from an unknown true distribution $p^*(\mathbf{x})$.

The learning process involves finding the parameters $\theta$ of a chosen model $p_\theta(\mathbf{x})$ that approximate the true data distribution $p^*(\mathbf{x})$ The goal is to make $p_\theta(\mathbf{x})$ as close as possible to $p^*(\mathbf{x})$ for any observed data point $\mathbf{x}$

Flexibility in the model $p_\theta(\mathbf{x})$ is crucial to adapt to different datasets effectively, while incorporating prior knowledge about the data distribution aids in constructing more accurate models. The ultimate aim is to strike a balance between model flexibility and prior knowledge to achieve an accurate representation of the underlying data distribution.

## Conditional Models

Conditional models, denoted as $p_\theta(y|x)$, approximate the underlying conditional distribution $p^*(y|x)$, where $x$ represents the input variable and $y$ represents the output variable. These models are optimized to closely match the unknown underlying distribution $p^*(y|x)$, ensuring that for any given $x$ and $y$, $p_\theta(y|x)$ approximates $p^*(y|x)$. A common example is image classification, where $x$ is an image and $y$ is the corresponding class label. Conditional models become more challenging when dealing with high-dimensional output variables, such as image or video prediction tasks. Despite the complexity, the methods introduced for unconditional modeling are generally applicable to conditional models. In such cases, the conditioning data can be treated as inputs to the model, akin to parameters, albeit without optimization over their values.

## Neural Networks and Parameterized Conditional Distributions

Parameterizing conditional distributions with neural networks involves using neural networks, a type of deep learning model, as flexible function approximators. These models consist of multiple layers of artificial neurons and are adept at learning complex relationships in data.

In the context of probabilistic modeling, neural networks can represent probability distributions, such as probability density functions (PDFs) for continuous variables or probability mass functions (PMFs) for discrete variables.

- **Probability Density Functions (PDFs)**: PDFs describe the likelihood of a continuous random variable taking on a specific value. They represent the probability of observing a particular outcome within a continuous range of possible outcomes. For example, in the case of image classification, a PDF might describe the likelihood of an image belonging to a certain class.
- **Probability Mass Functions (PMFs)**: PMFs describe the likelihood of a discrete random variable taking on each possible value. They represent the probability of observing a specific outcome from a finite set of possible outcomes. In image classification, a PMF could describe the probability distribution over different class labels for a given image.

```{figure} figures/nnsoftmax.png
---
height: 400px
name: nnsoftmax-fig
---
Neural Network for Conditional Distributions
```
This is achieved by applying a softmax function to the output of the neural network, often used to convert the raw output of the network into a probability distribution. It takes a vector of arbitrary real-valued scores as input and normalizes it into a probability distribution over multiple classes. The softmax function ensures that the probabilities sum up to one, making it suitable for representing categorical distributions. In image classification tasks, the output of the neural network is passed through a softmax function to obtain the probabilities of different classes for a given input image.

By using neural networks to parameterize conditional distributions, we can effectively model complex relationships between input and output variables, enabling tasks such as image classification to be addressed in a probabilistic framework. This approach leverages the scalability and flexibility of neural networks, allowing for efficient optimization and handling of large datasets.

## Neural Networks and Directed Graphical Models

Directed Graphical Models (DGMs), also known as directed probabilistic graphical models (PGMs) or Bayesian networks, organize variables into a directed acyclic graph. In these models, the joint distribution over variables factors into prior and conditional distributions.

$$
p_\theta(x_1,...,x_M)=\prod_{j=1}^Mp_\theta(x_j|Pa(x_j))

$$

where $Pa(x_j)$ is the set of parent variables of node $j$ in the directed graph and for root-nodes, the set of parents is the empty set; such that the distribution is unconditional.

Neural networks take the parents of a variable as input and output the distributional parameters for that variable. This enables DGMs to capture complex relationships between variables. The next step is to discuss how to learn the parameters of such models when all variables are observed in the data.

## Learning in Fully Observed Models with Neural Networks

In fully observed models with neural networks, computing and differentiating the log-probability of the data under the model leads to straightforward optimization. The dataset consists of independent and identically distributed (i.i.d.) datapoints sampled from an unchanging underlying distribution.

$$
D=\{x^{(1)},x^{(2)},...,x^{(N)}\}=\{x^{(i)}\}_{i=1}^N=x^{(i:N)}

$$

The most common criterion for probabilistic models is maximum log-likelihood (ML), which involves maximizing the sum or average of the log-probabilities assigned to the data by the model and is given by:

$$
\text{log}\space p_\theta (D)=\sum_{x \in D}\text{log}\space p_\theta (x)

$$

This objective is typically optimized using stochastic gradient descent (SGD), which utilizes randomly drawn minibatches of data $M \subset D$ or size $N_M$, to form an unbiased estimator of the ML criterion formed by:

$$
\frac{1}{N_D}\text{log}\space p_\theta(D)\simeq\frac{1}{N_M}\text{log}\space p\theta(M)=\frac{1}{N_M}\sum_{x\in M}\text{log}\space p_\theta(x)

$$

where the $\simeq$ symbol denotes that one side is an $unbiased$ $estimator$ of the other.

The unbiased stochastic gradients computed from these minibatches are then used to iteratively optimize the objective function using the backpropagation algorithm to update the parameters $\theta$ as follows:

$$
\theta_{t+1} \leftarrow \theta_t + \alpha_t \cdot \nabla \mathbb{L}(\theta,\mathcal{E})

$$

where $\alpha_t$ is  a learning rate, $\mathbb{L}(\theta,\mathcal{E})$ is the unbiased estimator of the objective i.e. $\mathbb{E}_{\mathcal{E}\sim p(\mathcal{E})}[\mathbb{L}(\theta,\mathcal{E})]=L(\theta)$, and the random variable $\mathcal{E}$ denotes posterior sampling noise.

From a Bayesian perspective, improvements upon ML can be achieved through methods such as maximum a posteriori (MAP) estimation or inference, which maximizes the log-posterior w.r.t. $\theta$. Given i.i.d. data $\mathcal{D}$, this is:

$$
L^{MAP}(\theta)=\text{log}\space p(\theta) + L^{ML}(\theta)+\text{constant}

$$

## Learning and Inference in Deep Latent Variable Models

### Latent Variables

In directed models with latent variables, these variables $z$ are integrated into the model but remain unobserved in the dataset. Denoted as $z$, they contribute to the joint distribution $p_\theta(x, z)$, encompassing both observed $x$ and latent variables. The marginal distribution $p_\theta(x)$ over observed variables is obtained by integrating out the latent variables from the joint distribution. This marginal likelihood, or model evidence, captures the overall probability of observing the data under the model. Such implicit distributions over $x$ offer flexibility; for instance, if $z$ is discrete and $p_\theta(x|z)$ is Gaussian, $p_\theta(x)$ becomes a mixture-of-Gaussians distribution. With continuous $z$, $p_\theta(x)$ represents an infinite mixture, potentially more expressive than discrete mixtures, referred to as compound probability distributions.

### Deep Latent Variable Models

Deep latent variable models (DLVMs) are models that incorporate latent variables $z$ and are parameterized by neural networks. These latent variables are unobserved in the dataset but are essential for capturing underlying patterns or factors influencing the observed data $x$.

DLVMs offer a significant advantage in that they can model complex dependencies in the data, even when each individual factor (such as the prior or conditional distribution) in the model is relatively simple. For example, even if the prior distribution $p_\theta(z)$ or the conditional distribution $p_\theta(x|z)$ follows a straightforward distribution like Gaussian, the resulting marginal distribution $p_\theta(x)$ can be highly complex and capable of representing almost arbitrary dependencies among the observed variables. This flexibility allows DLVMs to approximate intricate underlying distributions effectively.

One of the simplest and most common DLVM structures involves factorizing the joint distribution $p_\theta(x, z)$ into a product of a prior distribution $p_\theta(z)$ over the latent variables and a conditional distribution $p_\theta(x|z)$ given the latent variables. This factorization provides a straightforward yet powerful way to model the relationship between the observed and latent variables. The prior distribution $p_\theta(z)$ represents the underlying distribution of the latent variables, while the conditional distribution $p_\theta(x|z)$ captures how the observed variables are generated from the latent variables.

Overall, DLVMs offer a versatile framework for learning complex representations of data, making them suitable for a wide range of applications in machine learning and artificial intelligence.


### Intractabilities

In deep latent variable models (DLVMs), the challenge lies in computing the marginal probability of data $p_\theta(x)$, which is typically intractable due to the integral involved in its computation. This intractability stems from the fact that there is no analytic solution or efficient estimator for the integral involved in computing $p_\theta(x)$.

This intractability extends to the posterior distribution $p_\theta(z|x)$, which is also challenging to compute precisely. The joint distribution $p_\theta(x, z)$ is computationally efficient to evaluate, and the relationship between $p_\theta(z|x)$ and $p_\theta(x)$ is described through a basic identity:

$$
p_\theta(z|x)=\frac{p_\theta(x,z)}{p_\theta(x)}

$$

However, both the marginal likelihood $p_\theta(x)$ and the posterior $p_\theta(z|x)$ are intractable in DLVMs. Furthermore, the posterior over the parameters of neural networks $p(\theta|D)$ is also generally intractable to compute exactly.

To address this challenge, approximate inference techniques are employed. These techniques allow for the approximation of the posterior $p_\theta(z|x)$ and the marginal likelihood $p_\theta(x)$ in DLVMs. They are widely employed in various areas of machine learning and statistics, including deep latent variable models (DLVMs).

Here are some common approximate inference techniques:

1. **Variational Inference (VI)**: Variational inference approximates the posterior distribution by framing the inference problem as an optimization task. It involves finding a distribution (often from a predefined family of distributions) that best approximates the true posterior distribution by minimizing a divergence measure, such as the Kullback-Leibler (KL) divergence, between the approximate and true distributions.
2. **Expectation Maximization (EM)**: EM is an iterative optimization algorithm used to find maximum likelihood or maximum a posteriori estimates in the presence of latent variables. In each iteration, it alternates between the expectation (E-step), where it computes the expected value of the latent variables given the observed data and current parameter estimates, and the maximization (M-step), where it updates the parameters to maximize the expected log-likelihood obtained from the E-step.
3. **Monte Carlo Methods**: Monte Carlo methods rely on random sampling to estimate complex distributions. Techniques such as Markov Chain Monte Carlo (MCMC) and Sequential Monte Carlo (SMC) generate samples from the target distribution and use these samples to approximate expectations or compute posterior probabilities.
4. **Sampling-based Methods**: Sampling-based methods, including importance sampling, rejection sampling, and Gibbs sampling, draw samples from a proposal distribution and use them to approximate the target distribution. These methods can be effective for approximating complex distributions but may suffer from high variance or inefficiency in high-dimensional spaces.
5. **Approximate Bayesian Computation (ABC)**: ABC methods approximate the posterior distribution by simulating data from the model and comparing simulated data with observed data. Instead of directly computing the posterior distribution, ABC methods accept parameter values that produce simulated data similar to the observed data within a specified tolerance level.

Each of these techniques has its advantages and limitations, and the choice of method often depends on the specific characteristics of the problem at hand, such as the complexity of the model, the dimensionality of the data, and computational resources available. In DLVMs, approximate inference techniques play a crucial role in training models, making predictions, and performing various inference tasks.
