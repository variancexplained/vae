---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Variational Encoders: Going Deeper

## Encoder
In the last section, we introduced deep latent-variable models (DLVMs), and the problem of estimating the log-likelihood and posterior distributions in these models. Variational Autoencoders (VAEs) provide an efficient way to optimize deep latent-variable models (DLVMs) and perform posterior inference using stochastic gradient descent (SGD). To address the intractable posterior inference in DLVMs, VAEs introduce a parametric inference model $q_\phi(z|x)$, also known as the **encoder** or recognition model. This model *approximates* the true posterior distribution $p_\theta(z|x)$ by optimizing the variational parameters $\phi$, which include the weights and biases of a neural network.

The encoder maps the input $x$ to the parameters of a Gaussian distribution, typically the mean $\mu$ and log-variance $\log \sigma$:

$$
(\mu, \log \sigma) = \text{EncoderNeuralNet}_\phi(x) \\
q_\phi(z|x) = \mathcal{N}(z; \mu, \text{diag}(\sigma))

$$

This approach, called **amortized variational inference**, uses a single encoder neural network to perform posterior inference for all data points simultaneously. This avoids the need for separate optimization for each datapoint, significantly improving efficiency by leveraging stochastic gradient descent (SGD).

## Evidence Lower Bound (ELBO)
Before delving into the derivation of the Evidence Lower Bound (ELBO) using the log likelihood, let's set up the process. Our objective is to formulate a tractable optimization objective for Variational Autoencoders (VAEs) that provides insight into the model's performance and the tightness of its approximation to the true log-likelihood. To achieve this, we aim to express the log likelihood of the data in terms of an expectation over a simpler distribution, which leads us to the ELBO. By deriving the ELBO using the log likelihood, we can establish a foundational understanding of how VAEs learn and optimize their parameters. This derivation process will offer valuable insights into the underlying principles of VAEs and their effectiveness in capturing complex data distributions.

The optimization objective of a Variational Autoencoder (VAE) is the Evidence Lower Bound (ELBO), also known as the variational lower bound. ELBO is derived to provide insight into its tightness without using Jensen's inequality. 

Using Jensen's inequality in the derivation of the Evidence Lower Bound (ELBO) for Variational Autoencoders (VAEs) typically results in a looser bound compared to alternative derivations that avoid it. While Jensen's inequality is a useful tool in many mathematical contexts, in the context of VAEs, tighter bounds are preferred because they more accurately approximate the true log-likelihood of the data.

By avoiding Jensen's inequality, we can derive the ELBO in a way that provides greater insight into its tightness. This allows us to better understand how well the ELBO approximates the true log-likelihood and how effectively the VAE model is learning the underlying data distribution. In essence, bypassing Jensen's inequality enables a more nuanced analysis of the VAE optimization process and its performance.

For any choice of inference model $q_\phi(z|x)$ and variational parameters $\phi$, the log-likelihood $\log p_\theta(x)$ can be expressed as:

$$
\log p_\theta(x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x)] \\ 
= \mathbb{E}_{q_\phi(z|x)}\left[ \log \left( \frac{p_\theta(x, z)}{p_\theta(z|x)} \right) \right] \\
= \mathbb{E}_{q_\phi(z|x)}\left[ \log \left( \frac{p_\theta(x, z) q_\phi(z|x)}{q_\phi(z|x) p_\theta(z|x)} \right) \right] \\
= \mathbb{E}_{q_\phi(z|x)}\left[ \log \left( \frac{p_\theta(x, z)}{q_\phi(z|x)} \right) \right] + \mathbb{E}_{q_\phi(z|x)}\left[ \log \left( \frac{q_\phi(z|x)}{p_\theta(z|x)} \right) \right]
$$

The first term is the ELBO, denoted as $L_{\theta, \phi}(x)$:

$$L_{\theta, \phi}(x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x, z) - \log q_\phi(z|x)]
$$

The second term is the Kullback-Leibler (KL) divergence between $q_\phi(z|x)$ and $p_\theta(z|x)$:
$$

\text{KL}(q_\phi(z|x) \| p_\theta(z|x)) \geq 0

$$
The ELBO is a lower bound on the log-likelihood of the data:
$$
L_{\theta, \phi}(x) = \log p_\theta(x) - \text{KL}(q_\phi(z|x) \| p_\theta(z|x)) \leq \log p_\theta(x)
$$
The KL divergence measures two distances:
1. The divergence of the approximate posterior from the true posterior.
2. The gap between the ELBO and the marginal likelihood, also called the tightness of the bound.

The closer $q_\phi(z|x)$ approximates the true posterior $p_\theta(z|x)$, the smaller the KL divergence and the tighter the bound.

Thus, maximizing the  ELBO $L_{\theta, \phi}(x)$ w.r.t. the parameters $\theta$ and $\phi$, we concurrently optimize two things we care about:
1. Approximately maximize the marginal likelihood $p_\theta(x)$, meaning that our generative model will improve.
2. Minimize the KL divergence of the approximation $q_\phi(z|x)$ from the true posterior $p_\theta(z|x)$, so $q_\phi(z|x)$ also improves.

## Stochastic Gradient-Based Optimization of the ELBO
In stochastic gradient-based optimization of the ELBO, we exploit its key property, enabling joint optimization with respect to all parameters (φ and θ) using stochastic gradient descent (SGD). The ELBO objective, $L_{\theta, \phi}(D)$, is computed as the sum or average of individual-datapoint ELBOs over the dataset $D$. While the individual-datapoint ELBO and its gradient are generally intractable, unbiased estimators exist, facilitating minibatch SGD. Obtaining unbiased gradients of the ELBO with respect to the generative model parameters $\theta$ is straightforward. However, for the variational parameters $\phi$, obtaining unbiased gradients is more challenging due to the dependence on the distribution $q_\phi(z|x)$. Nevertheless, for continuous latent variables, the reparameterization trick offers a means to compute unbiased estimates of $\nabla_{\theta, \phi} L_{\theta, \phi}(x)$, enabling optimization of the ELBO using SGD. This approach forms the basis for algorithmic implementation, facilitating efficient optimization of VAEs.

```{figure} figures/sgd-vae.png
---
height: 400px
name: sgd-vae-fig
---
Stochastic Gradient-Based Optimization of the ELBO
```
In {numref}`sgd-vae-fig` we have a simple schematic of the computational flow in the variational autoencoder.