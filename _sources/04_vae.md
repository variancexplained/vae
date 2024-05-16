# Variational Encoders: Explained

Having introduced the foundational principles, this section explains the basics of variational autoencoders.


## Encoder

Variational Autoencoders (VAEs) provide an efficient way to optimize deep latent-variable models (DLVMs) and perform posterior inference using stochastic gradient descent (SGD). To address the intractable posterior inference in DLVMs, VAEs introduce a parametric inference model $q_\phi(z|x)$, also known as the encoder or recognition model. This model approximates the true posterior distribution $p_\theta(z|x)$ by optimizing the variational parameters $\phi$, which include the weights and biases of a neural network.

The encoder maps the input $x$ to the parameters of a Gaussian distribution, typically the mean $\mu$ and log-variance $\log \sigma$:

$$
(\mu, \log \sigma) = \text{EncoderNeuralNet}_\phi(x) \\
q_\phi(z|x) = \mathcal{N}(z; \mu, \text{diag}(\sigma))

$$

This approach, called amortized variational inference, uses a single encoder neural network to perform posterior inference for all data points simultaneously. This avoids the need for separate optimization for each datapoint, significantly improving efficiency by leveraging stochastic gradient descent (SGD).This process is known as amortized variational inference, where a single encoder neural network performs posterior inference for all data points, avoiding the need for per-datapoint optimization and leveraging the efficiency of SGD.

## Evidence Lower Bound (ELBO)
The optimization objective of a Variational Autoencoder (VAE) is the Evidence Lower Bound (ELBO), also known as the variational lower bound. ELBO is derived to provide insight into its tightness without using Jensen's inequality. For any choice of inference model $q_\phi(z|x)$ and variational parameters $\phi$, the log-likelihood $\log p_\theta(x)$ can be expressed as:

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
