---
layout: post
title:  "How object tracking works - Diving into the MOSSE algorithm"
date:   2024-06-02 00:14:00 +0200
categories: computer-vision
---

**TLDR: [https://github.com/oralian/mosse](https://github.com/oralian/mosse)**

If you've done a bit of research on object tracking, you probably came across MOSSE. Let's see how it works.

## The FFT trick with correlation filters

Let's say we're trying to search an image (the template) into another larger image (the target). One common way to do this would to slide the template across each positions of the target, and compute a sort of matching "score" that would indicate us if we have something similar or not. This is exactly what correlation filters do! The "correlation" part is the computation of the score, and the "filter" part is applying a sliding window across our image, and doing something with it ([convolution]). There are many ways to compute this correlation score, I'll let you figure out some possibilities ;)

As you can imagine however, this can be quite slow. One strategy that has been quite successful to speed up things is the use of the Fourier transform. This  trick involes computing the correlation in the *frequency domain* instead of the *spatial domain*. The Fourier transform is a mathematical tool that allows us to switch between these two domains, and instead of sliding the template across each position of the image, we can get our correlation result in one go by multiplying it all in the frequency domain ([convolution theorem]).

## The MOSSE algorithm

### Preprocessing

Before getting into the algorithm itself, let's see an important part of it: the preprocessing of the images. MOSSE uses the same preprocessing steps as "Average of Synthetic Exact Filters" (David S. Bolme, Bruce A. Draper and J. Ross Beveridge). Here are the steps:

1. Log transformation to reduce the effect of shadows and intense lighting.
```python
def log_transform(image: np.ndarray) -> np.ndarray:
    return np.log(image + 1)
```

2. Normalization to a zero mean and squared sum of 1 for more consistent intensity values.
```python
def normalize(image: np.ndarray) -> np.ndarray:
    return (image - image.mean()) / (image.std() + 1e-5)
```

3. Cosine window (the Hanning window is used) to reduce the frequency effects at the edges of the image.
```python
def hanning_window(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    mask_col, mask_row = np.meshgrid(np.hanning(width), np.hanning(height))
    window = mask_col * mask_row
    return image * window
```

### Initialization step

The first step is to initialize the filter. The relationship between the filter, the template and the image in the Fourier domain is defined as:

$$ G = F \odot H^{*} \hspace{1cm} \text{and} \hspace{1cm} H^{*} = \dfrac{G}{F} $$

Where *G* is the correlation result (or response), *F* is the image and *H* is the filter. The * denotes the complex conjugate and ⊙ the element-wise multiplication. For the initial frame, we set the filter to return a 2D gaussian correlation centered at the center of the target:

```python
g = gauss_reponse(frame.shape, bbox.x, bbox.y)
g = g[bbox.top : bbox.bottom, bbox.left : bbox.right]
f = frame[bbox.top : bbox.bottom, bbox.left : bbox.right]

G = np.fft.fft2(g)
F = np.fft.fft2(preprocess(f))
H = G / F
```

Which gives the following result:

![MOSSE training initialization visualization](/assets/mosse/mosse_init.png)

However, as this filter overfits the first frame, it can fail to generalize to the next frames. To overcome this, MOSSE applies multiple random transformations and averages the filters to make it more robust.

### Update step

When updating the filter with new frames, MOSSE finds a filter that optimzes the following objective function:

$$ \min_{H^{*}} \sum_{i} \left| F_{i} \odot H^{*} - G_{i} \right|^{2} $$

This optimization problem can be solved using the following equation:

$$ H^{*} = \dfrac{\sum_{i} F_{i} \odot G_{i}^{*}}{\sum_{i} F_{i} \odot F_{i}^{*}} $$

MOSSE uses a running average with a learning rate *η* to update the filter:

$$ H^{*} = \dfrac{A_{i}}{B_{i}} $$

$$ A_{i} = \eta G_{i} \odot F_{i}^{*} + (1 - \eta) A_{i - 1} \hspace{1cm} and \hspace{1cm} B_i{i} = \eta F_{i} \odot F_{i}^{*} + (1 - \eta) B_{i -1} $$

More details are available in the original paper. This can be implemented as follows:

```python
f = new_frame[bbox.top : bbox.bottom, bbox.left : bbox.right]
f = preprocess(f)
G = H * F

g = normalize_range(np.fft.ifft2(G)) # Normalize between 0 and 1

max_pos = np.where(g == g.max())
bbox.x = max_pos[1].item() + bbox.left
bbox.y = max_pos[0].item() + bbox.top

f = frame[bbox.top : bbox.bottom, bbox.left : bbox.right]
F = np.fft.fft2(preprocess(f))
G = H * F

A = (learning_rate * (G * np.conj(F)) + (1 - learning_rate) * A)
B = (learning_rate * (F * np.conj(F)) + (1 - learning_rate) * B)
H = A / B
```

[convolution]: https://en.wikipedia.org/wiki/Kernel_(image_processing)
[convolution theorem]: https://en.wikipedia.org/wiki/Convolution_theorem
