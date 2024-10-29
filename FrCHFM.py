#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import cv2
from scipy.special import eval_chebyt
import matplotlib.pyplot as plt
import cupy as cp


# In[25]:


def convert_to_polar(image):
    """
    Convert the input image from Cartesian (x, y) coordinates to polar (r, θ) coordinates.
    
    Parameters:
        image (2D numpy array): Grayscale image.
    
    Returns:
        polar_image (2D numpy array): The image in polar coordinates.
        r_coords (1D numpy array): The radial distances for each pixel.
        theta_coords (1D numpy array): The angular coordinates for each pixel.
    """
    rows, cols = image.shape
    center = (cols // 2, rows // 2)  # Center of the image

    # Create coordinate grids (X, Y) in Cartesian space
    x = np.linspace(-center[0], center[0], cols)
    y = np.linspace(-center[1], center[1], rows)
    X, Y = np.meshgrid(x, y)
    
    # Convert to polar coordinates
    r = np.sqrt(X**2 + Y**2) / np.max([center[0], center[1]])  # Normalize r to the range [0, 1]
    theta = np.arctan2(Y, X)
    
    return image, r, theta


# In[26]:


def radial_basis_function(n, t, r):
    """
    Compute the radial basis function R_t^n(r) for the fractional Chebyshev-Fourier moments.
    
    Parameters:
        n (int): Order of the Chebyshev polynomial.
        t (float): Fractional parameter.
        r (2D numpy array): Radial coordinates of the image.
    
    Returns:
        Rtn (2D numpy array): The computed radial basis function values.
    """
    Rtn = np.zeros_like(r)
    
    for k in range(n//2 + 1):
        Rtn += (-1)**k * (np.math.factorial(n - k) /
                          (np.math.factorial(k) * np.math.factorial(n - 2*k))) * \
                (4 * r**t - 2)**(n - 2*k)
    
    # Multiply by the fractional part
    Rtn *= np.sqrt(t) * r**(t - 1)
    
    return Rtn

# GPU-accelerated version of radial_basis_function
def radial_basis_function_gpu(n, t, r):
    
    # Compute factorials on CPU first, then transfer to GPU
    factorials = cp.array([np.math.factorial(n - k) for k in range(n // 2 + 1)], dtype=cp.float32)
    
    Rtn = cp.zeros_like(r, dtype=cp.float32)

    # Use a loop to compute each term to manage memory usage
    for k in range(n // 2 + 1):
        coeff = (-1)**k * factorials[k] / (np.math.factorial(k) * np.math.factorial(n - 2 * k))
        Rtn += coeff * (4 * r**t - 2)**(n - 2 * k)

    # Apply the fractional term after the loop
    result = cp.sqrt(t) * r**(t - 1) * Rtn

    return result


# In[27]:


def compute_weighted_amplitude(frchfm, tau=1, xi=1):
    """
    Compute the top 24 Weighted Amplitude (WA) values from the Fractional Chebyshev-Fourier Moments (FrCHFM).
    
    Parameters:
        frchfm (2D numpy array): Fractional Chebyshev-Fourier moments matrix (complex values).
        tau (int): Control parameter for order weight (default is 1).
        xi (int): Control parameter for repetition weight (default is 1).
    
    Returns:
        wa_top24 (1D numpy array): Array containing the top 24 weighted amplitudes.
    """
    # Generate weights in a vectorized form
    n_indices = np.arange(frchfm.shape[0])
    m_indices = np.arange(frchfm.shape[1])
    weight_matrix = tau * n_indices[:, None] + xi * m_indices[None, :]

    # Apply weights to the absolute values of frchfm and flatten
    wa_flat = (weight_matrix * np.abs(frchfm)).flatten()

    # Return the top 24 values
    return np.partition(wa_flat, -24)[-24:]


# In[28]:


def compute_frchfm(image, max_degree, t):
    """
    Compute Fractional Chebyshev-Fourier moments (FrCHFMs) of an input image.
    
    Parameters:
        image (2D numpy array): Grayscale image.
        max_degree (int): Maximum degree of the Chebyshev polynomial.
        t (float): Fractional parameter.
    
    Returns:
        frchfm (2D numpy array): The computed Fractional Chebyshev-Fourier moments.
    """
    # Convert image to polar coordinates
    polar_image, r, theta = convert_to_polar(image)
    
    # Fourier transform in angular direction (theta)
    F_image = np.fft.fftshift(np.fft.fft2(polar_image))

    # Initialize an array to store the FrCHFMs
    frchfm = np.zeros((max_degree + 1, 2 * max_degree + 1), dtype=np.complex128)

    # Precompute angular components and store radial basis for each degree n
    m_values = np.arange(-max_degree, max_degree + 1)
    angular_components = np.exp(-1j * m_values[:, None, None] * theta)  # Broadcasting over m, theta

    for n in range(max_degree + 1):
        # Precompute the radial basis function once per degree
        Rtn = radial_basis_function(n, t, r)

        # Compute each FrCHFM moment using the precomputed angular components
        for idx, m in enumerate(m_values):
            frchfm[n, idx] = np.sum(F_image * Rtn * angular_components[idx] * r)

    return frchfm

# GPU-accelerated version of compute_frchfm
def compute_frchfm_gpu(image, max_degree, t):
    """
    Compute Fractional Chebyshev-Fourier moments (FrCHFMs) of an input image on the GPU.
    
    Parameters:
        image (2D numpy array): Grayscale image.
        max_degree (int): Maximum degree of the Chebyshev polynomial.
        t (float): Fractional parameter.
    
    Returns:
        frchfm (2D numpy array): The computed Fractional Chebyshev-Fourier moments.
    """
    # Convert image to polar coordinates and then transfer to GPU
    polar_image, r, theta = convert_to_polar(image)
    polar_image_gpu = cp.array(polar_image)
    r_gpu = cp.array(r)
    theta_gpu = cp.array(theta)
    
    # Fourier transform in angular direction (theta), shifted and moved to GPU
    F_image = cp.fft.fftshift(cp.fft.fft2(polar_image_gpu))

    # Initialize an array on GPU to store the FrCHFMs
    frchfm_gpu = cp.zeros((max_degree + 1, 2 * max_degree + 1), dtype=cp.complex128)

    # Precompute angular components on GPU
    m_values = cp.arange(-max_degree, max_degree + 1)
    angular_components = cp.exp(-1j * m_values[:, None, None] * theta_gpu)

    for n in range(max_degree + 1):
        # Compute radial basis function for each degree
        Rtn_gpu = radial_basis_function_gpu(n, t, r_gpu)

        # Compute FrCHFMs for each value of m in a vectorized way
        frchfm_gpu[n, :] = cp.sum(F_image * Rtn_gpu * angular_components * r_gpu, axis=(1, 2))

    # Transfer result back to CPU
    return cp.asnumpy(frchfm_gpu)


# In[29]:


def generate_binary_sequence(frchfm, tau=1, xi=1):
    """
    Compute the binary sequence from the top 24 Fractional Chebyshev-Fourier Moments (FrCHFMs) based on their intensities.

    Parameters:
        frchfm (2D numpy array): Fractional Chebyshev-Fourier moments matrix (complex values).
        tau (int): Control parameter for order weight (default is 1).
        xi (int): Control parameter for repetition weight (default is 1).

    Returns:
        binary_sequence_str (str): Binary sequence as a string (e.g., "011001101") based on top 24 moments.
    """
    # Step 1: Flatten and get the absolute values (intensities) of the FrCHFM elements
    frchfm_flat = np.abs(frchfm).flatten()
    
    # Step 2: Identify the indices of the top 24 intensities
    top_24_indices = np.argsort(frchfm_flat)[-24:]

    # Step 3: Extract only the top 24 FrCHFM elements from the original matrix
    top_24_values = frchfm_flat[top_24_indices]
    
    # Step 4: Calculate weights for the top 24 elements based on their indices (row, col in the original matrix)
    wa_values = np.zeros(24, dtype=float)
    for i, index in enumerate(top_24_indices):
        n, m = divmod(index, frchfm.shape[1])  # Convert flat index back to 2D index
        weight_w_nm = tau * n + xi * m  # Calculate weight for the index
        wa_values[i] = weight_w_nm * top_24_values[i]  # Apply weight

    # Step 5: Calculate the threshold (average of weighted values)
    threshold = np.mean(wa_values)
    
    # Step 6: Generate binary sequence by comparing each weighted value to the threshold
    binary_sequence = [(1 if wa >= threshold else 0) for wa in wa_values]
    binary_sequence_str = ''.join(map(str, binary_sequence))  # Convert list to a string
    
    return binary_sequence_str


# In[30]:


def get_wa_sequences_for_group(image_group):
    """
    Calculate the top 24 WA sequences for each image in an image group.
    
    Parameters:
        image_group (list of numpy arrays): List of images in the current group.
        
    Returns:
        wa_sequences (list of arrays): List of top 24 WA sequences for each image.
    """
    wa_sequences = []
    for image in image_group:
        # Calculate FrCHFM for the current image
        frchfm = compute_frchfm(image)
        
        # Compute and retrieve top 24 weighted amplitudes
        wa_sequence = compute_weighted_amplitude(frchfm)
        
        # Append the top 24 moments to wa_sequences list
        wa_sequences.append(wa_sequence)
    
    return wa_sequences


# In[31]:


def construct_binary_sequence_mapping(image_groups): # PUT THIS SOMEWHERE ELSE. NOT RELATED TO FrCHFM LOGIC
    """
    Constructs the final binary sequence mapping by combining PM and AM mappings.

    Parameters:
        image_groups (list of lists): List of image groups, where each sublist represents a group of images.

    Returns:
        binary_sequence_mapping (dict): A dictionary mapping each image to its binary sequence.
    """
    binary_sequence_mapping = {}
    
    # Iterate over each image group and process WA sequences
    pm_mapping = construct_pm_mapping(image_groups)
    for pm_index, image_group in pm_mapping.items():
        # Obtain top 24 WA sequences for images in the current group
        wa_sequences = get_wa_sequences_for_group(image_group)
        
        # Construct AM mapping for the current group
        am_mapping = construct_am_mapping(image_group, wa_sequences)
        
        # Combine PM and AM mappings for each image
        for am_index, image in am_mapping.items():
            binary_sequence = (pm_index, am_index)
            binary_sequence_mapping[image] = binary_sequence
            
    return binary_sequence_mapping


# In[32]:


def demofunctions():
    # Load the grayscale image
    image_path = "IMG_0612.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Define the maximum degree for Chebyshev polynomials and the fractional parameter t
    max_degree = 5  # Change depending on desired precision
    fractional_parameter_t = 0.5  # Fractional parameter t

    # Compute the FrCHFMs
    frchfm = compute_frchfm(image, max_degree, fractional_parameter_t)
    binary = generate_binary_sequence(frchfm)

    # Display results
    print("Fractional Chebyshev-Fourier Moments: ")
    print(frchfm)
    print("Binary sequence:")
    print(binary, len(binary))

    # Visualize the magnitude of the FrCHFMs
    plt.imshow(np.abs(frchfm), cmap='gray')
    plt.colorbar()
    plt.title('Fractional Chebyshev-Fourier Moments (Magnitude)')
    plt.show()


# In[33]:


get_ipython().system('jupyter nbconvert --to python FrCHFM.ipynb')


# In[34]:


#pip install opencv-python

