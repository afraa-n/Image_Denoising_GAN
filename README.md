# Applied Computer Vision Online (AIPI 590.06.Sp25) - Course Project 3
# Image Denoising with GANs

**Name: Afraa Noureen**  
**Net ID: an300**

---

## I. Project Overview
This report presents a project on using Generative Adversarial Networks (GANs) to remove noise from images. The main objective was to develop a deep learning model that can take noisy images as input and produce clean, denoised images as output.

---

## II. Background
Images often contain noise, which appears as grainy or speckled patterns that reduce visual quality. Noise can be caused by various factors such as camera sensors, poor lighting conditions, etc. Removing noise from images is an important task in many applications.  

GANs are a type of deep learning model that consists of two neural networks: a generator and a discriminator. The generator creates new data samples, while the discriminator tries to distinguish between real and generated samples. By training these networks together, the generator learns to produce realistic outputs.

---

## III. Methodology

The GAN model used in this project has two main components:

1. A U-Net generator, which takes noisy images as input and produces denoised images. The U-Net architecture allows the model to effectively capture and combine features at different scales.  
2. A PatchGAN discriminator, which looks at small patches of the image to determine if they are real or generated. This helps the model focus on local details and textures.  

The model was trained on the CIFAR-10 dataset, which contains 60,000 small color images. To create training data, random Gaussian noise was added to the clean images. The model learned to map noisy images to their clean counterparts.
The loss function used to train the model included three parts:   
1. L1 pixel loss to ensure the output matches the target.  
2. Adversarial loss to make the output look realistic.  
3. Perceptual loss based on VGG19 features to capture high-level similarities.  

---

## IV. Metrics Used

The following metrics were used to evaluate the performance of the denoising model:  
1. Peak Signal-to-Noise Ratio (PSNR): PSNR measures the ratio between the maximum possible power of a signal and the power of corrupting noise. A higher PSNR indicates better image quality. It is calculated using the mean squared error (MSE) between the denoised and clean images.    
2. Structural Similarity Index (SSIM): SSIM assesses the perceived quality of an image by comparing the similarity of its luminance, contrast, and structure with the reference image. SSIM values range from -1 to 1, with 1 indicating perfect similarity.  
3. Fréchet Inception Distance (FID): FID measures the difference between the distributions of generated and real images. It is calculated by comparing the activations of a pre-trained Inception v3 model on the generated and real images. Lower FID scores suggest that the generated images are more similar to the real images.

---

## V. Results & Discussion

The model was trained for 50 epochs, but early stopping was triggered after 13 epochs based on the validation PSNR. The figure below shows the training metrics over the epochs.
![image](https://github.com/user-attachments/assets/038ecb93-d442-4a1e-9531-81092a2d29b3)

The PSNR and SSIM values steadily increased, indicating an improvement in the quality of the denoised images. The generator and discriminator losses stabilized as the training progressed. The FID scores fluctuated but generally showed a downward trend, suggesting that the distribution of the denoised images became closer to the clean images.

To visually assess the denoising performance, the figure below shows an example of noisy input, denoised output, and the corresponding clean ground truth image.

![image](https://github.com/user-attachments/assets/4a7a183e-a9fd-4c79-b623-d2a9e1eea621)

The denoised image shows a significant reduction in noise compared to the input while preserving the main image content. However, some fine details may be slightly blurred or lost in certain cases.

On the test set, the trained model achieved an average PSNR of 15.39 dB and an average SSIM of 0.3630. These quantitative metrics indicate an improvement in image quality compared to the noisy inputs, but there is still room for further improvement.

---

## VI. Conclusion

In this project, I developed a GAN-based model for image denoising. The U-Net generator and PatchGAN discriminator, trained with a combination of pixel loss, adversarial loss, and perceptual loss, were able to effectively remove noise from images while preserving the essential content.

The results show the potential of GANs for image denoising tasks. However, there is room for further improvement, such as expanding the training data, exploring more advanced architectures, and fine-tuning the loss functions.

---

## VII. Coding Implementation

The implementation can be found at the end of the `Image Denoising using GANs - Afraa.ipynb` file
