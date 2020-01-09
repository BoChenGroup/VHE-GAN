This code is a demo for manuscript "Variational Hetero-Encoder Randomized Generative Adversarial Networks for Joint Image-Text Modeling", submitted to ICLR 2020.

VHE-StackGAN++: the demo for VHE-StackGAN++ on Birds. Please run 'main' directly.

VHE-raster-scan-GAN: the demo for VHE-raster-scan-GAN on Birds. Please run 'main' directly.

data:
overview: the Birds data for VHE-StackGAN++ and VHE-raster-scan-GAN, we provide our pre-processed data as pkl files in train and test folders. Due to the space limit, the user should download images from 'http://www.vision.caltech.edu/visipedia/CUB-200-2011.html' and put the images of CUB in the folder---./data/birds/CUB_200_2011/images/.
concretely:
the train folder has the components below:
	filenames.pickle(training images name)
	class_info.pickle(training images label)
	char_textbows.pickle(bag of words processed by us)
the test folder has the same components as in train folder.

NOTE: We only run our code on Win10 and not try other systems; in order to use winPGBN_sampler, you may firstly need to install visual studio. Then you will also need to step in PGBN_sampler.py and alternate the relative path with absolute path. If you can not handle the configuration, you can comment updatePhi and associated content in train.py and directly load our pre-trained Phi in the data path named---Birds_1000_Phi.mat.
