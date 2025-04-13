# image_text_question_answering
Image text question answering by leveraging phi-3 model with SigLib as image embedding

This repository is used for building multimodal question answering LLM.

Various steps are as below:

## Image Text Question Answering Generation

In this step, taken 55 images from ciphar-10 dataset. Used the hugging face SmolVLM2-2.2B-Instruct and each image the following 5 questions are given 
   i.Can you describe this image?
   ii. Explain the purpose of image
   iii. Can you comment on size and kind of this image
  iv. Create a interesting story about this image
  v. Elaborate the setting of the image

  The image, question, answer are used for generating dataset. The colab notebook used is as below:
  
https://github.com/monimoyd/image_text_question_answering/blob/main/imge_text_question_answering_dataset_generation.ipynb

The generated dataset is given below:
https://github.com/monimoyd/image_text_question_answering/blob/main/image_text_question_answer_dataset.csv

## Alignment of Phi-3 with SigLib embedding

* Data Preparation:
   - Dataset Format: Your dataset should consist of image files and corresponding text descriptions. Organize it into a structure that PyTorch Dataset can handle. A        common format is a CSV or JSON file listing image paths and text captions.
   - Custom Dataset Class: Create a PyTorch Dataset class to load and preprocess your data.
 * Image Transformations: Define image transformations using torchvision.transforms. Common transformations include resizing, normalization, and data augmentation.

 * Text Tokenizer: Load the Phi-3 tokenizer using AutoTokenizer.
   
 * . Model Definition: SigLIP Image Encoder: Use a pre-trained SigLIP image encoder from timm.
 
*  Phi-3 Text Encoder (Frozen): Load the pre-trained Phi-3 model using AutoModel. Crucially, freeze its parameters.- 

## SFT on Phi-3 model uding QLoRA

 In this step, Supervised Fine Tuning is done on Phi-3 model using QLoRA
 
* Image and Text Input: The ImageTextQADataset  handles image and text inputs, combining the question and answer for supervised fine-tuning.

* SigLIP Image Encoder: The SigLIPImageEncoder class is included for encoding images using a pre-trained SigLIP model.

*Phi-3 Integration: The Phi3WithImage class integrates the Phi-3 model with the SigLIP image encoder. It includes a projection layer to map the image embeddings to the same dimensionality as the Phi-3 embeddings.

* QLoRA Implementation: The code implements QLoRA (Quantization-Aware Low-Rank Adaptation) for efficient fine-tuning of the Phi-3 model. This involves:
          - BitsAndBytesConfig: Using BitsAndBytesConfig to load the Phi-3 model in 4-bit precision.
          - prepare_model_for_kbit_training: Preparing the model for k-bit training.
          - LoraConfig: Creating a LoraConfig to specify the LoRA parameters.
          -   get_peft_model: Wrapping the Phi-3 model with the LoRA adapter.
          - Data Loading and Preprocessing: The load_and_preprocess_data function loads and preprocesses the data from a JSON file. You'll need to adapt this function to your specific data format.
  
* Training Loop: The train function implements the training loop, iterating over the data loader and updating the model parameters.
Concatenation of Image Embeddings: The code includes a basic approach for concatenating the image embeddings to the input sequence. This is a simplified approach, and more sophisticated methods exist, such as using a cross-attention mechanism or learned positional embeddings for the image tokens.


  

  


