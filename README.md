# alt-text-generator
This repository contains a Google Colab notebook designed to generate alternative text (alt text) for images using the BLIP (Bootstrapping Language-Image Pre-training) model. The generated alt text descriptions have words separated by hyphens to meet specific formatting requirements.

Features
Image Processing: Fetches images from provided URLs and processes them using the BLIP model.
Alt Text Generation: Generates descriptive alt text for each image.
Hyphenated Descriptions: Formats the generated descriptions by adding a hyphen between each word.
Google Drive Integration: Reads image URLs from an Excel file stored in Google Drive and writes the generated alt text back to an Excel file in Google Drive.
Dependencies
The following Python libraries are required to run the notebook:

pandas
requests
pillow
transformers
These dependencies are automatically installed in the provided Google Colab environment.

Usage
Step 1: Clone the Repository
Clone this repository to your local machine or directly open it in Google Colab.

Step 2: Set Up Google Drive
Ensure your Google Drive is mounted in the Colab environment. This can be done using the following code snippet:

python
Copy code
from google.colab import drive
drive.mount('/content/drive')
Step 3: Prepare Input File
Upload an Excel file (images_missing_alt_text-1.xlsx) to your Google Drive. This file should contain image URLs under the column named image_url.

Step 4: Run the Notebook
Execute the cells in the notebook to generate hyphenated alt text for each image URL. The notebook performs the following actions:

Installs the necessary libraries.
Loads the BLIP model and processor.
Reads the image URLs from the provided Excel file.
Generates hyphenated alt text for each image.
Saves the results back to an Excel file in Google Drive.
Example Code
Below is the example code used in the notebook:

python
Copy code
# Install necessary libraries
!pip install pandas requests pillow transformers

from google.colab import drive
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration

# Mount Google Drive
drive.mount('/content/drive')

input_file_path = '/content/drive/MyDrive/images_missing_alt_text-1.xlsx'  # Verify the file name and path
output_file_path = '/content/drive/MyDrive/your_output_file.xlsx'

# BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_alt_text(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Check for HTTP errors
        img = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Process the image and generate a description from the model
        inputs = processor(img, return_tensors="pt")
        out = model.generate(**inputs)
        description = processor.decode(out[0], skip_special_tokens=True)
        
        # Add "-" between each word
        words = description.split()
        hyphenated_description = "-".join(words)
        
        return hyphenated_description
    except requests.exceptions.RequestException as e:
        return f"Error fetching image: {str(e)}"
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Read the Excel file
df = pd.read_excel(input_file_path)

# Generate alt text and add it to the dataframe
df['alt_text'] = df['image_url'].apply(generate_alt_text)

# Write the results to an Excel file on Google Drive
df.to_excel(output_file_path, index=False)
print(f"Alt text descriptions have been saved to {output_file_path}")
Step 5: Check the Output
After running the notebook, check your Google Drive for the output Excel file. This file will contain the original image URLs and the generated hyphenated alt text descriptions.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contributions
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

Contact
If you have any questions or suggestions, feel free to reach out by opening an issue on this repository.

