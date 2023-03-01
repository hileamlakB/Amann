from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os

# Define the font files
font_files = ["font1.ttf", "font2.ttf", "font3.ttf"]

# Define the font size and image size
font_size = 100
image_size = 128

# Define the input CSV file
csv_file = "input.csv"

# Define the column to use for drawing characters
column_name = "text"

# Create the output directory
output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)

# Load the CSV file into a pandas dataframe
df = pd.read_csv(csv_file)

# Loop through each font file
for font_file in font_files:

    # Load the font
    font = ImageFont.truetype(font_file, font_size)

    # Loop through each row in the dataframe
    for index, row in df.iterrows():

        # Get the text to draw from the specified column
        text = str(row[column_name])

        # Create a new image
        image = Image.new("L", (image_size, image_size), color=255)

        # Draw the text
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), text, font=font, fill=0)

        # Save the image
        filename = os.path.join(output_dir, f"{font_file[:-4]}_{index}.png")
        image.save(filename)

