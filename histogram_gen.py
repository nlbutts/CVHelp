import os
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
import shutil

"""This script generates a histogram of images
"""

parser = argparse.ArgumentParser(
                    prog='Histogram Generator',
                    description='Plots histograms all all images')

parser.add_argument('-d', '--directory', required=True, help="Input directory")
parser.add_argument('-o', '--outputdir', default='histograms', help="Output directory")

args = parser.parse_args()

# List of supported image file extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')

# Initialize an empty list to hold all found images
all_images = []

if not os.path.exists(args.outputdir):
    os.mkdir(args.outputdir)
else:
    shutil.rmtree(args.outputdir)
    os.mkdir(args.outputdir)

pinput = os.path.abspath(args.directory)
poutput = os.path.abspath(args.outputdir)
plen = len(pinput)

print(pinput)
print(poutput)

# Walk through the directory and its subdirectories
for root, dirs, files in os.walk(args.directory):
    for file in files:
        # Check if the file has a supported image extension
        if file.lower().endswith(image_extensions):
            root_path = root[plen::]
            new_dir = poutput + root_path
            new_file = new_dir + '/' + file.split('.')[0] + '.jpg'
            img_file = root + '/' + file
            print(f'Processing {img_file} Saving to {new_file}')
            img = Image.open(img_file)
            #img = cv2.imread(img_file)
            hist = img.histogram()
            if len(hist) == (256 * 3):
                # Plot histograms on the image
                f, ax = plt.subplots(3, 1, sharex=True, sharey=True)
                ax[0].plot(hist[0:256], 'r')
                ax[1].plot(hist[256:512], 'g')
                ax[2].plot(hist[512::], 'b')
                ax[0].set_title('Histograms')
                ax[2].set_xlabel('Pixel Value')
                ax[0].set_ylabel('Counts')
                ax[1].set_ylabel('Counts')
                ax[2].set_ylabel('Counts')
                ax[0].grid()
                ax[1].grid()
                ax[2].grid()
            else:
                # Plot histograms on the image
                plt.plot(hist, 'r')
                plt.title('Histograms')
                plt.xlabel('Pixel Value')
                plt.ylabel('Counts')
                plt.grid()

            # Save the merged histogram image to output directory
            if not os.path.exists(os.path.dirname(new_file)):
                os.makedirs(os.path.dirname(new_file))
            plt.savefig(new_file)
            plt.close()




