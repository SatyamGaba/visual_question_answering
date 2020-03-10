import os, sys
import argparse
# from PIL import Image
import cv2
from tqdm import tqdm


def resize_image(image, size):
    """Resize an image to the given size."""
#     frame = cv2.resize(frame, size, interpolation = cv2.INTER_AREA)
#     return image.resize(size, Image.ANTIALIAS)
    return cv2.resize(image, size, interpolation = cv2.INTER_AREA)


def resize_images(input_dir, output_dir, size):
    """Resize the images in 'input_dir' and save into 'output_dir'."""
    for idir in os.scandir(input_dir):
        if not idir.is_dir():
            continue
        if not os.path.exists(output_dir+'/'+idir.name):
            os.makedirs(output_dir+'/'+idir.name)    
        images = os.listdir(idir.path)
        n_images = len(images)
        for iimage, image in tqdm(enumerate(images)):
#             try:
#             with open(os.path.join(idir.path, image), 'r+b') as f:
#               with Image.open(f) as img:
            f = os.path.join(idir.path, image)
            img = cv2.imread(f)
            img = resize_image(img, size)
#             img.save(os.path.join(output_dir+'/'+idir.name, image), img.format)
            cv2.imwrite(os.path.join(output_dir+'/'+idir.name, image),img)
#             except(IOError, SyntaxError) as e:
#                 pass
            if (iimage+1) % 10000 == 0:
                sys.stdout.write("[{}/{}] resized and saved.".format(iimage+1, n_images))
            
            
def main(args):

    input_dir = args.input_dir
    output_dir = args.output_dir
    image_size = (args.image_size, args.image_size)
    resize_images(input_dir, output_dir, image_size)

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='/run/media/hoosiki/WareHouse3/mtb/datasets/VQA/Images',
                        help='directory for input images (unresized images)')

    parser.add_argument('--output_dir', type=str, default='/run/media/hoosiki/WareHouse3/mtb/datasets/VQA/Resized_Images',
                        help='directory for output images (resized images)')

    parser.add_argument('--image_size', type=int, default=224,
                        help='size of images after resizing')

    args = parser.parse_args()

    main(args)
