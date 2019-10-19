import cv2 as cv
import numpy as np

class CVHelp():
    """
    This is a Opencv helper class. It handles displaying data and resizing
    image so they fit on the screen. It also handles displaying multiple
    images in a grid
    """
    def __init__(self, max_width = 1800, max_height = 900):
        """
        img -- thei mage
        max_width -- maximum width
        max_height -- maximum height
        """
        self.max_width = max_width
        self.max_height = max_height

    def resize(self, img, max_width, max_height):
        """Image resize helper. This reszie function will intelligently resize
        the image to maintain the aspect ratio but resize to a max width and height.
        Keyword arguments:
        img -- thei mage
        max_width -- maximum width
        max_height -- maximum height
        """
        w = img.shape[1]
        h = img.shape[0]

        if w > max_width:
            ratio = max_width / w
            new_height = int(h * ratio)
            img = cv.resize(img, (max_width, new_height))

        w = img.shape[1]
        h = img.shape[0]
        if h > max_height:
            ratio = max_height / h
            new_width = int(w * ratio)
            img = cv.resize(img, (new_width, max_height))

        return img

    def display(self, img, title = 'img', delay = -1, clear_windows = False):
        """Helper function to display. It will display an image and resize it
        to keep it below a certain self.max_width and height.
        Keyword arguments:
        img -- the image to display
        title -- the title for the window
        delay -- how long to wait for a key
        """
        if clear_windows:
            cv.destoryAllWindows()
        img = self.resize(img, self.max_width, self.max_height)
        cv.imshow(title, img)
        key = ''
        while key != ord('q'):
            key = cv.waitKey(delay)


    def display_channels(self, img):
        """Helper function to display the separate color channels
        Keyword arguments:
        img -- the image to display
        """
        cv.imshow("Red", img[:,:,2])
        cv.imshow("Green", img[:,:,1])
        cv.imshow("Blue", img[:,:,0])
        cv.waitKey(1)

    def generate_grid(self, imgs, grid_width, grid_height, labels = None):
        """Helper function to generate a grid of images. It is currently
        hard coded to display a grid of 2 images wide by three images high (6 images)
        It uses the grid_width and grid_height as a maximum for ALL images
        Keyword arguments:
        img -- the image to display
        grid_width -- the max width of the entire grid
        grid_height -- the max height of the entire grid
        labels -- text labels to place on each grid image
        """
        new_img = np.zeros((self.max_height, self.max_width, 3), np.uint8)

        img_width = int(self.max_width // grid_width)
        img_height = int(self.max_height // grid_height)

        x = 0
        y = 0
        for i, img in enumerate(imgs):
            ri = self.resize(img, img_width, img_height)
            if len(ri.shape) == 2:
                # Convert to color
                ri = cv.cvtColor(ri, cv.COLOR_GRAY2BGR)
            xstart = x * img_width
            ystart = y * img_height
            new_img[ystart:ystart + ri.shape[0], xstart:xstart + ri.shape[1], :] = ri
            if labels is not None:
                #print("i: {:} -- x/y: {:}/{:} -- label: {:}".format(i, xstart, ystart, labels[i]))
                cv.putText(new_img, labels[i], (xstart + 10, ystart + 20), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))

            x += 1
            if x >= grid_width:
                x = 0
                y += 1

        return new_img

    def generate_2dhist(self, image, channels=[0, 1], hist_scale = 10, Mask=None, histSize=[180, 256], ranges=[0,180,0,256]):
        """Helper function to generate a 2D histogram that can be displayed
        Keyword arguments:
        image -- the image to plot
        channels -- which channels to plot
        Mask -- Mask to use for the histogram
        histSize -- The number of bins for each channel
        ranges -- The min and max range for each channel
        """
        hsv_map = np.zeros((histSize[0], histSize[1], 3), np.uint8)
        h, s = np.indices(hsv_map.shape[:2])
        hsv_map[:,:,0] = h
        hsv_map[:,:,1] = s
        hsv_map[:,:,2] = 255
        hsv_map = cv.cvtColor(hsv_map, cv.COLOR_HSV2BGR)

        #dark = hsv[...,2] < 32
        #hsv[dark] = 0
        hist = cv.calcHist(image, channels, None, histSize, ranges)
        hist = np.clip(hist*0.005*hist_scale, 0, 1)
        vis = hsv_map*hist[:,:,np.newaxis]
        vis = vis.astype('uint8')
        return vis

    def generate_backproj_mask(self, image_to_mask, image_to_generate_mask):
        """This function takes an image_to_generate_mask, uses it to generate
        an HSV mask that gets applied to image_to_mask.
        # calculating object histogram

        Keyword arguments:
        image_to_mask -- the image to apply the back project mask to
        image_to_generate_mask -- the image to use as the HS back project mask
        """
        hsvt = cv.cvtColor(image_to_mask, cv.COLOR_BGR2HSV)
        roi  = cv.cvtColor(image_to_generate_mask, cv.COLOR_BGR2HSV)

        # Generate histograms
        #hist_target = cv.calcHist([hsvt], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist_roi    = cv.calcHist([roi],  [0, 1], None, [180, 256], [0, 180, 0, 256])


        # normalize histogram and apply backprojection
        cv.normalize(hist_roi,hist_roi,0,255,cv.NORM_MINMAX)
        dst = cv.calcBackProject([hsvt],[0,1],hist_roi,[0,180,0,256],100)
        cv.imwrite('dst.bmp', dst)
        # Now convolute with circular disc
        disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        cv.filter2D(dst,-1,disc,dst)
        # threshold and binary AND
        ret,thresh = cv.threshold(dst,50,255,0)
        cv.imwrite('thresh.bmp', thresh)
        thresh = cv.merge((thresh,thresh,thresh))
        cv.imwrite('merge.bmp', thresh)
        res = cv.bitwise_and(image_to_mask,thresh)
        cv.imwrite('res.bmp', res)
        #res = np.vstack((image_to_mask, thresh, res))
        return image_to_mask, thresh, ret

    def info(self, image):
        """This prints info on the image:
        width x height x channels - image type

        Keyword arguments:
        image -- the image to get info on
        """
        print("{} x {} x {} - {}".format(image.shape[1], image.shape[0], image.shape[2], image.dtype))