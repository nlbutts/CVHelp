import cv2
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
            img = cv2.resize(img, (max_width, new_height))

        w = img.shape[1]
        h = img.shape[0]
        if h > max_height:
            ratio = max_height / h
            new_width = int(w * ratio)
            img = cv2.resize(img, (new_width, max_height))

        return img


    def display(self, img, title = 'img', delay = 1):
        """Helper function to display. It will display an image and resize it
        to keep it below a certain self.max_width and height.
        Keyword arguments:
        img -- the image to display
        title -- the title for the window
        delay -- how long to wait for a key
        """
        img = self.resize(img, self.max_width, self.max_height)
        cv2.imshow(title, img)
        cv2.waitKey(delay)

    def display_channels(self, img):
        """Helper function to display the separate color channels
        Keyword arguments:
        img -- the image to display
        """
        cv2.imshow("Red", img[:,:,2])
        cv2.imshow("Green", img[:,:,1])
        cv2.imshow("Blue", img[:,:,0])
        cv2.waitKey(1)

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
                ri = cv2.cvtColor(ri, cv2.COLOR_GRAY2BGR)
            xstart = x * img_width
            ystart = y * img_height
            new_img[ystart:ystart + ri.shape[0], xstart:xstart + ri.shape[1], :] = ri
            if labels is not None:
                #print("i: {:} -- x/y: {:}/{:} -- label: {:}".format(i, xstart, ystart, labels[i]))
                cv2.putText(new_img, labels[i], (xstart + 10, ystart + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))

            x += 1
            if x >= grid_width:
                x = 0
                y += 1

        return new_img
