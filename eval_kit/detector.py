from abc import ABC, abstractmethod


class FaceDetectorBase(ABC):
    def __init__(self):
        """
        Participants should define their own initialization process.
        During this process you can set up your network. The time cost for this step will
        not be counted in runtime evaluation
        """

    @abstractmethod
    def process_image(self, image):
        """
        Process one image, the evaluation toolkit will measure the runtime of every call to this method.
        The time cost will include any thing that's between the image input to the final bounding box output.
        The image will be given as a numpy array in the shape of (H, W, C) with dtype np.uint8.
        The color mode of the image will be **BGR**.
        :param image: a numpy array with dtype=np.uint8 representing a image of (H, W, C) shape.
        :return: a numpy array of bounding boxes in the following format
            [
                [left, top, width, height, confidence],
                ...
            ], dtype=np.float32
            The bounding box locations should be relative to the image sizes, i.e in [0, 1]

        """
        pass