""" color_engine.py for the color processiong."""
from abc import ABC, abstractmethod
import numpy as np
import cv2
from pathlib import Path
import color_utils
import image_utils 

class HistogramStrategy(ABC):
    """ color startegy interface base"""

    @abstractmethod
    def calculate(self, img: np.ndarray, mask: np.ndarray | None) -> np.ndarray | tuple[np.ndarray, ...]:
        """
        computing the color histogram of the single image

        Args:
            img: BRG(h, w, 3) or BGRA(h, w, 4)
            mask: optional
        Returns:
            color histogram: 
                - multiple channels
                - single channel
        """
        pass
    @abstractmethod
    def calculate_mean_channel(self, img: np.ndarray, mask: np.ndarray | None):
        """
        computing  the mean of the one channel

        Args:
            img: BRG(h, w, 3) or BGRA(h, w, 4)
            mask: optional
        Returns: the mean of the pixels
        """
        pass
  
    @property
    @abstractmethod
    def channel_names(self) -> list[str]:
        """Return the channel name list"""
        pass
 
class RGBHistogramStrategy(HistogramStrategy):
    """RGB color space using the order of BGR"""

    def calculate(self, img: np.ndarray, mask: np.ndarray | None) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        hists = []
        for channel in range(3):
            hist_bgr = cv2.calcHist([img], [channel], mask, [256], [0, 256])
            hists.append(hist_bgr)

        return tuple(hists)
    
    def calculate_mean_channel(self, img: np.ndarray, mask: np.ndarray | None):
        return cv2.mean(img, mask)



    @property
    def channel_names(self) -> list[str]:
        return ['B', 'G', 'R']

class HueOnlyHistogramStrategy(HistogramStrategy):
    """hue only for color distribution analysis"""

    def calculate(self, img: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return cv2.calcHist([img_hsv], [0], mask, [180], [0, 180])
    
    def calculate_mean_channel(self, img: np.ndarray,img_mask: np.ndarray | None) -> float:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0:1]
        hue_remain = np.ma.masked_array(img_hsv,mask = (img_mask == 0))
        linear_mean = np.mean(hue_remain)
        return linear_mean

    def calculate_mean_circle(self, img: np.ndarray, img_mask: np.ndarray | None) -> float:
        img_hue =  cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0:1]
        hue_remain = np.ma.masked_array(img_hue,mask = (img_mask == 0))
        mean_ring = color_utils.circular_mean(hue_remain)
        return mean_ring


    @property    
    def channel_names(self) ->list[str]:
        return ['Hue']

class LABHistogramStrategy(HistogramStrategy):
    """ CIELAB color space, return the color histogram of L, a, b channel"""

    def calculate(self, img: np.ndarray, mask: np.ndarray | None) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        hists = []
        for channel in range(3):
            hist_lab = cv2.calcHist([img_lab], [channel], mask, [256], [0,256])
            hists.append(hist_lab)
        return tuple(hists)

    def calculate_mean_channel(self, img: np.ndarray, mask: np.ndarray | None) -> float:
        return 0.0

    @property
    def channel_names(self) -> list[str]:
        return ['L', 'a', 'b']

class GrayHistogramStrategy(HistogramStrategy):
    """ Return the color histogram of the gray graph"""

    def calculate(self, img: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.calcHist[img_grey, [0], mask, [256], [0, 256]]

    def calculate_mean_channel(self, img: np.ndarray, mask: np.ndarray | None) -> float:
        return 0.0

    @property
    def channel_name(self):
        return ['Gray']

class ColorHistogramEngine:
    """ color histogram , support multiple color space

    motivation: why not directly call the OpenCV cv2.calcHist function
    1. transparent background(alpha channel):
    considering  calculation of the alpha channel
    2. color space convert:
    automative convert through configuration
    

    Attributes:
        color_space(str): color space(including 'rgb', 'hsv')
        ignore_transparent(bool): calculating the color distribution with/without the alpha channel
    
    Examples:
        >>> # using color space string
        >>> engine = ColorHistogramEngine('rgb')

        >>># using strategy
        >>> hue_strategy = HueOnlyStrategy()
        >>> engine = ColorHistogramEngine(hue_strategy)
    """
    _STRATEGY_MAP = {
        'rgb': RGBHistogramStrategy,
        'hue': HueOnlyHistogramStrategy,
        'lab': LABHistogramStrategy,
        'gray': GrayHistogramStrategy
    }

    def __init__(
                self, 
                strategy: str|HistogramStrategy, 
                ignore_transparent: bool = True):
        if isinstance(strategy, str):
            strategy_cls = self._STRATEGY_MAP.get(strategy.lower())
            if strategy_cls is None:
                raise ValueError(f"unknown strategy:{strategy}, optional: {list(self._STRATEGY_MAP.keys())}")
            self.strategy = strategy_cls()
        else:
            self.strategy = strategy
        self.ignore_transparent = ignore_transparent

    def calculate_hist(self, img_raw: str|Path|np.ndarray):
        """computing the color histogram of a single or multiple image
        Parameters:
            imgs: various type supported:
                - str: image path
                - np.ndarray: image array from openCV.imread(including BGR or BGRA)
        Returns:
            list: the color histogram of the image
                - multiple channels: e.g. rgb
                - single channels: e.g. hue-only        
        """
        img_bgr, img_mask = image_utils.load_img(img_raw)
        if img_bgr is None:
            return None
        if self.ignore_transparent:
            hist = self.strategy.calculate(img_bgr, img_mask)
        else:
            hist = self.strategy.calculate(img_bgr, None)
        return hist   

    def calculate_hists(self, imgs: list[str|Path|np.ndarray]):
        """ computing the color histogram of a single or multiple image
        Parameters:
            imgs: various number of images, type supported:
                - str: image path
                - np.ndarray: image array from openCV.imread(including BGR or BGRA)
        Returns:
            list: the color histogram of each image
                - multiple channels: e.g. rgb
                - single channels: e.g. hue-only        

        """

        results = []  
        for img_raw in imgs:
            hist = self.calculate_hist(img_raw)
            results.append(hist)
        return results
        
    def calculate_linear_mean(self, img_raw: str|Path|np.ndarray):
        """ computing the mean of the channel for single image
        Parameters:
            imgs: type supported:
                - str: image path
                - np.ndarray: image array from openCV.imread(including BGR or BGRA)
        Returns: the mean of one channel
       
        """
        img_bgr, img_mask = image_utils.load_img(img_raw)
        if img_bgr is None:
            return None
        if self.ignore_transparent:
            mean_channel= self.strategy.calculate_mean_channel(img_bgr, img_mask)
        else:
            mean_channel = self.strategy.calculate_mean_channel(img_bgr, None)
        return mean_channel   

    def calculate_linear_means(self, imgs: list[str|Path|np.ndarray]):
        """ computing the mean of the channel for single or multiple images
        Parameters:
            imgs: various number of images, type supported:
                - str: image path
                - np.ndarray: image array from openCV.imread(including BGR or BGRA)
        Returns: the mean of one channel
       
        """
        results = []  
        for img_raw in imgs:
            channel_mean =  self.calculate_linear_mean(img_raw)
            results.append(channel_mean)
        return results

        return reuslts
    
    def calculate_linear_mean_total(self, imgs: list[str|np.ndarray]):
        img_bgr, img_mask = image_utils.load_imgs_flatten(imgs)
        if img_bgr is None:
            return None
        mean_channel = self.strategy.calculate_mean_channel(img_bgr, img_mask)
        return mean_channel 


    @property
    def channel_names(self) -> list[str]:
        """ Return channel name"""
        return self.strategy.channel_names

class ColorAnalysisFacade:
    """ color anglysis based on the color distribution """

    def __init__(self, 
                imgs: str|np.ndarray|list[str|np.ndarray], 
                strategy: str|HistogramStrategy, 
                ignore_transparent: bool = True):

        self.analyzer = ColorHistogramEngine(strategy, ignore_transparent)
        self.img_bgr = None
        self.img_mask = None
        if isinstance(imgs, list):
            self.img_bgr, self.img_mask = image_utils.load_imgs(imgs)
        else:
            self.img_bgr, self.img_mask = image_utils.load_img(imgs)

    def get_hue_circular_mean(self, input_scale = 'opencv', output_scale = 'opencv'):
        hue_mean = self.analyzer.strategy.calculate_mean_circle(self.img_bgr, self.img_mask)
        return hue_mean

    def get_hue_circular_means(self, input_scale = 'opencv', output_scale = 'opencv'):
        if self.img_bgr is None:
            return None
        hue_means = []
        n = len(self.img_bgr)
        for i in range(n):
            hue_mean = self.analyzer.strategy.calculate_mean_circle(self.img_bgr[i], self.img_mask[i])
            hue_means.append(hue_mean)

        return hue_means

    def get_hue_circular_mean_total(self, input_scale = 'opencv', output_scale = 'opencv'):
        img_bgr_flat, img_mask_flat = image_utils.imgs_flatten(self.img_bgr, self.img_mask)
        hue_mean = self.analyzer.strategy.calculate_mean_circle(img_bgr_flat, img_mask_flat)
        return hue_mean

    def get_hue_linear_mean(self, input_scale = 'opencv', output_scale = 'opencv'):
        hue_mean = self.analyzer.strategy.calculate_mean_channel(self.img_bgr, self.img_mask)
        return hue_mean

    def get_hue_linear_means(self, input_scale = 'opencv', output_scale = 'opencv'):
        
        if self.img_bgr is None:
            return None

        hue_means = []
        n = len(self.img_bgr)
        for i in range(n):
            hue_mean = self.analyzer.strategy.calculate_mean_channel(self.img_bgr[i], self.img_mask[i])
            hue_means.append(hue_mean)

        return hue_means
    
    def get_hue_linear_mean_total(self, input_scale = 'opencv', output_scale = 'opencv'):
        img_bgr_flat, img_mask_flat = image_utils.imgs_flatten(self.img_bgr, self.img_mask)
        hue_mean = self.analyzer.strategy.calculate_mean_channel(img_bgr_flat, img_mask_flat)
        return hue_mean

    def get_hue_circular_mean_alpha(self, input_scale = 'opencv', output_scale = 'opencv'):
        hue_mean = self.analyzer.strategy.calculate_mean_circle(self.img_bgr, None)
        return hue_mean

    def get_hue_circular_means_alpha(self, input_scale = 'opencv', output_scale = 'opencv'):
        if self.img_bgr is None:
            return None
        hue_means = []
        n = len(self.img_bgr)
        for i in range(n):
            hue_mean = self.analyzer.strategy.calculate_mean_circle(self.img_bgr[i], None)
            hue_means.append(hue_mean)

        return hue_means

    def get_hue_circular_mean_total_alpha(self, input_scale = 'opencv', output_scale = 'opencv'):
        img_bgr_flat, img_mask_flat = image_utils.imgs_flatten(self.img_bgr, self.img_mask)
        hue_mean = self.analyzer.strategy.calculate_mean_circle(img_bgr_flat, None)
        return hue_mean

    def get_hue_linear_mean_alpha(self, input_scale = 'opencv', output_scale = 'opencv'):
        hue_mean = self.analyzer.strategy.calculate_mean_channel(self.img_bgr, None)
        return hue_mean

    def get_hue_linear_means_alpha(self, input_scale = 'opencv', output_scale = 'opencv'):
        
        if self.img_bgr is None:
            return None

        hue_means = []
        n = len(self.img_bgr)
        for i in range(n):
            hue_mean = self.analyzer.strategy.calculate_mean_channel(self.img_bgr[i], None)
            hue_means.append(hue_mean)

        return hue_means

    def get_hue_linear_mean_total_alpha(self, input_scale = 'opencv', output_scale = 'opencv'):
        img_bgr_flat, img_mask_flat = image_utils.imgs_flatten(self.img_bgr, self.img_mask)
        hue_mean = self.analyzer.strategy.calculate_mean_channel(img_bgr_flat, None)
        return hue_mean

    def get_hists_total(self, input_scale = 'opencv', output_scale = 'opencv'):
        img_bgr_flat, img_mask_flat = image_utils.imgs_flatten(self.img_bgr, self.img_mask)
        hist = self.analyzer.strategy.calculate(img_bgr_flat, img_mask_flat)
        return hist

    def get_hists(self, input_scale = 'opencv', output_scale = 'opencv'):
        if self.img_bgr is None:
            return None

        hue_hists = []
        n = len(self.img_bgr)
        for i in range(n):
            hue_hist = self.analyzer.strategy.calculate(self.img_bgr[i], self.img_mask[i])
            hue_hists.append(hue_hist)

        return hue_hists