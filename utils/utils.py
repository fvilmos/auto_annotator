'''************************************************************************** 
Author: fvilmos, https://github.com/fvilmos
***************************************************************************'''

import numpy as np
import cv2

def img_format(img,bbox=None,out_size=(256,256)):
    """
    Format image, takeing the longest dimention (w,h)
    Args:
        img (_type_): input image RGB
        bbox (_type_, optional): bounding box [x1,y1,x2,y2]. Defaults to None.
        out_size (tuple, optional): out out img dimension. Defaults to (256,256).

    Returns:
        _type_: new image, adjusted bbox
    """
    ih,iw,ch = img.shape
    max_size = max(ih,iw)
    ratio = max_size / out_size[0]
    h_n = int(ih/ratio)
    w_n = int(iw/ratio)

    img_n = cv2.resize(img, (w_n,h_n), interpolation=cv2.INTER_LANCZOS4)
    img_res = np.zeros([out_size[0],out_size[1],ch], dtype=np.uint8)
    
    img_res[:h_n,:w_n] = img_n

    bbox_n = None
    if bbox is not None:
        x,y,w,h = bbox
        bbox_n = [int((x-0.5*w)*iw/ratio), int((y-0.5*h)*ih/ratio), int(w*iw/ratio), int(h*ih/ratio)]
    
    return img_res, bbox_n

def puttext_bg(img,text='',position=(10,160), font_type=None, font_size=0.4, font_color=[0,255,0],bg_color=[0,0,0],font_thickness=1):
    """
    Text with background color
    Args:
        img: input image
        text (str, optional): Defaults to ''.
        position (tuple, optional): Defaults to (10,160).
        font_type (_type_, optional): Defaults to None.
        font_size (float, optional): Defaults to 0.4.
        font_color (list, optional): Defaults to [0,255,0].
        bg_color (list, optional): Defaults to [0,0,0].
        font_thickness (int, optional): Defaults to 1.
    """

    if font_type is None:
        font_type = cv2.FONT_HERSHEY_SIMPLEX
    (t_w,t_h),_ = cv2.getTextSize(text, font_type, font_size, font_thickness)
    cv2.rectangle(img, position, (position[0] + t_w, position[1] + t_h), bg_color, -1)
    cv2.putText(img,text ,(position[0], int(position[1]+t_h+font_size-1)),font_type,font_size,font_color,font_thickness)

def plot_bbox(img, label="", bbox=[10,10,30,30], color=[0,255,0], shape=(256,256), factor=1, style="xywh"):
    """
    Plot a bounding box over an image.
    Bounding box format is x,y,w,h, chanve it to "xyxy" if needed
    """
    img = cv2.resize(img,shape)
    bbox *= factor
    x0,y0,x1,y1 = bbox
    
    if style=="xywh":
        cv2.rectangle(img,(int(x0-x1//2),int(y0-y1//2)),(int(x0+x1//2),int(y0+y1//2)),color,2)
    else:
        cv2.rectangle(img,(int(x0),int(y0)),(int(x1),int(y1)),color,2)
    puttext_bg(img,text=str(label),position=(int(x0+5),int(y0-5)),bg_color=[0,0,0])
    return img
    