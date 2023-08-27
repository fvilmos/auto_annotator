'''************************************************************************** 
Auto Annotator

Automated labeling tool
Author: fvilmos, https://github.com/fvilmos
***************************************************************************'''

import cv2
import numpy as np
import torch
import os
import glob
import json
from datetime import datetime
import uuid
from utils import utils
from groundingdino.util.inference import Model

jf = open(".\\utils\\cfg.json",'r')
cfg_data=json.load(jf)
data_dict = {}
count = 0

if cfg_data["FORCE_CPU"] == 1:
    torch.cuda.is_available = lambda : False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ("GPU available:", DEVICE)

# prepare model for inference
model_gd = Model(model_config_path=cfg_data["CONFIG_PATH"], model_checkpoint_path=cfg_data["MODEL_PATH"])

# create folders to store data
if cfg_data["WRITE_TYPE"] !=1:
    def create_unique_folder():
        now = datetime.now()
        ts = int(now.timestamp()*10000)
        return str(ts) + "\\"

    unique_folder = create_unique_folder()
    out_path = cfg_data['BASE_PATH'] + unique_folder

    if os.path.exists(out_path) == False:
        os.makedirs(out_path)

    f = open(out_path + cfg_data['OUT_FILE'],'a')


class DetectInputSource():
    '''
    This class decides the type of the data input, and transforms it from data-consumption
    '''
    def __init__(self, source) -> None:
        self.source = source
        self.file_name = str(source)
        self.isImg = False
        self.process_file = None
        self.count = 0

        if isinstance(source, str)==True:
            self.process_file = glob.glob(cfg_data["BASE_PATH"]+cfg_data["SOURCE"], recursive=True)
            if len(self.process_file) == 0: 
                print ("No file to process!!!")
                exit(0)

            self.file_name  = self.process_file [cfg_data["FILE_INDEX"]]
            print ("SOURCE LIST:", self.process_file)
            print ("SOURCE:", cfg_data["BASE_PATH"]+cfg_data["SOURCE"])
            
            # test if input is an image
            if sum([1 if a in source else 0 for a in cfg_data["ACCEPTED_IMAGE_TYPES"]]) > 0:
                self.cap = 1
                self.isImg  = True
            else:
                self.cap = cv2.VideoCapture(self.file_name)
        else:
            print ( "SOURCE:", int(cfg_data["SOURCE"]))
            self.cap = cv2.VideoCapture(int(cfg_data["SOURCE"]))
    
    def device_opened(self):
        ret = False
        if isinstance(self.source, int):
            ret = self.cap.isOpened()
        return ret
    def read(self):
        if self.isImg==False: 
            _,img = self.cap.read()
        else:
            if self.count < len(self.process_file):
                img = cv2.imread(self.process_file[self.count])
                self.count +=1
            else:
                exit(0)
        return img


device_test = DetectInputSource(cfg_data["SOURCE"])

while(device_test):
    bbox_list = []
    oimg = device_test.read()
    img = oimg.copy()

    # apply formating strategy
    if cfg_data['OUT_IMG_SIZE'] is not None and isinstance(cfg_data['OUT_IMG_SIZE'],list):
        if  int(cfg_data["OUT_RESHAPE"])==1:
            img,_ = utils.img_format(img,None,cfg_data["OUT_IMG_SIZE"])
        else:
            img = cv2.resize(img, dsize=cfg_data["OUT_IMG_SIZE"], interpolation=cv2.INTER_LANCZOS4)

    img_src = img.copy()
    img_rgb = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

    detections = model_gd.predict_with_classes(image=img_rgb,classes=list(cfg_data["SEARCH_PATTERN"].values()), box_threshold=cfg_data["CONFIDENCE_THRESHOLD"], text_threshold=0.15)

    # collect all bounding box detections
    for i in range (len(detections.xyxy)):
        ldata = {}
        bbox = detections.xyxy[i]
        confidence = np.round(detections.confidence[i],2)
        
        # test confidence ignore if required
        conf_th = cfg_data['CONFIDENCE_THRESHOLD']
        if isinstance(conf_th, str)==False:
            if confidence < conf_th:
                continue
        
        # get labels
        label = detections.class_id[i]
        label_name = list(cfg_data["SEARCH_PATTERN"].keys())[int(label)]
        
        # visualize
        img = utils.plot_bbox(img, bbox=bbox, label=str(label_name)+":"+str(confidence), style="xyxy",shape=(img.shape[1], img.shape[0]))
        
        # prepare data to store
        x0,y0,x1,y1 = bbox
        ldata['bbox'] = list([int(x0), int(y0), int(x1), int(y1)])
        ldata['label'] = str(label_name)
        ldata['confidence'] = str(int(confidence*100))
        ldata['style']= 'xyxy'

        bbox_list.append(ldata)
    
    # store only on detections
    if (len(detections.xyxy)) > 0 and (len(bbox_list)>0):
        # construct data_dict
        id = str(uuid.uuid4().int)
        data_dict['id'] = id
        data_dict['source'] = str(device_test.file_name)
        data_dict['file'] = str(count) +"_"+ str(id) + str(cfg_data["OUT_FILE_EXTENSION"])
        data_dict['orig_img_shape'] = list(img.shape)
        if cfg_data['OUT_IMG_SIZE'] is not None and isinstance(cfg_data['OUT_IMG_SIZE'],list):
            data_dict['work_img_shape'] = list([cfg_data['OUT_IMG_SIZE'][0],cfg_data['OUT_IMG_SIZE'][1],img.shape[-1]])
        else:
            data_dict['work_img_shape'] = list(img.shape)
        data_dict['b_bbox'] = list(bbox_list)
        count +=1
        
        # write strategy
        if cfg_data["WRITE_TYPE"] == 0:
            print ("data:", data_dict)
            d_dict_data = json.dumps(data_dict)

            # save to meta file
            f.write(d_dict_data)
            f.write('\n')
            cv2.imwrite(out_path + data_dict['file'],img_src)
        elif cfg_data["WRITE_TYPE"] == 1:
            pass
        elif cfg_data["WRITE_TYPE"] == 2:
            file_name = str(count) +"_"+ str(id) + "_a" + str(cfg_data["OUT_FILE_EXTENSION"])
            cv2.imwrite(out_path + file_name,img)
        elif cfg_data["WRITE_TYPE"] == 3:

            file_name = str(count) +"_"+ str(id) + "_a" + str(cfg_data["OUT_FILE_EXTENSION"])
            cv2.imwrite(out_path + file_name,img)
            
            print ("data:", data_dict)
            d_dict_data = json.dumps(data_dict)

            # save to meta file
            f.write(d_dict_data)
            f.write('\n')
            cv2.imwrite(out_path + data_dict['file'],img_src)

    # exit on ESC
    k = cv2.waitKey(1)
    if k == 27:
        exit()

    # show detections
    cv2.imshow("Test_detections",img)

cv2.destroyAllWindows()