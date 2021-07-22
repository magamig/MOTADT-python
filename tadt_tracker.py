#20190512 by zikun
#定义tracker类
import math

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from matplotlib.patches import Rectangle

from backbone_v2 import VGG, build_vgg16
from feature_utils_v2 import feature_selection  # for visualization
from feature_utils_v2 import (features_selection, generate_patch_feature,
                              get_subwindow, get_subwindow_feature,
                              resize_tensor, round_python2,
                              get_frame_features)
from image_loader import default_image_loader
from siamese import SiameseNet
from taf import taf_model, taf_model_diff
from tracking_utils import (cal_window_size, calculate_scale, fuse_feature,
                            generate_2d_window)

# matplotlib.use('Qt5Agg') # fix for MacOS bug - "AttributeError: 'FigureManagerMac' object has no attribute 'window'"

class Tadt_Tracker(object):
    def __init__(self, config, model = None, display = True, device = 'cpu', name = 'TADT', model_from_mat = True):
        """
        args:
            target_location - [x1,y1,w,h], 1-index for OTB benchmark
            TODO: model_from_mat
        """
        super(Tadt_Tracker, self).__init__()
        #---------------trackers parameters initialization--------------------------
        self.name = name
        self.config = config
        self.display = display
        self.device = device
        self.rescale = 1
        self.results = []
        self.model_from_mat = model_from_mat

        #-------------model initialization--------------------
        if model is None:
            self.model = build_vgg16(self.config).to(self.device)
        else:
            self.model = model.to(self.device)
        self.model.train()
        self.siamese_model = SiameseNet().to(self.device)
        self.toc = 0

    def initialize_tadt(self, img_path, target_loc, visualize = False):

        #------------sequence parameters initialization----------------------------
        img = default_image_loader(img_path)#<class 'numpy.ndarray'> [height, width, channel]
        self.target_location = target_loc[0]
        origin_target_size = math.sqrt(self.target_location[2] * self.target_location[3]) #area of bounding box
        origin_image_size = img.shape[0: 2][::-1] # [width,height]
        if origin_target_size > self.config.MODEL.MAX_SIZE:
            self.rescale = self.config.MODEL.MAX_SIZE / origin_target_size
        elif origin_target_size < self.config.MODEL.MIN_SIZE:
            self.rescale = self.config.MODEL.MIN_SIZE / origin_target_size

        #----------------scale image cv2 numpy.adarray---------------
        image = cv2.resize(
                img,
                tuple((np.ceil(np.array(origin_image_size) * self.rescale)).astype(int)),
                interpolation=cv2.INTER_LINEAR
        )
        
        #----------------shift image---------------
        shift = np.float32([
	        [1, 0, 1],
	        [0, 1, 1]
        ])
        shifted_img = cv2.warpAffine(image, shift, (image.shape[1], image.shape[0]))

        #------scaled target location, get position and size [x1,y1,width,height]------
        self.target_location = round_python2(np.array(self.target_location) * self.rescale)-np.array([1,1,0,0])#0-index
        target_size = self.target_location[2: 4]# [width, height]
        image_size = image.shape[0:2]# [height, width]
        search_size, ratio = cal_window_size(self.config.MODEL.MAX_SIZE, image_size, 2, self.config.MODEL.TOTAL_STRIDE)
        self.input_size = np.array([search_size, search_size])

        #------------First frame processing--------------------
        self.srch_window_location = cal_srch_window_location(self.target_location, search_size)
        self.srch_window_location2 = cal_srch_window_location(round_python2(np.array(target_loc[1]) * self.rescale)-np.array([1,1,0,0]), search_size)
        
        features = get_subwindow_feature(self.model, image, self.srch_window_location2, self.input_size) #two tensors, one from each Conv layer
        features2 = get_subwindow_feature(self.model, image, self.srch_window_location, self.input_size) #two tensors, one from each Conv layer
        features3 = get_subwindow_feature(self.model, shifted_img, self.srch_window_location, self.input_size) #two tensors, one from each Conv layer
        
        
        #----------- crop the target exemplar from the feature map------------------
        patch_features, patch_locations = generate_patch_feature(target_size[::-1], self.srch_window_location, features)
        self.feature_pad = 2
        self.b_feature_pad = int(self.feature_pad / 2)
        self.filter_sizes = [torch.tensor(feature.shape).numpy() for feature in patch_features]


        #-------------compute the indices of target-aware features----------------
        #self.feature_weights, self.balance_weights = taf_model([features, features2], self.filter_sizes, self.device)
        self.feature_weights, self.balance_weights = taf_model_diff([features, features2], 1, self.device)

        #-------------select the target-awares features---------------------------
        self.exemplar_features = features_selection(patch_features, self.feature_weights, self.balance_weights, mode = 'reduction')
        #self.exemplar_features = fuse_feature(patch_features)
        
        #-------------calculate global average pooling of exemplar features-------------------
        kernel_size = self.exemplar_features[0].shape[1:3]
        self.exemplar_features_gap = nn.AvgPool2d(kernel_size)(self.exemplar_features[0])

        #------------visualization------------------------------------------------
        if self.display:
            self.prepare_visualize()
            self.visualization(img, target_loc[0], 0)
        self.results.append(target_loc[0])

        #------------------------to visualize what is inside subwindow-----------
        display_subwindow = False
        if display_subwindow:
            subwindow = get_subwindow(self.srch_window_location, image, self.input_size, visualize = display_subwindow)
        
        #------------------------to visualize heatmap on full frame or subwindow-----------
        vis_heatmap_full_frame = False
        vis_heatmap_subwindow = False

        if vis_heatmap_full_frame or vis_heatmap_subwindow:

            if vis_heatmap_full_frame:
                subwindow, track_features = get_frame_features(self.model, img)
            elif vis_heatmap_subwindow:
                subwindow = get_subwindow(self.srch_window_location, image, self.input_size, visualize = False)
                track_features = features
            
            srch_window_size = (subwindow.shape[2],subwindow.shape[1])

            self.visualize_feature(
                            features = track_features,
                            stage = 'conv4_1',
                            srch_window_size = srch_window_size,
                            subwindow = subwindow,
                            feature_weights = self.feature_weights,
                            balance_weights = self.balance_weights 
                            )
        
        #------------------------to visualize convolution between feature maps and exemplar-----------
        vis_conv_feature_map = True
        
        if vis_conv_feature_map:
            subwindow, track_features = get_frame_features(self.model, img)
            self.visualize_conv(
                            features = track_features,
                            stage = 'conv4_3',
                            maps_num = 0,
                            exemplar_features = patch_features,
                            feature_weights = self.feature_weights,
                            balance_weights = self.balance_weights 
                            )

    def tracking(self, img_path, frame, visualize = False):
        #-------------read image and rescale the image-----------------------------
        img = default_image_loader(img_path)#<class 'numpy.ndarray'>[height, width, channel]
        image = cv2.resize(
                img,
                tuple((np.ceil(np.array(img.shape[0:2][::-1]) * self.rescale)).astype(int)),
                interpolation=cv2.INTER_LINEAR
        )
        tic = cv2.getTickCount()
        #-------------get multi-scale feature--------------------------------------
        features = get_subwindow_feature(self.model, image, self.srch_window_location, self.input_size)
        feature_size = (torch.tensor(features[0].shape)).numpy().astype(int)[-2:]
        
        #selected_features = fuse_feature(features)

        #-------------select the target-aware features new frame (not exemplar)---------------------------
        selected_features = features_selection(features, self.feature_weights, self.balance_weights, mode = 'reduction')
        selected_features_1 = resize_tensor(selected_features, tuple(feature_size + self.feature_pad))
        selected_features_3 = resize_tensor(selected_features, tuple(feature_size - self.feature_pad))
        selected_features_1 = selected_features_1[:,:,self.b_feature_pad:feature_size[0]+self.b_feature_pad, self.b_feature_pad:feature_size[1]+self.b_feature_pad]

        selected_features_3 = torch.nn.functional.pad(selected_features_3, (self.b_feature_pad, self.b_feature_pad, self.b_feature_pad, self.b_feature_pad))
        scaled_features = torch.cat((selected_features_1,selected_features,selected_features_3), dim = 0)


        #-------------get response map (final target aware bluish feature map result of correlation)-------------------------
        response_map = self.siamese_model(scaled_features, self.exemplar_features).to('cpu')

        scaled_response_map = torch.squeeze(resize_tensor(
                                                response_map,
                                                tuple(self.srch_window_location[-2:].astype(int)),
                                                mode = 'bicubic',
                                                align_corners = True))

        hann_window = generate_2d_window('hann', tuple(self.srch_window_location[-2:].astype(int)), scaled_response_map.shape[0])
        scaled_response_maps = scaled_response_map + hann_window

        #-------------calculate ROI----------------------------------------------
        scale_ind = calculate_scale(scaled_response_maps, self.config.MODEL.SCALE_WEIGHTS)

        #response_map_reshaped = response_map[scale_ind,0,:,:].numpy()
        #center_h, center_w = np.where(response_map_reshaped == np.max(response_map_reshaped)) #find center ROI
        #center_h, center_w = center_h[0], center_w[0]

        #region_size = self.exemplar_features[0].shape[1:3]
        #width_size = int(region_size[0]/2)
        #width_remainder = region_size[0] % 2
        #height_size = int(region_size[1]/2)
        #height_remainder = region_size[1] % 2

        #plt.imshow(response_map_reshaped)
        #plt.plot(center_w, center_h, "xr", markersize=5)
        #plt.plot(center_w - width_size, center_h - height_size, "or", markersize=5)
        #plt.plot(center_w + width_size + width_remainder, center_h + height_size + height_remainder, "or", markersize=5)
        #plt.show()

        #roi_features = scaled_features[scale_ind, : , center_w - width_size : center_w + width_size + width_remainder, center_h - height_size : center_h + height_size + height_remainder]
        #roi_size = roi_features.shape[1:3]

        #-------------calculate Global Average Pooling current frame features--------------------
        #roi_features_gap = nn.AvgPool2d(roi_size)(roi_features)

        #-------------calculate Affinity Matrix--------------------
        #self.affinity_matrix = torch.sum(self.exemplar_features_gap * roi_features_gap) / len(roi_features_gap)

        #-------------find max-response----------------------------------------------
        response_map = scaled_response_maps[scale_ind,:,:].numpy()
        max_h, max_w = np.where(response_map == np.max(response_map))
        if len(max_h)>1:
            max_h = np.array([max_h[0],])
        if len(max_w)>1:
            max_w = np.array([max_w[0],])

        #-------------update tracking state and save tracking result----------------------------------------
        target_loc_center = np.append(self.target_location[0:2]+(self.target_location[2:4])/2, self.target_location[2:4])
        target_loc_center[0:2] = target_loc_center[0:2] + (np.append(max_w,max_h)-(self.srch_window_location[2:4]/2-1))*self.config.MODEL.SCALES[scale_ind]
        target_loc_center[2:4] = target_loc_center[2:4] * self.config.MODEL.SCALES[scale_ind]
        #print('target_loc_center in current frame:',target_loc_center)
        self.target_location = np.append(target_loc_center[0:2]-(target_loc_center[2:4])/2, target_loc_center[2:4])
        #print('target_location in current frame:', self.target_location)

        self.srch_window_location[2:4] = (round_python2(self.srch_window_location[2:4] * self.config.MODEL.SCALES[scale_ind]))
        self.srch_window_location[0:2] = target_loc_center[0:2]-(self.srch_window_location[2:4])/2

        #print('srch_window_location: ', self.srch_window_location)

        tracking_bbox = (self.target_location + np.array([1,1,0,0]))/self.rescale - np.array([1,1,0,0])#tracking_bbox: 0-index
        self.results.append(tracking_bbox)

        #-------------calculate global average pooling of new frame features-------------------
        kernel_size = selected_features[0].shape[1:2]
        self.selected_features_gap = nn.AvgPool2d(kernel_size)(selected_features[0])

        self.toc += cv2.getTickCount() - tic
        if self.display:
            self.visualization(img,tracking_bbox.astype(int),frame)

    def cal_fps(self, num_frame):
        toc = self.toc/cv2.getTickFrequency()
        return num_frame/toc

    def saving_result(self, result_path, video_name, zero_index = True):

        if zero_index:
            results = self.results
        else:
            results = [result + np.array([1,1,0,0]) for result in self.results]
        with open(result_path + video_name +'.txt','w') as f:
            for bbox in results:
                newline = str(bbox[0])+','+str(bbox[1])+','+str(bbox[2])+','+str(bbox[3]) + '\n'
                f.write(newline)

    def prepare_visualize(self):
        fig,self.ax = plt.subplots(1,1)
        self.ax.axis('off')

    def visualization(self, img, location, frame):
        self.ax.clear()
        plt.imshow(img)
        position = (location[0], location[1])
        width = location[2]
        height = location[3]

        self.ax.add_patch(Rectangle(position, width, height,fill=False,color='r'))
        self.ax.text(0, 0, 'frame: {}'.format(frame), fontsize=15)
        plt.ion()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry = (100,100,800,500)
        plt.pause(0.01)

    def end_visualize(self):
        if self.display:
            plt.clf()
            plt.close()
        else:
            pass

    def visualize_feature(self, features = None, stage = 'conv4_1', srch_window_size = None, subwindow = None, feature_weights = None, balance_weights = None):
        """
        function: visualize the selected feature of the first frame
        """
        assert(stage == 'conv4_1' or stage == 'conv4_3' or stage == 'all'), 'For now, TADT only support for conv4_1 and conv4_3'
        if stage == 'conv4_1':
            stage = 0
        elif stage == 'conv4_3':
            stage = 1
        if feature_weights is None or balance_weights is None:
            if stage == 'all':
                feature = torch.cat(features, dim = 1)
            else:
                feature = features[stage]
        else:
            if stage == 'all':
                feature = features_selection(features, feature_weights, balance_weights, mode = 'reduction')
            else:
                feature = features[stage]
                feature_weight = feature_weights[stage]
                feature = feature_selection(feature, feature_weight, mode = 'reduction')

        heatmap = torch.sum(feature, dim = 1)
        max_value = torch.max(heatmap)
        min_value = torch.min(heatmap)
        heatmap = (heatmap-min_value)/(max_value-min_value)*255
        heatmap = heatmap.cpu().numpy().astype(np.uint8).transpose(1,2,0)

        heatmap = cv2.resize(heatmap, srch_window_size, interpolation=cv2.INTER_LINEAR)
        heatmap=cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #subwindow = subwindow.numpy().astype(np.uint8).transpose(1,2,0)
        subwindow = subwindow.cpu().numpy().astype(np.uint8).transpose(1,2,0) + 128 #convert torch tensor back to open cv image
        cv2.imshow('heatmap',cv2.addWeighted(cv2.cvtColor(subwindow,cv2.COLOR_BGR2RGB), 0.6, heatmap, 0.4, 0.0))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def visualize_conv(self, features = None, stage = 'conv4_1', maps_num = 0, exemplar_features = None, feature_weights = None, balance_weights = None):
        """
        function: visualize the selected feature of the first frame
        """
        assert(stage == 'conv4_1' or stage == 'conv4_3' or stage == 'all'), 'For now, TADT only support for conv4_1 and conv4_3'
        if stage == 'conv4_1':
            stage = 0
        elif stage == 'conv4_3':
            stage = 1
        if feature_weights is None or balance_weights is None:
            if stage == 'all':
                feature = torch.cat(features, dim = 1)
                exemplar = torch.cat(exemplar_features, dim = 1)
            else:
                feature = features[stage]
                exemplar = exemplar_features[stage]
        else:
            if stage == 'all':
                feature = features_selection(features, feature_weights, balance_weights, mode = 'reduction')
                exemplar = features_selection(exemplar_features, feature_weights, balance_weights, mode = 'reduction')
            else:
                feature = features[stage]
                feature_weight = feature_weights[stage]
                feature = feature_selection(feature, feature_weight, mode = 'reduction')
                
                exemplar = exemplar_features[stage]
                exemplar = feature_selection(exemplar, feature_weight, mode = 'reduction')
        
        convolution = nn.functional.conv2d(Variable(feature.view(feature.shape)),Variable(exemplar.view(exemplar.shape)))
        convolution = torch.sum(convolution, dim = 1)
        convolution = convolution.cpu().numpy().astype(np.uint8).transpose(1,2,0) #convert torch tensor back to open cv image
        
        feature_map = torch.sum(feature, dim = 1)
        feature_map = feature_map.cpu().numpy().astype(np.uint8).transpose(1,2,0)
        
        exemplar_map = torch.sum(exemplar, dim = 1)
        exemplar_map = exemplar_map.cpu().numpy().astype(np.uint8).transpose(1,2,0)
        
        #match_template = cv2.matchTemplate(
        #                        feature[:,1,:,:].cpu().numpy().astype(np.uint8).transpose(1,2,0),
        #                        exemplar[:,1,:,:].cpu().numpy().astype(np.uint8).transpose(1,2,0),
        #                        3) #3 = cv.TM_CCORR_NORMED
        
        #cv2.imshow('exemplar', exemplar_map)
        #cv2.imshow('feature_map', feature_map)
        #cv2.imshow('convolution', match_template)
        #cv2.imwrite('./feature_map_conv4_3_layer_359.jpg', feature_map)
        
        #for i in range(exemplar.shape[1]):
        #    match_template = cv2.matchTemplate(
        #                feature[:,i,:,:].cpu().numpy().astype(np.uint8).transpose(1,2,0),
        #                exemplar[:,i,:,:].cpu().numpy().astype(np.uint8).transpose(1,2,0),
        #                3) #3 = cv.TM_CCORR_NORMED
        #    cv2.imwrite('./match_templates/match_template_' + str(i) + '.jpg', match_template)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def cal_feature_pad(features):
    feature_size = (torch.tensor(features[0].shape)).numpy().astype(int)[-2:]
    if feature_size[0] * 0.05 % 2 > 1:
        feature_pad = (np.floor(feature_size[0] * 0.05) + 1).astype(int)
    else:
        feature_pad = (np.ceil(feature_size[0] * 0.05) - 1).astype(int)
    assert feature_pad % 2 == 0, 'feature_pad need to be an even number\n'

    return feature_pad

def cal_srch_window_location(target_location, search_size):
    """
    function: cal_srch_window_location
    """
    srch_window_position = np.floor(target_location[0:2]+target_location[2:4]/2 - search_size/2)
    srch_window_size = np.array([search_size, search_size])
    srch_window_location = np.append(srch_window_position, srch_window_size)
    assert(srch_window_location.shape == (4,)), "the shape of srch_window_location is {}".format(srch_window_location.shape)
    return srch_window_location
