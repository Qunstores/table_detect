# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 21:11:43 2018

@author: Jameswang
"""

import cv2
import numpy  as np
import os 
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from perspective_trans import transfer_img

class detect_table(object):
    def __init__(self, src_img, debug_model):
        self.src_img = src_img
        self.debug_model = debug_model
    
    def run(self):
        if len(self.src_img.shape) == 2:
            gray_img = self.src_img
            ori_img = self.src_img
        if len(self.src_img.shape) == 3:
            gray_img = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2GRAY)
            ori_img = self.src_img
        # show begin
        if self.debug_model:
            cv2.namedWindow('gray_img', cv2.WINDOW_NORMAL)
            cv2.imshow('gray_img', gray_img)
        '''
        k = cv2.waitKey(0)&0xFF
        if k == 27:
            cv2.destroyWindow('gray_img')
        elif k == ord('s'):
            cv2.write('./my_images/gray_img.jpg', gray_img)
            cv2.destroyWindow('gray_img')
        '''
        # show end
        #自适应阈值分割
        mean = gray_img.mean()
        if mean < 100:
            gray_img = 255-gray_img
        thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 101, 21)
        #thresh_img = cv2.adaptiveThreshold(~gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        #show begin
        if self.debug_model:
            cv2.namedWindow('thresh_img', cv2.WINDOW_NORMAL)
            cv2.imshow('thresh_img', thresh_img)
            cv2.waitKey(0)
        '''
        k = cv2.waitKey(0)&0xFF
        if k == 27:
            cv2.destroyWindow('thresh_img')
        elif k == ord('s'):
            cv2.write('./my_images/thresh_img.jpg', thresh_img)
            cv2.destroyWindow('thresh_img')
        '''
        #show end
        #腐蚀+膨胀
        h_img = thresh_img.copy()
        v_img = thresh_img.copy()
        scale = 40
        h_size = int(h_img.shape[1] / scale)
        h_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
        print("original structure is: ", h_img.shape)
        print("h_structure is :", h_structure.shape)
        h_erode_img = cv2.erode(h_img, h_structure, 1)
        h_dilate_img = cv2.dilate(h_erode_img, h_structure, 6)
        #cv2.line(ori_img,(x1,y1),(x2,y2),(255,255,255),8)
        # show begin
        lines_h = cv2.HoughLinesP(h_dilate_img, 1, np.pi/180, 200, 0)
        #print('line number: ', len(lines_h))
        for x in range(0, len(lines_h)):
            for x1,y1,x2,y2 in lines_h[x]:
                cv2.line(ori_img,(x1,y1),(x2,y2),(255,255,255),6)
        if self.debug_model:
            cv2.namedWindow('h_dilate_img', cv2.WINDOW_NORMAL)
            cv2.imshow('h_dilate_img', h_dilate_img)
        
        #k = cv2.waitKey(0)&0xFF
        #if k == 27:
        #    cv2.destroyAllWindows()
        # show end
        scale =14
        v_size = int(v_img.shape[0] / scale)
        v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
        v_erode_img = cv2.erode(v_img, v_structure, 1)
        v_dilate_img = cv2.dilate(v_erode_img, v_structure, 4)
        lines_v = cv2.HoughLinesP(v_dilate_img, 1, np.pi/180, 200, 0)
        print('line number: ', len(lines_v))
        for x in range(0, len(lines_v)):
            for x1,y1,x2,y2 in lines_v[x]:
                cv2.line(ori_img,(x1,y1),(x2,y2),(255,255,255),6)
                
        # show begin
        if self.debug_model:
            cv2.namedWindow('v_dilate_img', cv2.WINDOW_NORMAL)
            cv2.imshow('v_dilate_img', v_dilate_img)
            k = cv2.waitKey(0)&0xFF
            if k == 27:
                cv2.destroyAllWindows()
        mask_img = h_dilate_img + v_dilate_img
        # show begin
        if self.debug_model:
            cv2.namedWindow('mask_img', cv2.WINDOW_NORMAL)
            cv2.imshow('mask_img', mask_img)
        '''
        k = cv2.waitKey(0)&0xFF
        if k == 27:
            cv2.destroyAllWindows()
        # show end
        '''
        joints_img = cv2.bitwise_and(h_dilate_img, v_dilate_img)
                # show begin
        if self.debug_model:
            cv2.namedWindow('joints_img', cv2.WINDOW_NORMAL)
            cv2.imshow('joints_img', joints_img)
            k = cv2.waitKey(0)&0xFF
            if k == 27:
                cv2.destroyAllWindows()
        # show end
        
        return mask_img, joints_img, gray_img
    def max_rectangle(gray_image):
        mser = cv2.MSER_create(_min_area=10, _max_area=3000)
        regions, boxes = mser.detectRegions(gray_image)
        blank_image = np.zeros(gray_image.shape, np.uint8)
        block_list = []
        for box in boxes:
            x, y, w, h = box
            # # only for draw
            cv2.rectangle(gray_image, (x, y), (x + w, y + h), (255, 0, 0), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(gray_image, f'{x},{y}',(x, y), font, 1,(255,255,255),1,cv2.LINE_AA)
            block_list.append((x, y,x + w, y + h))
            blank_image[y:y+h,x:x + w]=255
        h_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 形态学因子
        blank_image = cv2.dilate(blank_image, h_structure, iterations=1)
        return blank_image, block_list
    
def drawLine(all_lines, src_image, height=841, width=595):
    #blank_image = np.zeros((height, width, 3), np.uint8)
    blank_image = src_image
    #color = tuple(reversed((0, 0, 0)))
    #blank_image[:] = color
    for line in all_lines:
        p1 = [int(np.round(line[0])), int(np.round(line[1]))]
        p2 = [int(np.round(line[2])), int(np.round(line[1]))]
        p3 = [int(np.round(line[0])), int(np.round(line[3]))]
        p4 = [int(np.round(line[2])), int(np.round(line[3]))]
        cv2.rectangle(blank_image, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 0), 1)
        cv2.rectangle(blank_image, (p1[0], p1[1]), (p3[0], p3[1]), (255, 0, 0), 1)
        cv2.rectangle(blank_image, (p2[0], p2[1]), (p4[0], p4[1]), (255, 0, 0), 1)
        cv2.rectangle(blank_image, (p3[0], p4[1]), (p4[0], p4[1]), (255, 0, 0), 1)
    cv2.namedWindow('img_blank', cv2.WINDOW_NORMAL)
    cv2.imshow('img_blank', blank_image)
    k = cv2.waitKey(0)&0xFF
    if k == 27:
        cv2.destroyAllWindows()
    
def table_discarding(input_img, debug_model):
    mask, joint, gray_image = detect_table(input_img, debug_model).run()
    print("mask shape is: ", mask.shape)
    kernel = np.ones((3,11), np.uint8)
    closing_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if debug_model:
        cv2.namedWindow('closing_img', cv2.WINDOW_NORMAL)
        cv2.imshow("closing_img", closing_img)
    tmp = np.zeros(input_img.shape, np.uint8)
    print("temp shape is: ", tmp.shape)
    tmp[:, :, 0], tmp[:, :, 1], tmp[:, :, 2] = mask, mask, mask
    if debug_model:
        cv2.namedWindow('tmp', cv2.WINDOW_NORMAL)
        cv2.imshow("tmp", tmp)
    out_table = cv2.add(input_img, tmp)
    return out_table
               
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action = 'store_true')
    parser.add_argument("--use_transfer", action = 'store_true')
    args = vars(parser.parse_args())
    
    use_transfer = args["use_transfer"]
    debug_model = args["debug"]
    input_img = cv2.imread('jsyh3.jpg')
    #img = cv2.resize(img, (800, 500))
    # show begin
    
    #if use_transfer:
        #warped, ratio, map_matrix = transfer_img(input_img)
        #cv2.namedWindow('warped_image', cv2.WINDOW_NORMAL)
        #cv2.imshow('warped_image', warped)
    #else:
        #map_matrix = False
    '''
    k = cv2.waitKey(0)&0xFF
    if k == 27:
        cv2.destroyWindow('img')
    elif k == ord('s'):
        cv2.write('./my_images/origin_img.jpg', img)
        cv2.destroyWindow('img')
    # show end
    '''
    
    mask, joint, gray_image = detect_table(input_img, debug_model).run()
    
    print("mask shape is: ", mask.shape)
    kernel = np.ones((3,11), np.uint8)
    closing_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if debug_model:
        cv2.namedWindow('closing_img', cv2.WINDOW_NORMAL)
        cv2.imshow("closing_img", closing_img)
    tmp = np.zeros(input_img.shape, np.uint8)
    print("temp shape is: ", tmp.shape)
    tmp[:, :, 0], tmp[:, :, 1], tmp[:, :, 2] = mask, mask, mask
    if debug_model:
        cv2.namedWindow('tmp', cv2.WINDOW_NORMAL)
        cv2.imshow("tmp", tmp)
    out_table = cv2.add(input_img, tmp)
    if debug_model:
        cv2.namedWindow('out_table', cv2.WINDOW_NORMAL)
        cv2.imshow("out_table", out_table)
        k = cv2.waitKey(0)&0xFF
        if k == 27:
            cv2.destroyAllWindows()
    cv2.imwrite('out_table.jpg', out_table)

    ##test max_rectangle
    """
    blank_image, block_list = detect_table.max_rectangle(out_table)
    cv2.namedWindow('blank_image', cv2.WINDOW_NORMAL)
    cv2.imshow('blank_image', blank_image)
    
    k = cv2.waitKey(0)&0xFF
    if k == 27:
        cv2.destroyAllWindows()

        
    drawLine(block_list,img, 1175, 2483)
    """
    
    

