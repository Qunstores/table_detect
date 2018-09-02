# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 21:11:43 2018

@author: Jameswang
"""

import cv2
import numpy  as np
import os 
import os.path
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from perspective_trans import transfer_img
from notrans_detect import table_discarding
import hashlib

class detect_table(object):
    def __init__(self,src_img, map_matrix, input_img, debug_model):
        self.src_img = src_img
        self.map_matrix = map_matrix
        self.input_img = input_img
        self.debug_model = debug_model
        
    def run(self):
        if len(self.src_img.shape) == 2:
            gray_img = self.src_img
            #ori_img = self.src_img
        if len(self.src_img.shape) == 3:
            gray_img = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2GRAY)
            #ori_img = self.src_img
        # show begin
        #warped_img =self.src_img
        input_img = self.input_img
        map_matrix = self.map_matrix
        if self.debug_model:
            cv2.namedWindow('gray_img', cv2.WINDOW_NORMAL)
            cv2.imshow('gray_img', gray_img)
            cv2.waitKey(0)
            cv2.destroyWindow('gray_img')
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
            cv2.destroyWindow('thresh_img')
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
        h_dilate_img = cv2.dilate(h_erode_img, h_structure, 8)
        #blank_img = np.ones(input_img.shape, np.uint8)
        #print(map_matrix)
        recovered_img_h = cv2.warpPerspective(h_dilate_img, map_matrix, (input_img.shape[1], input_img.shape[0]))
        if self.debug_model:
            cv2.namedWindow('recovered_img_h', cv2.WINDOW_NORMAL)
            cv2.imshow('recovered_img_h', recovered_img_h)
            cv2.waitKey(0)
        lines_h = cv2.HoughLinesP(recovered_img_h, 1, np.pi/180, 180)
        print(lines_h[0])
        print('line number: ', len(lines_h))
        for x in range(0, len(lines_h)):
            for x1,y1,x2,y2 in lines_h[x]:
                #x1 = int((map_matrix[0][0]*x1 + map_matrix[0][1]*y1 + map_matrix[0][2])/(map_matrix[2][0]*x1 + map_matrix[2][1]*y1 + map_matrix[2][2]))
                #y1 = int((map_matrix[1][0]*x1 + map_matrix[1][1]*y1 + map_matrix[1][2])/(map_matrix[2][0]*x1 + map_matrix[2][1]*y1 + map_matrix[2][2]))
                #x2 = int((map_matrix[0][0]*x2 + map_matrix[0][1]*y2 + map_matrix[0][2])/(map_matrix[2][0]*x2 + map_matrix[2][1]*y2 + map_matrix[2][2]))
                #y2 = int((map_matrix[1][0]*x2 + map_matrix[1][1]*y2 + map_matrix[1][2])/(map_matrix[2][0]*x2 + map_matrix[2][1]*y2 + map_matrix[2][2]))
                cv2.line(input_img,(x1,y1),(x2,y2),(255,255,255),8)
                #cv2.line(ori_img,(x1,y1),(x2,y2),(255,255,255),8)
        # show begin
        if self.debug_model:
            cv2.namedWindow('h_dilate_img', cv2.WINDOW_NORMAL)
            cv2.imshow('h_dilate_img', input_img)
            cv2.namedWindow('h_dilate_img_1', cv2.WINDOW_NORMAL)
            cv2.imshow('h_dilate_img_1', h_dilate_img)
            cv2.waitKey(0)
            cv2.destroyWindow('h_dilate_img')
            cv2.destroyWindow('h_dilate_img_1')

        #k = cv2.waitKey(0)&0xFF
        #if k == 27:
        #    cv2.destroyAllWindows()
        # show end
        scale =18
        v_size = int(v_img.shape[0] / scale)
        v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
        v_erode_img = cv2.erode(v_img, v_structure, 1)
        v_dilate_img = cv2.dilate(v_erode_img, v_structure, 4)
        recovered_img_v = cv2.warpPerspective(v_dilate_img, map_matrix, (input_img.shape[1], input_img.shape[0]))
        if self.debug_model:
            cv2.namedWindow('recovered_img_v', cv2.WINDOW_NORMAL)
            cv2.imshow('recovered_img_v', recovered_img_v)
            cv2.waitKey(0)
        lines_v = cv2.HoughLinesP(recovered_img_v, 1, np.pi/180, 200, 0)
        print('line number: ', len(lines_v))
        for x in range(0, len(lines_v)):
            for x1,y1,x2,y2 in lines_v[x]:
                #x1 = int((map_matrix[0][0]*x1 + map_matrix[0][1]*y1 + map_matrix[0][2])/(map_matrix[2][0]*x1 + map_matrix[2][1]*y1 + map_matrix[2][2]))
                #y1 = int((map_matrix[1][0]*x1 + map_matrix[1][1]*y1 + map_matrix[1][2])/(map_matrix[2][0]*x1 + map_matrix[2][1]*y1 + map_matrix[2][2]))
                #x2 = int((map_matrix[0][0]*x2 + map_matrix[0][1]*y2 + map_matrix[0][2])/(map_matrix[2][0]*x2 + map_matrix[2][1]*y2 + map_matrix[2][2]))
                #y2 = int((map_matrix[1][0]*x2 + map_matrix[1][1]*y2 + map_matrix[1][2])/(map_matrix[2][0]*x2 + map_matrix[2][1]*y2 + map_matrix[2][2]))
                cv2.line(input_img,(x1,y1),(x2,y2),(255,255,255),6)
                
        # show begin
        if self.debug_model:
            cv2.namedWindow('v_dilate_img', cv2.WINDOW_NORMAL)
            cv2.imshow('v_dilate_img', input_img)
            k = cv2.waitKey(0)&0xFF
            if k == 27:
                cv2.destroyAllWindows()

        mask_img = h_dilate_img + v_dilate_img
        mask_img = cv2.warpPerspective(mask_img, map_matrix, (input_img.shape[1], input_img.shape[0]))
        # show begin
        if self.debug_model:
            cv2.namedWindow('mask_img', cv2.WINDOW_NORMAL)
            cv2.imshow('mask_img', mask_img)
            cv2.waitKey(0)
            cv2.destroyWindow('mask_img')

        joints_img = cv2.bitwise_and(h_dilate_img, v_dilate_img)
        # show begin
        if self.debug_model:
            cv2.namedWindow('joints_img', cv2.WINDOW_NORMAL)
            cv2.imshow('joints_img', joints_img)
            k = cv2.waitKey(0)&0xFF
            if k == 27:
                cv2.destroyAllWindows()
            #elif k == ord('q'):
        # show end
        
        return mask_img, joints_img, gray_img, input_img
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
    
                       
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action = 'store_true')
    args = vars(parser.parse_args())
    debug_model = args["debug"]
    
    path_origin = "./origin_images/"
    #path_origin = "./test_img/"
    path_result = "./result_images/"
    input_imgs = []
    if os.path.exists(path_origin) is not True:
        os.makedirs(path_origin)
        print(path_origin, " has been created ")
    
    if os.path.exists(path_result) is not True:
        os.makedirs(path_result)
        print(path_result, "has been created!")
    
    valid_images = [".jpg"]
    img_names = []
    for f in os.listdir(path_origin):
        ext = os.path.splitext(f)[1]
        img_name = os.path.split(f)[1]
        if ext.lower() not in valid_images:
            continue
        input_imgs.append(cv2.imread(os.path.join(path_origin, f)))
        img_names.append(img_name)
    
    for num in range(len(input_imgs)):
        input_img = input_imgs[num]
        print(path_result+img_names[num])
        warped, ratio, map_matrix, no_transfer = transfer_img(input_img, debug_model)
        #table_discard =table_discarding(input_img, debug_model)
        #cv2.imwrite(path_result+img_names[num], table_discard)
        #print(path_result+img_names[num], 'write ok...')

        if no_transfer:
            table_discard =table_discarding(input_img, debug_model)
            hash_digest = hashlib.md5(table_discard.tostring()).hexdigest()
            cv2.imwrite(path_result+hash_digest+'.jpg', table_discard)
            #cv2.imwrite(path_result+img_names[num], table_discard)
            print(path_result+img_names[num], 'write ok...')
        else:
            if debug_model:
                cv2.namedWindow('warped_image', cv2.WINDOW_NORMAL)
                cv2.imshow('warped_image', warped)
                cv2.waitKey(0)
                cv2.destroyWindow('warped_image')
            mask, joint, gray_image, out_img = detect_table(warped, map_matrix, input_img, debug_model).run()
            print("mask shape is: ", mask.shape)
            kernel = np.ones((3,11), np.uint8)
            closing_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            if debug_model:
                cv2.namedWindow('closing_img', cv2.WINDOW_NORMAL)
                cv2.imshow("closing_img", closing_img)
                cv2.destroyWindow('closing_img')
            tmp = np.zeros(out_img.shape, np.uint8)
            print("temp shape is: ", tmp.shape)
            tmp[:, :, 0], tmp[:, :, 1], tmp[:, :, 2] = mask, mask, mask
            if debug_model:
                cv2.namedWindow('tmp', cv2.WINDOW_NORMAL)
                cv2.imshow("tmp", tmp)
                cv2.waitKey(0)
                cv2.destroyWindow('tmp')
            table_discard = cv2.add(out_img, tmp)
            if debug_model:
                cv2.namedWindow('out_table', cv2.WINDOW_NORMAL)
                cv2.imshow("out_table", table_discard)
                cv2.waitKey(0)
                cv2.destroyWindow('out_table')
            #cv2.imwrite(path_result+img_names[num], table_discard)
            hash_digest = hashlib.md5(table_discard.tostring()).hexdigest()
            cv2.imwrite(path_result+hash_digest+'.jpg', table_discard)
            print(path_result+img_names[num], 'write ok...')
                
  
    
"""
        if debug_model:
            cv2.namedWindow('warped_image', cv2.WINDOW_NORMAL)
            cv2.imshow('warped_image', warped)
            cv2.waitKey(0)
            cv2.destroyWindow('warped_image')
        
        mask, joint, gray_image, out_img = detect_table(warped, map_matrix, input_img, debug_model).run()
        print("mask shape is: ", mask.shape)
        kernel = np.ones((3,11), np.uint8)
        closing_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if debug_model:
            cv2.namedWindow('closing_img', cv2.WINDOW_NORMAL)
            cv2.imshow("closing_img", closing_img)
            cv2.destroyWindow('closing_img')
        tmp = np.zeros(out_img.shape, np.uint8)
        print("temp shape is: ", tmp.shape)
        tmp[:, :, 0], tmp[:, :, 1], tmp[:, :, 2] = mask, mask, mask
        
        if debug_model:
            cv2.namedWindow('tmp', cv2.WINDOW_NORMAL)
            cv2.imshow("tmp", tmp)
            cv2.waitKey(0)
            cv2.destroyWindow('tmp')
        table_discard = cv2.add(out_img, tmp)
        if debug_model:
            cv2.namedWindow('out_table', cv2.WINDOW_NORMAL)
            cv2.imshow("out_table", table_discard)
            cv2.waitKey(0)
            cv2.destroyWindow('out_table')
        
        cv2.imwrite(path_result+img_names[num], table_discard)
        print('write image ok...')
"""           
        

"""    
    input_img = cv2.imread('jsyh3.jpg')
    #img = cv2.resize(img, (800, 500))
    # show begin
    warped, ratio, map_matrix = transfer_img(input_img, debug_model)
    if debug_model:
        cv2.namedWindow('warped_image', cv2.WINDOW_NORMAL)
        cv2.imshow('warped_image', warped)
        cv2.waitKey(0)
        cv2.destroyWindow('warped_image')
    
    mask, joint, gray_image, out_img = detect_table(warped, map_matrix, input_img, debug_model).run()
    
    print("mask shape is: ", mask.shape)
    kernel = np.ones((3,11), np.uint8)
    closing_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if debug_model:
        cv2.namedWindow('closing_img', cv2.WINDOW_NORMAL)
        cv2.imshow("closing_img", closing_img)
        cv2.destroyWindow('closing_img')
    tmp = np.zeros(out_img.shape, np.uint8)
    print("temp shape is: ", tmp.shape)
    tmp[:, :, 0], tmp[:, :, 1], tmp[:, :, 2] = mask, mask, mask
    
    if debug_model:
        cv2.namedWindow('tmp', cv2.WINDOW_NORMAL)
        cv2.imshow("tmp", tmp)
        cv2.waitKey(0)
        cv2.destroyWindow('tmp')
    table_discard = cv2.add(out_img, tmp)
    if debug_model:
        cv2.namedWindow('out_table', cv2.WINDOW_NORMAL)
        cv2.imshow("out_table", table_discard)
        cv2.waitKey(0)
        cv2.destroyWindow('out_table')
    cv2.imwrite('out_table.jpg', table_discard)
        
    ##test max_rectangle
    '''
    blank_image, block_list = detect_table.max_rectangle(out_table)
    cv2.namedWindow('blank_image', cv2.WINDOW_NORMAL)
    cv2.imshow('blank_image', blank_image)
    
    k = cv2.waitKey(0)&0xFF
    if k == 27:
        cv2.destroyAllWindows()

        
    drawLine(block_list,img, 1175, 2483)
    '''
    
"""
    
    

