# table_detect
this project rely on python3.6.5, the below module should be installed on your PC( my environment is windows anaconda3):

   pip install numpy
   
   pip install Matplotlib
   
   pip install opencv-python
   
the project is used to remove table in in china construction bank Receipt, it could be totally removed table on Receipt and result used to feeding OCR engine, the prediction accuracy could improved 10% after this image preprocessing. it also could expand to other receipt or image files table remove issues, the image processing effect like this:

original receipt:

![Image text](https://github.com/Qunstores/table_detect/blob/master/origin_images/6505af84870ba941294d9bf6f58c9c98.jpg)

out table receipt:

![Image text](https://github.com/Qunstores/table_detect/blob/master/result_images/7dcc642bef9100c9d6f8287644171b1b.jpg)

Usage:

   python table_detect.py 
   python table_detect.py --debug (debug arguments to graphical interaction)
   
   
   
   
