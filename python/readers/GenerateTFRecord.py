import os
import glob
import pickle
import string
import random
import argparse
import traceback
import functools

import cv2
import PIL
import torch
import numpy as np
import pytesseract
from xml.etree import ElementTree as ET

from libs.configuration_manager import ConfigurationManager as gconfig

class Logger:
    def __init__(self):
        pass
        #self.file=open('logtxt.txt','a+')

    def write(self,txt):
        file = open('logfile.txt', 'a+')
        file.write(txt)
        file.close()

class GenerateTFRecord:
    def __init__(self,unlvimagespath,unlvocrpath,unlvtablepath,visualizeimgs,visualizebboxes,
                    device):
        self.unlvocrpath=unlvocrpath                    #unlv ocr ground truth files
        self.unlvimagespath=unlvimagespath              #unlv images
        self.unlvtablepath=unlvtablepath                #unlv ground truth of tabls
        self.visualizeimgs=visualizeimgs                #wheter to store images separately or not
        self.logger=Logger()                            #if we want to use logger and store output to file
        
        self.num_of_max_vertices=gconfig.get("max_vertices", int)         #number of vertices (maximum number of words in any table)
        self.max_length_of_word=gconfig.get("max_words_len", int)         #max possible length of each word
        
        self.max_height=gconfig.get("max_image_height", int)              #max image height
        self.max_width=gconfig.get("max_image_width", int)                #max image width

        self.max_columns = gconfig.get("max_columns", int)
        self.max_rows = gconfig.get("max_rows", int)

        self.device = device

        self.visualizebboxes=visualizebboxes
        self.counter = 0
        self.tmp_unlv_tables = None
        self.xmlfilepaths = glob.glob(os.path.join(self.unlvtablepath, "*.xml"))
        random.shuffle(self.xmlfilepaths)
        
    def create_dir(self,fpath):                         #creates directory fpath if it does not exist
        if(not os.path.exists(fpath)):
            os.mkdir(fpath)

    def str_to_int(self,str):                           #converts each character in a word to equivalent int
        intsarr=np.array([ord(chr) for chr in str])
        padded_arr=np.zeros(shape=(self.max_length_of_word),dtype=np.int64)
        padded_arr[:len(intsarr)]=intsarr
        return padded_arr

    def convert_to_int(self, arr):                      #simply converts array to a string
        return [int(val) for val in arr]

    def pad_with_value(self,arr,shape, value=0):                 #will pad the input array with zeros to make it equal to 'shape'
        dummy=np.zeros(shape,dtype=np.int64)
        dummy[:] = value
        dummy[:arr.shape[0],:arr.shape[1]]=arr
        return dummy

    def generate_tf_record(self, im, bboxes, cell_ids, gt_right, gt_down, rows, columns,imgindex,output_file_name):
        '''This function generates tfrecord files using given information'''
        num_rows, num_cols = len(rows), len(columns)
        gt_right = self.pad_with_value(gt_right, (self.max_rows - 1, self.max_columns - 2), value=2).astype(np.uint8)
        gt_down = self.pad_with_value(gt_down, (self.max_rows - 2, self.max_columns - 1), value=2).astype(np.uint8)
        rows = self.pad_with_value(np.array(rows)[np.newaxis, :], (1, self.max_rows))[0].astype(np.int64)
        columns = self.pad_with_value(np.array(columns)[np.newaxis, :], (1, self.max_columns))[0].astype(np.int64)

        im = im.astype(np.int64)

        lengths_arr = bboxes[:, 0].astype(np.int64)[:, np.newaxis]
        vertex_features = np.zeros(shape=(self.num_of_max_vertices, bboxes.shape[1] - 1), dtype=np.int64)

        vertex_features[:bboxes.shape[0], :4] = bboxes[:, 2:]
        vertex_features[:bboxes.shape[0], 4] = bboxes[:, 0]

        assert vertex_features.shape[1] == 5

        cell_ids_padded = np.ones(shape=(self.num_of_max_vertices,2), dtype=np.int64) * -1
        cell_ids_padded[:cell_ids.shape[0],:] = cell_ids

        if(self.visualizebboxes):
            self.draw_matrices(im,bboxes,None,imgindex,output_file_name)

        # vertex_text = np.zeros((self.num_of_max_vertices,self.max_length_of_word), dtype=np.int64)
        # vertex_text[:no_of_words]=np.array(list(map(self.str_to_int,words_arr)))

        return [vertex_features.astype(np.float32),
                cell_ids_padded.astype(np.float32),
                # vertex_features[:, -2:].astype(np.float32),
               # vertex_text.astype(np.int64), \
               im.astype(np.float32), \
               np.array([*im.shape[:2],bboxes.shape[0],num_rows,num_cols,0]).astype(np.float32),
               gt_down,
               gt_right,
               rows,
               columns]

    @staticmethod
    def read_unlv_ocr(filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        bboxes = []
        for obj in root.findall(".//word"):
            x0 = obj.attrib["left"]
            y0 = obj.attrib["top"]
            x1 = obj.attrib["right"]
            y1 = obj.attrib["bottom"]
            txt = obj.text.strip()
            bboxes.append([len(txt), txt, int(x0), int(y0), int(x1), int(y1)])
        return bboxes

    @staticmethod
    def apply_ocr(path, image):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            w, h = image.size
            r = 2500 / w
            image = image.resize((2500, int(r * h)))

            print("OCR start")
            ocr = pytesseract.image_to_data(image,
                                            output_type=pytesseract.Output.DICT,
                                            config="--oem 1")
            print("OCR end")
                        
            bboxes = []
            for i in range(len(ocr['conf'])):
                if ocr['level'][i] > 4 and ocr['text'][i].strip()!="":
                    bboxes.append([
                        len(ocr['text'][i]),
                        ocr['text'][i],
                        int(ocr['left'][i] / r),
                        int(ocr['top'][i] / r),
                        int(ocr['left'][i] / r) + int(ocr['width'][i] / r),
                        int(ocr['top'][i] / r) + int(ocr['height'][i] / r)
                        ])
        
            bboxes = sorted(bboxes, key=lambda box: (box[4] - box[2]) * (box[5] - box[3]), reverse=True)
            threshold = np.average([(box[4] - box[2]) * (box[5] - box[3]) for box in bboxes[len(bboxes) // 20: -len(bboxes) // 4]])
            bboxes = [box for box in bboxes if (box[4] - box[2]) * (box[5] - box[3]) < threshold * 30]

            with open(path, "wb") as f:
                pickle.dump(bboxes, f)

            return bboxes

    def create_same_matrix(self,arr,ids):
        '''Given a list of lists with each list consisting of all ids considered same, this function
         generates a matrix '''
        matrix=np.zeros(shape=(ids,ids))
        for subarr in arr:
            for element in subarr:
                matrix[element,subarr]=1
        return matrix

    def data_generator(self, start, end):
        def compare(w1, w2):
            if max(w1[3], w2[3])-min(w1[5], w2[5]) > 0.2 * (w1[5] - w1[3]):
                if w1[3] < w2[3]:
                    return -1
                elif w1[3] > w2[3]:
                    return 1
                else:
                    return 0
            else:
                if w1[2] < w2[2]:
                    return -1
                elif w1[2] > w2[2]:
                    return 1
                else:
                    return 0

        n = len(self.xmlfilepaths)
        print(start, end)
        for counter, filename in enumerate(self.xmlfilepaths[int(start*n): int(end*n)]):
            # print(self.xmlfilepaths.index(filename), filename)
            filename = ".".join(filename.split('/')[-1].split('.')[:-1])
            if not os.path.exists(os.path.join(self.unlvtablepath, filename + ".xml")):
                print("WARNING: Ground truth not found for image ", filename)
                continue
            tree = ET.parse(os.path.join(self.unlvtablepath, filename + ".xml"))
            root = tree.getroot()
            xml_tables = root.findall(".//Table")
            if os.path.exists(os.path.join(self.unlvimagespath, filename + ".png")):
                im = PIL.Image.open(os.path.join(self.unlvimagespath, filename + ".png")).convert("RGB")
            else:
                continue

            bboxes = GenerateTFRecord.apply_ocr(os.path.join(self.unlvocrpath, filename + ".pkl"), im.copy())

            for i, obj in enumerate(xml_tables):
                x0 = int(obj.attrib["x0"])
                y0 = int(obj.attrib["y0"])
                x1 = int(obj.attrib["x1"])
                y1 = int(obj.attrib["y1"])
                im2 = im.crop((x0, y0, x1, y1))

                cells = []
                same_cell_boxes = [[] for _ in range(len(obj.findall('.//Cell')))]
                same_row_boxes = [[] for _ in range(len(obj.findall('.//Row')) + 1)]
                same_col_boxes = [[] for _ in range(len(obj.findall('.//Column')) + 1)]
                
                bboxes_table = []
                for box in bboxes:
                    coords = box[2:]
                    intrsct = [
                                max(coords[0], x0), 
                                max(coords[1], y0), 
                                min(coords[2], x1), 
                                min(coords[3], y1)
                                ]
                    w = intrsct[2] - intrsct[0]
                    h = intrsct[3] - intrsct[1] 

                    w2 = coords[2] - coords[0]
                    h2 = coords[3] - coords[1]
                    if w > 0 and h > 0 and w * h > 0.5 * w2 * h2:
                        box = list(box)
                        text = box[1]
                        text = text.translate(str.maketrans('', '', string.punctuation)).strip()

                        if len(text) == 0:
                            continue

                        if len(box[1]) > self.max_length_of_word:
                            box[1] = box[1][:self.max_length_of_word]
                        bboxes_table.append(box)
                bboxes = [box for box in bboxes if box not in bboxes_table]

                bboxes_table.sort(key=functools.cmp_to_key(compare))

                if len(bboxes_table) > self.num_of_max_vertices:
                    print("\n\nWARNING: Number of vertices (", len(bboxes_table) ,")is greater than limit (", self.num_of_max_vertices, ").\n\n")
                    bboxes_table = bboxes_table[:self.num_of_max_vertices]

                w_org, h_org = im2.size
                h,w=self.max_height, self.max_width

                if im2.size[0]< 20 or im2.size[1]< 20:
                    continue
                    
                im2 = im2.resize((im2.size[0] * 2500 // im.size[0], im2.size[1] * 2500 // im.size[0]))
                if im2.size[0] > w:
                    im2 = im2.resize((w, im2.size[1] * w // im2.size[0]))
                if im2.size[1] > h:
                    im2 = im2.resize((im2.size[0] * h // im2.size[1], h))
                w_new, h_new = im2.size
                new_im = PIL.Image.new("RGB", (w, h), color="white")
                new_im.paste(im2)

                r = w_org / h_org
                
                rows = [0]
                columns = [0]
                for row in obj.findall('.//Row'):
                    rows.append((int(row.attrib["y0"]) - y0) * h_new // h_org)
                rows.append(h_new)
                for col in obj.findall('.//Column'):
                    columns.append((int(col.attrib["x0"]) - x0) * w_new // w_org)
                columns.append(w_new)

                gt_down =   np.zeros((len(rows) - 1, len(columns)))
                gt_right =  np.zeros((len(rows), len(columns) - 1))

                flag = True
                for idx, cell in enumerate(obj.findall('.//Cell')):                    
                    if cell.attrib['dontCare'] == "true":
                        continue
                    for j in range(int(cell.attrib["startCol"]), int(cell.attrib["endCol"]) + 1):
                        for k in range(int(cell.attrib["startRow"]), int(cell.attrib["endRow"])): 
                            gt_down[k, j] = 1
                    for j in range(int(cell.attrib["startRow"]), int(cell.attrib["endRow"]) + 1): 
                        for k in range(int(cell.attrib["startCol"]), int(cell.attrib["endCol"])):
                            gt_right[j, k] = 1
                            flag = False
                if flag:
                    continue

                for j in range(len(bboxes_table)):
                    bboxes_table[j][2] -= x0
                    bboxes_table[j][4] -= x0
                    bboxes_table[j][2] = bboxes_table[j][2] * w_new // w_org
                    bboxes_table[j][4] = bboxes_table[j][4] * w_new // w_org

                    bboxes_table[j][3] -= y0
                    bboxes_table[j][5] -= y0
                    bboxes_table[j][3] = bboxes_table[j][3] * h_new // h_org
                    bboxes_table[j][5] = bboxes_table[j][5] * h_new // h_org

                cell_ids = []
                for box in bboxes_table:
                    coords = box[2:]

                    w = coords[2] - coords[0]
                    h = coords[3] - coords[1]

                    overlaps_r = []
                    overlaps_c = []
                    for j in range(len(rows) - 1):
                        h2 = min(coords[3], rows[j+1]) - max(coords[1], rows[j])
                        overlaps_r.append(h2)
                    for k in range(len(columns) - 1):
                        w2 = min(coords[2], columns[k+1]) - max(coords[0], columns[k])
                        overlaps_c.append(w2)
                    
                    cell_ids.append((np.argmax(overlaps_r), np.argmax(overlaps_c)))

                assert len(cell_ids) == len(bboxes_table)
                cell_ids = np.array(cell_ids)

                if len(bboxes_table) == 0:
                    print("WARNING: No word boxes found inside table #", i, " in image ", filename)
                    continue

                # bboxes_table = np.array(np.concatenate((bboxes_table, cell_ids), axis=1)) 

                # if(self.visualizeimgs):
                #     dirname=os.path.join('visualizeimgs/')
                #     new_im.save(os.path.join(dirname,'img',filename + '-' + str(i) + ".png"))  

                img=np.asarray(new_im, np.int64)[:,:,0]

                gt_right = np.array(gt_right, dtype=np.uint8)
                gt_down = np.array(gt_down, dtype=np.uint8)

                num_rows = len(rows)
                num_columns = len(columns)
                # rows = np.array(rows, dtype=np.int64)
                # columns = np.array(columns, dtype=np.int64)

                tensors = self.generate_tf_record(img, np.array(bboxes_table), cell_ids, gt_right, gt_down, rows, columns, counter, "_")
                tensors = list(map(torch.Tensor, tensors))

                tensors[2] = tensors[2][None, :, :].repeat(3, 1, 1)
                tensors.append(torch.LongTensor([num_rows]).squeeze(0))
                tensors.append(torch.LongTensor([num_columns]).squeeze(0))
                yield tuple(tensors)


    def draw_matrices(self, img, bboxes, matrices, imgindex, output_file_name):
        '''Call this fucntion to draw visualizations of a matrix on image'''
        no_of_words=len(bboxes)
        colors = np.random.randint(0, 220, (100, 3))
        bboxes = bboxes[:, 2:]

        img=img.astype(np.uint8)
        img=np.dstack((img,img,img))

        output_file_name=output_file_name.replace('.tfrecord','')

        cell_ids = np.array(bboxes[:, -2:], dtype=np.uint8)

        im = img.copy()
        for i in range(self.max_rows):
            c = tuple(colors[i, :].tolist())
            indices = np.argwhere(cell_ids[:, 0] == i)

            for index in indices:
                cv2.rectangle(im, (int(bboxes[index, 0])-3, int(bboxes[index, 1])-3),
                                  ((int(bboxes[index, 0]) + int(bboxes[index, 2]))//2, int(bboxes[index, 3])+3),
                                  c, -1)


        for i in range(self.max_columns):
            c = tuple(colors[i, :].tolist())
            indices = np.argwhere(cell_ids[:, 1] == i)

            for index in indices:
                cv2.rectangle(im, ((int(bboxes[index, 0]) + int(bboxes[index, 2]))//2, int(bboxes[index, 1])-3),
                                  (int(bboxes[index, 2])+3, int(bboxes[index, 3])+3),
                                  c, -1)
            

        # for matname,matrix in zip(mat_names,matrices):
        #     im=img.copy()
        #     for x in range(len(matrix)):
        #         indices = np.argwhere(matrix[x] == 1)
        #         for index in indices:
        #             cv2.rectangle(im, (int(bboxes[index, 0])-3, int(bboxes[index, 1])-3),
        #                           (int(bboxes[index, 2])+3, int(bboxes[index, 3])+3),
        #                           (int(colors[x][0]), int(colors[x][1]), int(colors[x][2])), 3)

        img_name=os.path.join('visualizeimgs/bboxes/',output_file_name+'_'+str(imgindex)+'.jpg')
        cv2.imwrite(img_name,im)