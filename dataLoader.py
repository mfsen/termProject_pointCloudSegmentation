import glob,os,yaml
from operator import le
import numpy as np
from tensorflow.keras.utils import Sequence

class dataLoader():
    
    def __init__(self,dataPath) -> None:
        self.dataPath = dataPath

    def loadDataPaths(self):
        """This function finds all the point datas and correspınding label data in given path"""

        pointCloudPath =  glob.glob(os.path.join(self.dataPath,"velodyne/*.bin"))
        labelPath =  glob.glob(os.path.join(self.dataPath,"labels/*.label"))

        return pointCloudPath,labelPath

class dataPreProcessor():
    def __init__(self) -> None:
        self.SemKITTI_label_name = dict()
        with open("semantic-kitti.yaml", 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
            self.SemKITTI_label_name[i] = semkittiyaml['color_map'][i]
        self.uniqueClasses = np.unique(list(self.SemKITTI_label_name.keys()))
        self.numberOfClasses = len(self.uniqueClasses)

    def scale_2_255(self,data,minValue,maxValue,dtype=np.uint8):
        """This function maps the data to 0-255"""
        return (((data - minValue)/ float(maxValue-minValue))* 255).astype(dtype)

    def points2BirdEye(self,points,res=0.1,horizontalRange=(-18.4,18.4),verticalRange=(-24.0, 24.0),heightRange=(-2,10)):
        """This fuction groups the point clouds into given resolution which is metric also crops the points that are in the range specified above
        This function returns both BEV representation of the point cloud and mask image
        """

        #excracting the point array into dimentions
        x_points = points[:, 0]
        y_points = points[:, 1]
        z_points = points[:, 2]
        intencity = points[:, 3]
        rgb = points[:, 4:]

        # filtering points according to range given in parameters
        f_filt = np.logical_and((x_points > verticalRange[0]), (x_points < verticalRange[1]))
        s_filt = np.logical_and((y_points > -horizontalRange[1]), (y_points < -horizontalRange[0]))
        filter = np.logical_and(f_filt, s_filt)
        indices = np.argwhere(filter).flatten()

        x_points = x_points[indices]
        y_points = y_points[indices]
        z_points = z_points[indices]
        intencity = intencity[indices]
        rgb = rgb[indices]

        # creating a grid based on x and y values
        x_img = (-y_points / res).astype(np.int32)
        y_img = (-x_points / res).astype(np.int32)

        # since we areforming a image the minimum index should be 0,0
        x_img -= int(np.floor(horizontalRange[0] / res))
        y_img += int(np.ceil(verticalRange[1] / res))

        # the filtering of the height values
        featureValueZ = np.clip(a=z_points,
                        a_min=heightRange[0],
                        a_max=heightRange[1])
        
        # the normalization of the height values and intencity values
        pixel_values = self.scale_2_255(featureValueZ,heightRange[0],heightRange[1])

        pixel_values2 = self.scale_2_255(intencity,min(intencity),max(intencity))

        # forming empty image arrays 
        x_max = 1 + int((horizontalRange[1] - horizontalRange[0]) / res)
        y_max =  int((verticalRange[1] - verticalRange[0]) / res)

        im = np.zeros([y_max, x_max,3], dtype=np.uint8)
        im2 = np.zeros([y_max, x_max,1], dtype=np.uint8)

        # initilayzing point values to corresponding grid
        im[y_img, x_img] = np.vstack([pixel_values,pixel_values2,pixel_values2]).T
        im2[y_img, x_img] = rgb
        #np.expand_dims(im,0)
        return im,im2
    
    def readPoints(self,pointPaths,labelPaths,predictedlabel):
        pointArray = []
        labelArray = []

        
        for i in range(len(pointPaths)):
            scan = np.fromfile(pointPaths[i], dtype=np.float32)
            scan = scan.reshape((-1, 4))
            
            
            scan2 = np.fromfile(labelPaths[i], dtype=np.uint32)
            scan2 = scan2.reshape((-1))
            sem_label = scan2 & 0xFFFF  # label
            
            
            for a in range(len(self.uniqueClasses)):
                sem_label[sem_label==self.uniqueClasses[a]] = a
            
            pointswithRGB = np.append(scan,np.expand_dims(sem_label,axis=1),axis=1)
            pt  = self.BirdEye2points(pointswithRGB,predictedlabel)
            pointArray.append(pt)
           

        return np.array(pointArray)

    def BirdEye2points(self,points,resultLabels,res=0.1,horizontalRange=(-18.4,18.4),verticalRange=(-24.0, 24.0),heightRange=(-2,10)):
        """This fuction groups the point clouds into given resolution which is metric also crops the points that are in the range specified above
        This function returns points which are classified.
        """

        #excracting the point array into dimentions
        x_points = points[:, 0]
        y_points = points[:, 1]
        z_points = points[:, 2]
        intencity = points[:, 3]
        rgb = points[:, 4:]

        # filtering points according to range given in parameters
        f_filt = np.logical_and((x_points > verticalRange[0]), (x_points < verticalRange[1]))
        s_filt = np.logical_and((y_points > -horizontalRange[1]), (y_points < -horizontalRange[0]))
        filter = np.logical_and(f_filt, s_filt)
        indices = np.argwhere(filter).flatten()

        x_points = x_points[indices]
        y_points = y_points[indices]
        z_points = z_points[indices]
        intencity = intencity[indices]
        rgb = rgb[indices]

        filteredPoints = np.vstack([x_points,y_points,z_points]).T

        # creating a grid based on x and y values
        x_img = (-y_points / res).astype(np.int32)
        y_img = (-x_points / res).astype(np.int32)

        # since we areforming a image the minimum index should be 0,0
        x_img -= int(np.floor(horizontalRange[0] / res))
        y_img += int(np.ceil(verticalRange[1] / res))

        
        filteredPoints = np.append(filteredPoints,resultLabels[y_img,x_img],1)
        # for i in range(len(filteredPoints)):
        #     np.append(filteredPoints[i],resultLabels[y_img[i],x_img[i]])

        
        return filteredPoints

    def processBanchOfData(self,pointPaths,labelPaths,isTrain=True):
        """Convert batch of data to images"""
        pointArray = []
        labelArray = []

        
        for i in range(len(pointPaths)):
            scan = np.fromfile(pointPaths[i], dtype=np.float32)
            scan = scan.reshape((-1, 4))
            
            
            scan2 = np.fromfile(labelPaths[i], dtype=np.uint32)
            scan2 = scan2.reshape((-1))
            sem_label = scan2 & 0xFFFF  # label
            
            
            for a in range(len(self.uniqueClasses)):
                sem_label[sem_label==self.uniqueClasses[a]] = a
            
            pointswithRGB = np.append(scan,np.expand_dims(sem_label,axis=1),axis=1)
            pt , label = self.points2BirdEye(pointswithRGB)
            pointArray.append(pt)
            labelArray.append(label)

        return np.array(pointArray),np.array(labelArray)

class squenceDataGenerator(Sequence):
    def __init__(self,x_set,y_set,batch_size) -> None:
        super().__init__()
        self.x , self.y = x_set,y_set
        self.batch_size = batch_size
        self.preprocess = dataPreProcessor()
        self.numberOfClasses = self.preprocess.numberOfClasses

    
    def __len__(self):
        return int(np.ceil(len(self.x)/ float(self.batch_size)))

    def __getitem__(self, index):
        XBatch = self.x[index*self.batch_size:(index+1)*self.batch_size]
        YBatch = self.y[index*self.batch_size:(index+1)*self.batch_size]
        
        return self.preprocess.processBanchOfData(XBatch,YBatch)
