This Project is about segmenting point cloud data by using pprojected based methodology. I implement a U-net with using keras framework.

This project written by Mustafa Fatih ŞEN

1. Download the SemanticKitti Dataset url:http://www.semantic-kitti.org/dataset.html#overview
2. Collect all the point datas into velodyne folder and all the labels to labels folder
3. Change the dataPath variable according to settings 
4. Start Training, the program automaticly record the model parameters based on validatiion accuracy
5. in the testpy also change the dataPath and pretrained model path. 
6. it will produce segmented image, point cloud and cofusion matrix
