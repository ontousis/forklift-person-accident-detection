# forklift-person-accident-detection

This is an implementation of a system capable of detecting forklifts and people, estimating if a situation involving them is dangerous (the forklift moving towards the person at a significant speed at a close distance) and notifying accordingly.

### Requirements

The system was designed to be run on a raspberry pi 4b or later, running ubuntu 24.04. A usb camera is assumed to provide the image input. A Google Coral TPU Accelerator is also required for the object detection process. The "danger" indication is provided as a 2 second high output at GPIO pin 17, and the corresponding frame, annotated with the bounding box of the potentially dangerous forklift,is sent to a specified url in a POST request, the body of which also contains the corresponding date/time information.

### Parameter Configuration
The necessary parameters can be configured through the config.yaml file:

```
model_path: "best_full_integer_quant_edgetpu.tflite"   #path to the trained model in .tflite format
cam_path: 0                                            #path for the usb camera (0 by default)
det_interval: 2000                                     #time between detections in ms-only tracking is performed during that time
conf_threshold: 0.2                                    #confidence threshold for the detection model
vert_d_forklift_front_perc: 0.5                        #vertical distance between the centers of the boxes as a percentage of forklift width (forklift in front)
vert_d_human_front_perc: 0.4                           #vertical distance between the centers of the boxes as a percentage of forklift width (person in front)
cond1_ioh_thr: 0.3                                     #fraction of person bbox intersecing with forklift bbox
cond2_hor_dist_perc: 1.5                               #horizontal distance between the centers of the boxes as a percentage of forklift width
speed_tot_height_perc: 0.2                             #relative speed of the boxes (foklift_heights/sec)
speed_fork_height_perc: 0.15                           #forklift speed (foklift_heights/sec)
tracking_rescale: 6                                    #downscaling factor for performing tracking in large images
backend_url: "http://192.168.1.5:5000/upload"          #The url where the POST request is sent
```

### Included Files

Tn this repository there is the python source code and the config folder, which contains the file with the parameters mentioned above. A file with the trained yolov8n model used for object detection is also included (in .tflite format, required for inference using the TPU accelerator). In [this](https://drive.google.com/file/d/1Q3i4ucYXsX7tfRSqBWcJbTALbjhao5jP/view?usp=sharing) link there is a .tar file that can be used to load the application's docker image like this:
```
sudo docker load < forklifts_humans_10_7_25.tar
```
The resulting image can be run as shown here:
```
sudo docker run --privileged --network="host" --device /dev/bus/usb -v $(pwd)/config:/config forklifts_humans_10_7_25
```
$(pwd)/config assumes that the config folder in which the configuration file is located is inside the directory where the above command is executed. Of course the path can be adapted.
