# forklift-person-accident-detection

This is an implementation of a system capable of detecting forklifts and people, estimate if a situation involving them is dangerous (the forklift moving towards the person at a significant speed at a close distance) and notifying accordingly.

### Requirements

The system was designed to be run on a raspberry pi 4b or later, running ubuntu 24.04. A usb camera is assumed to provide the image input. A Google Coral TPU Accelerator is also required for the object detection process. The "danger" indication is provided as a 2 second high output at GPIO pin 17.

### Parameter Configuration
The necessary parameters can be configured through the config.yaml file:

```
model_path: "best_full_integer_quant_edgetpu.tflite"   #path to the trained model in .tflite format
cam_path: 0                                            #path for the usb camera (0 by default)
det_interval: 2000                                     #time between detections in ms-only tracking is performed during that time
conf_threshold: 0.2                                    #confidence threshold for the detection model
vert_d_forklift_front_perc: 0.5
vert_d_human_front_perc: 0.4
cond1_ioh_thr: 0.3                                     #fraction of person bbox intersecing with forklift bbox
cond2_hor_dist_perc: 1.5
speed_tot_height_perc: 0.2
speed_fork_height_perc: 0.15
tracking_rescale: 6                                    #rescaling factor for performing tracking in large images
```

### Included Files

Tn this repository there is the python source code and the config folder, which contains the file with the parameters mentioned above. In [this]() link there is a .tar file that can be used to load the application's docker image like this:
```
sudo docker load < forklifts_humans.tar
```
The resulting image can be run as shown here:
```
sudo docker run --privileged --device /dev/bus/usb -v $(pwd)/config:/config forklifts_humans
```
$(pwd)/config assumes that the config folder in which the configuration file is located is inside the directory where the above command is executed. Of course the path can be adapted.
