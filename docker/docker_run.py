#!/usr/bin/env python
import os

if __name__=="__main__":
#    cmd = "docker run -it --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all -v %s:/host cmr-image" % (os.path.join(os.getcwd(), '..'))
    cmd = "docker run -it --runtime=nvidia -v %s:/host neural-render" % (os.path.join(os.getcwd(), '..'))

    code = os.system(cmd)
