"""
Utilities for running navigation with a LeLaN checkpoint. Most of this code is
borrowed from the LeLaN repository.

@yjy0625
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../external/lelan/train"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../external/lelan/deployment/src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../external/diffusion_policy"))

import clip
import yaml
import torch
import numpy as np
from PIL import Image as PILImage
from lelan_utils import transform_images, transform_images_lelan, load_model


class Vector3:
    """Emulates the Vector3 message in ROS."""
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Vector3(x={self.x}, y={self.y}, z={self.z})"


class Twist:
    """Emulates the Twist message in ROS."""
    def __init__(self, linear=None, angular=None):
        self.linear = linear if linear is not None else Vector3()
        self.angular = angular if angular is not None else Vector3()

    def __repr__(self):
        return f"Twist(linear={self.linear}, angular={self.angular})"

    def set_linear(self, x, y, z):
        """Set linear velocity components."""
        self.linear.x = x
        self.linear.y = y
        self.linear.z = z

    def set_angular(self, x, y, z):
        """Set angular velocity components."""
        self.angular.x = x
        self.angular.y = y
        self.angular.z = z


class LeLaN:
    def __init__(self, ckpt_path=None):
        self.flag_once = 0
        self.store_hist = 0
        self.init_hist = 0
        self.image_hist = []
        self.feat_text = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        dirname = os.path.dirname(__file__)
        model_config_path = os.path.join(dirname, "../../external/lelan/train/config/lelan.yaml")
    
        with open(model_config_path, "r") as f:
            model_params = yaml.safe_load(f)
        
        if ckpt_path is None:
            ckpt_path = os.path.join(dirname, "../../external/lelan/deployment/model_weights/wo_col_loss_wo_temp.pth")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = load_model(
            ckpt_path,
            model_params,
            device,
        )
        model = model.to(device)
        model.eval()

        self.model_params = model_params
        self.model = model

    def forward(self, image, prompt):
        # takes numpy image of shape (224, 224, 3) and language prompt
        if True:
            im = PILImage.fromarray(image)
            
            # initialize image history
            if self.init_hist == 0:
                for ih in range(10):
                    self.image_hist.append(im)
                self.init_hist = 1
                    
            im_obs = [self.image_hist[9], im]
            obs_images, obs_current = transform_images_lelan(
                im_obs, self.model_params["image_size"], center_crop=False)
            obs_images = torch.split(obs_images, 3, dim=1)
            obs_images = torch.cat(obs_images, dim=1)
            batch_obs_images = obs_images.to(self.device)
            batch_obs_current = obs_current.to(self.device)

            with torch.no_grad():
                # text encoder only once at begging
                if self.flag_once == 0:
                    obj_inst = prompt
                    batch_obj_inst = clip.tokenize(obj_inst).to(self.device)            
                    self.feat_text = self.model("text_encoder", inst_ref=batch_obj_inst)
                else:
                    self.flag_once = 1
                
                # LeLaN inference
                if self.model_params["model_type"] == "lelan_col":
                    obsgoal_cond = self.model("vision_encoder", obs_img=batch_obs_images, feat_text=self.feat_text.to(torch.float32), current_img=batch_obs_current) 
                elif self.model_params["model_type"] == "lelan":
                    obsgoal_cond = self.model("vision_encoder", obs_img=batch_obs_current, feat_text=self.feat_text.to(torch.float32)) 
                linear_vel, angular_vel = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            
            vt = linear_vel.cpu().numpy()[0, 0]
            wt = angular_vel.cpu().numpy()[0, 0]

            # maximum linear and angular velocity
            maxv = 0.3
            maxw = 0.5           
            msg_pub = Twist()
            if np.absolute(vt) <= maxv:
                if np.absolute(wt) <= maxw:
                    msg_pub.linear.x = vt
                    msg_pub.linear.y = 0.0
                    msg_pub.linear.z = 0.0
                    msg_pub.angular.x = 0.0
                    msg_pub.angular.y = 0.0
                    msg_pub.angular.z = wt
                else:
                    rd = vt/wt
                    msg_pub.linear.x = maxw * np.sign(vt) * np.absolute(rd)
                    msg_pub.linear.y = 0.0
                    msg_pub.linear.z = 0.0
                    msg_pub.angular.x = 0.0
                    msg_pub.angular.y = 0.0
                    msg_pub.angular.z = maxw * np.sign(wt)
            else:
                if np.absolute(wt) <= 0.001:
                    msg_pub.linear.x = maxv * np.sign(vt)
                    msg_pub.linear.y = 0.0
                    msg_pub.linear.z = 0.0
                    msg_pub.angular.x = 0.0
                    msg_pub.angular.y = 0.0
                    msg_pub.angular.z = 0.0
                else:
                    rd = vt/wt
                    if np.absolute(rd) >= maxv / maxw:
                        msg_pub.linear.x = maxv * np.sign(vt)
                        msg_pub.linear.y = 0.0
                        msg_pub.linear.z = 0.0
                        msg_pub.angular.x = 0.0
                        msg_pub.angular.y = 0.0
                        msg_pub.angular.z = maxv * np.sign(wt) / np.absolute(rd)
                    else:
                        msg_pub.linear.x = maxw * np.sign(vt) * np.absolute(rd)
                        msg_pub.linear.y = 0.0
                        msg_pub.linear.z = 0.0
                        msg_pub.angular.x = 0.0
                        msg_pub.angular.y = 0.0
                        msg_pub.angular.z = maxw * np.sign(wt)

            # update of image history
            image_histx = [im] + self.image_hist[0:9]
            self.image_hist = image_histx
    
            return msg_pub
