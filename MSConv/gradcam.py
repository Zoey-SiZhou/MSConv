# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
# import torch
# from torchvision import models, transforms
# from PIL import Image
# import cv2
# import numpy as np
#
# # 定义设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # 加载预训练模型
# model = models.resnet50(pretrained=True).to(device)
# model.eval()
#
# # 选择目标层（根据需要调整）
# target_layers = [model.layer4[0], model.layer4[2]]  # 示例：ResNet50的layer4块中的第0和第2个子层
#
# # 初始化 GradCAM
# cam = GradCAM(
#     model=model,
#     target_layers=target_layers
# )
#
# # 读取图像
# img_path = 'D:/Dataset/test/IJB/IJBB/loose_crop/24.jpg'  # 替换为您的图像路径
# img = cv2.imread(img_path)
# if img is None:
#     raise ValueError(f"无法读取图像：{img_path}")
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# # 重尺寸图像以匹配模型输入大小
# resized_img = cv2.resize(img_rgb, (224, 224))
#
# # 处理输入张量
# input_tensor = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])(Image.fromarray(resized_img)).unsqueeze(0).to(device)
#
# # 生成 GradCAM
# grayscale_cams = cam(input_tensor=input_tensor, targets=None)
#
# # 可视化
# cam_image = show_cam_on_image(resized_img / 255.0, grayscale_cams[0], use_rgb=True)
# cv2.imwrite('gradcam_result.jpg', cam_image)
# print("GradCAM 结果已保存为 'gradcam_result.jpg'")
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
from backbones import get_model  # 确保你已经正确导入模型结构
import sys

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型
model_path = 'D:/PythonProject/arcface_torch/da_result/epoch20/r100_b64x4/model.pt'  # 替换为你的模型路径
model = get_model('da100', dropout=0, fp16=False).to(device)  # 使用自定义的网络架构
model.load_state_dict(torch.load(model_path, map_location=device))
# model = models.resnet50(pretrained=True).to(device)
model.eval()

# 选择目标层（根据你的模型架构）
target_layers = [model.layer3[29]]  # 根据之前的代码，使用layer4[2]作为目标层

# 初始化 GradCAM
cam = GradCAM(
    model=model,
    target_layers=target_layers
)

# 读取图像
img_path = 'C:/Users/Administrator/Desktop/face_images/45.jpg'  # 替换为你的图像路径
img = cv2.imread(img_path)
if img is None:
    raise ValueError(f"无法读取图像：{img_path}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 重尺寸图像以匹配模型输入大小
resized_img = cv2.resize(img_rgb, (112, 112))  # 调整为与训练时的输入大小一致

# 处理输入张量
input_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 使用与训练时相同的均值和标准差
])(Image.fromarray(resized_img)).unsqueeze(0).to(device)

# 生成 GradCAM
grayscale_cams = cam(input_tensor=input_tensor, targets=None)

# 可视化
cam_image = show_cam_on_image(resized_img / 255.0, grayscale_cams[0], use_rgb=True, colormap=cv2.COLORMAP_JET)
cv2.imwrite('C:/Users/Administrator/Desktop/da/da100_layer3_29.jpg', cam_image)
print("GradCAM 结果已保存为 'gradcam_result.jpg'")
