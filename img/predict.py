import os,torch,glob,cv2
from torch.autograd import Variable
from PIL import Image,ImageFile
import numpy as np
import imghdr
from models.seg_net import Segnet
from utils.dataset import ISBI_Loader as loader
palette = [0, 0, 0, 255, 255, 255]
def create_path(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask
import torchvision.transforms as transforms
mean_std = ([0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
# 将input变成tensor
input_transform = transforms.Compose([
    transforms.ToTensor(),
    ##如果是numpy或者pil image格式，会将[0,255]转为[0,1]，并且(hwc)转为(chw)
    transforms.Normalize(*mean_std) #[0,1]  ---> 符合imagenet的范围[-2.117,2.248][,][,]
])
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Segnet(1,1)
    try:
        net.to(device)
        net.load_state_dict(torch.load('best_model.pth',map_location=device))
        net.eval()
    except Exception as e:
        print(e)
    testpath = glob.glob('./sarData/test/image/*.png')
    create_path('./sarData/test/predict')
    #读所有图片
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    for test_path in testpath:
        #保存结果地址
        save_res_path = test_path.split('\\')[1]
        save_img = './sarData/test/predict\\'+save_res_path

        #读取图片
        img = Image.open(test_path).convert("RGB")
        img = np.array(img)
        # img = cv2.imread(test_path)
        #转为灰度图
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = img.reshape(1,1,img.shape[0],img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        #将tensor拷贝到设备中
        img_tensor = img_tensor.to(device=device,dtype=torch.float32)
        #预测
        pred = net(img_tensor)
        #提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        #处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        #存图
        cv2.imwrite(save_img,pred)
        print("iamge:{} is saved".format(save_res_path))