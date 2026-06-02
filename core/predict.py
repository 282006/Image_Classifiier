import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset , DataLoader
from PIL import Image
import cv2
import os

class CustomCnnModel(nn.Module):
  def __init__(self , x_dim , num_classes):
    super(CustomCnnModel, self).__init__()
    self.x_dim=x_dim
    self.nc = num_classes

    self.conv_layers= nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), #3 for rgb ; 32 output channels
        nn.BatchNorm2d(32), #batch normalization on 32 channels
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1), #3 for rgb ; 32 output channels
        nn.BatchNorm2d(64), #batch normalization on 32 channels
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), #3 for rgb ; 32 output channels
        nn.BatchNorm2d(128), #batch normalization on 32 channels
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), #3 for rgb ; 32 output channels
        nn.BatchNorm2d(256), #batch normalization on 32 channels
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        #4 blocks of convolution
    )

    self._to_linear = None
    self._get_conv_output(self.x_dim) # defines slef._to_linear

    self.fc_layers = nn.Sequential(
        nn.Linear(self._to_linear ,  512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, self.nc)
    )

    pass

  def _get_conv_output(self, x_dim):
    with torch.no_grad():
        dummy = torch.zeros(1, 3, x_dim, x_dim)
        out = self.conv_layers(dummy)
        self._to_linear = out.view(1, -1).size(1)



  def forward(self,x):
    x=self.conv_layers(x)
    x=x.view(x.size(0),-1)
    x=self.fc_layers(x)
    return x


class ImageClassifier():
    def __init__(self , model_path , class_name=None):
        
        # define CNN architecture
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CustomCnnModel(x_dim=128, num_classes=3).to(self.device)
        
        # load CNN arch trained weights
        
        self.model.load_state_dict(torch.load(model_path , map_location=self.device)) #load weights and biases
        self.model.eval() # make model ready for evaluation
        
        # index to label map
        
        # ✅ CORRECT
        if class_name is None:
            self.class_name = {0: "CAT", 1: "DOG", 2: "PERSON"}
        else:
            self.class_name = class_name
        
        # transformation
        
        self.transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
        ])
        
        
    def predict(self , image_path ):
        
        # load image with PILLOW
        # ✅ Add this line after opening the image
        image = Image.open(image_path).convert("RGB")
        input_image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # prediciton
        with torch.no_grad():
            output = self.model(input_image_tensor)
            _,predicted = torch.max(output,1) # 1 is for dim=1
            
        # map --> class
        predicted_class = self.class_name[predicted.item()] #class_name  is dict
        
        # open cv , write text on our input img
        img = cv2.imread(image_path)
        cv2.putText(img, predicted_class, (50, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        output_path = "class_done.jpg"
        cv2.imwrite(output_path , img)
        cwd = os.getcwd()
        output_path = os.path.join(cwd,output_path)
        
        return predicted_class, output_path
        
    
    