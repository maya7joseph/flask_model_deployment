import io
import json

import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image



imagenet_class_index = json.load(open('imagenet_class_index.json'))



def transform_image(image_bytes):
    """
    Transforms user uploaded image and converts to tensor for model input

    Parameters
    ----------
    image_bytes : image bytes
        image file uploaded by user and read in as bytes.

    Returns
    -------
    tensor
        image transformed and converted to tensor.

    """
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    
    image = Image.open(io.BytesIO(image_bytes))

    return my_transforms(image).unsqueeze(0)



def get_confidence(outputs):
    """
    Generates probabilites for outputs of model and returns the top probability

    Parameters
    ----------
    outputs : tensor
        result of forward pass through network.

    Returns
    -------
    confidence of top prediction.

    """
    # Calculate the probability of the selected prediction
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    
    sorted_probs = torch.sort(probabilities, descending=True)

    
    confidence_tensor = sorted_probs[0][0]
    
    
    confidence = confidence_tensor.item()
    
    return round(confidence)
    
    

def get_image_prediction(image_bytes):
    """
    Load model and pass in input image to generate the model's prediction'

    Parameters
    ----------
    image_bytes : read image file
        image file read and passed in as bytes.

    Returns
    -------
    prediction : string
        label generated from model prediction.
    confidence : float
        probability associated with prediction label.

    """
    # Load pretrained model
    model = models.resnet50(pretrained=True)
    
    # Set model to evaluation mode
    model.eval()

    # Transform image and convert to tensor
    tensor = transform_image(image_bytes=image_bytes)
    
    # Generate prediction outputs by using the tensor as model input
    outputs = model.forward(tensor)
    
    # Select the prediction with the highest probability
    _, pred = outputs.max(1)
    prediction_index = str(pred.item())
    
    
    # Select only the label name as the prediction (2nd element of each pair of index, label )
    prediction = imagenet_class_index[prediction_index][1]
    
    # Call get_confidence to return top probability for outouts after forward pass  
    confidence = get_confidence(outputs)


    return prediction, confidence
