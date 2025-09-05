import cv2
img = cv2.imread("C:/Users/dives/Downloads/Hair_Atlas(1)/dataset/diamond/images1.jpg")  
if img is None:
    print("Error: Image not loaded")
else:
    print("Image loaded successfully, shape:", img.shape)