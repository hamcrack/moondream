import moondream as md
from PIL import Image
import cv2

def extract_coordinates(s):
  cleaned_string = s.strip("[ ]")
  values_str = cleaned_string.split(",")
  values_float = [float(val.strip()) for val in values_str]
  if len(values_float) == 4:
    x1, y1, x2, y2 = values_float
    return x1, y1, x2, y2
  else:
    return None 

model = md.vl(model="moondream-2b-int8.mf")

image = Image.open("2_VLM_Scenario.jpeg")

encoded_image = model.encode_image(image)

prompts = ["Take the scissors",  "Pick the pen" , "Move the controller", "Hold the screwdriver", "Roll the tape"]

# val = input("Please enter prompt: ")
# print("Processing - '" + val + "' prompt...")

# # Generate caption
# caption = model.caption(encoded_image)["caption"]
# print("Caption:", caption)

for val in prompts:
  # Ask questions
  answer1 = model.query(encoded_image, "Please detect the item in the statement: " + val + ", and return the pixel coordinates of its bounding box.")["answer"]
  print("Answer1:", answer1)
  answer2 = model.query(encoded_image, "You are an expert object detector. Please detect the item mentioned in the statement and return the pixel coordinates of its bounding box. The statment is: " + val)["answer"]
  print("Answer2:", answer2)
  
  answer = model.query(encoded_image, "Which bounding box best highlights the item mentioned in the statement: " + val + ". " + answer1 + " or " + answer2)["answer"]
  
  print("Answer:", answer)

  cv2_image = cv2.cvtColor(cv2.imread("2_VLM_Scenario.jpeg"), cv2.COLOR_BGR2RGB)

  ratios = extract_coordinates(answer)
  x1_ratio = ratios[0]
  y1_ratio = ratios[1]
  x2_ratio = ratios[2]
  y2_ratio = ratios[3]
  print("Ratios:", ratios)
  x1 = int(x1_ratio * cv2_image.shape[1])
  y1 = int(y1_ratio * cv2_image.shape[0])
  x2 = int(x2_ratio * cv2_image.shape[1])
  y2 = int(y2_ratio * cv2_image.shape[0])
  print("Coordinates:", x1, y1, x2, y2)

  # Draw bounding box on the image
  cv2.rectangle(cv2_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

  # cv2.imshow("Image", cv2_image)
  # cv2.waitKey(0)

  filename = val + '.jpg'
  print("Saving image as:", filename)
  print("-----------------------------------------------------")
  cv2.imwrite(filename, cv2_image)
  cv2.waitKey(2000)

cv2.destroyAllWindows()