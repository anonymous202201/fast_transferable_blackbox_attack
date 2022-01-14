import ast


def get_imagenet_normalize():
  img_mean = (0.485, 0.456, 0.406)
  img_std = (0.229, 0.224, 0.225)
  return img_mean, img_std


def load_imagenet_label_dict() -> dict:
  with open(
      "fta/utils/dataset_utils/imagenet_label_dict.txt", "r") as f:
    contents = f.read()
    dictionary = ast.literal_eval(contents)
  return dictionary
