import cv2 as cv
import numpy as np
from collections import Counter
import itertools
import math

def most_frequent_color(image):
    """
    return the most common color in the input image.
    """
    pixels = image.reshape(-1, image.shape[-1])
    count = Counter(map(tuple, pixels))
    most_common = count.most_common(1)[0][0]
    return most_common

def create_mask(image, color):
    """
    inputs:
      image : image to create mask from
      color : most common color in the image
    Returns a binary mask where the input color is 0 and everything else is 1.
    """
    mask = np.all(image == color, axis=-1)
    return 1 - mask.astype(np.uint8)    

def extract_puzzle_pieces(image_path):
  """
  input: image path
  Returns a list with the puzzle pieces' contours, the image after loading it and the mask of the puzzle pieces.
  """
  image = cv.imread(image_path)

  mfc = most_frequent_color(image)

  mask = create_mask(image, mfc)

  # Apply morphological operations(opening to remove noise, then closing):
  kernel = np.ones((5,5), np.uint8)
  mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
  mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

  # Find contours with using the mask
  contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

  return contours, image, mask

def and_operation(puzzle_piece, mask_piece):
  """
  performs an "and" operation between two images and returns the result.
  """
  binary_image_3_channel = np.stack((mask_piece,)*3, axis=-1)
  result_image = np.bitwise_and(puzzle_piece, binary_image_3_channel * 255)
  return result_image

def get_border_mask(mask_piece):
  """
  mask_piece: binary image where 1 represents the puzzle piece, and 0 the background
  Returns a binary mask for the border of the piece.
  """
  kernel = np.ones((3,3), np.uint8)
  eroded_mask = cv.erode(mask_piece, kernel, iterations = 1)
  border_mask = mask_piece - eroded_mask
  return border_mask

def distance_px(val1, val2):
  """
  Calculates hamming distance between two points.
  """
  return abs(val1[0] - val2[0]) + abs(val1[1] - val2[1])

def sort_points(points):
    """
    Sorts a set of (x, y) points clockwise.
    """
    centroid = [sum(x) / len(points) for x in zip(*points)]
    def angle(point):
        return math.atan2(point[1] - centroid[1], point[0] - centroid[0])
    sorted_points = sorted(points, key=angle)
    return sorted_points

def distance(x, y, a, b):
    """
    Calculates euclidean distance between two points.
    """
    return math.sqrt((a - x)**2 + (b - y)**2)

def edge_length_error(points):
  """
  input: a set of 4 (x, y) points that represent a quadrilateral.
  returns the sum of difference of each side length to their average length.
  """
  dist = []
  avg_dist = 0
  for i in range(0, len(points)):
    j = i - 1
    if j < 0: j = len(points) - 1
    tmp = distance(*points[i], *points[j])
    avg_dist += tmp
    dist.append(tmp)
  avg_dist /= 4
  error = 0
  for val in dist:
    error += abs(avg_dist - val)
  return error

def area_with_mask(points, mask_piece):
  """
  input:
    points: a set of 4 (x, y) points that represent a quadrilateral
    mask_piece: a binary mask where 1 represents the image, 0 the background
  At first, the points are converted to a binary mask, where 1 represents the polygon and 0 otherwise.
  Returns the shared bits between the two masks (which is the shared space between them)
  """
  height, width = mask_piece.shape
  tmp_mask = np.zeros((height, width), dtype=np.uint8)
  reversed_points = [[y,x] for x, y in points]
  points_array = np.array([reversed_points], dtype=np.int32)
  cv.fillPoly(tmp_mask, points_array, 1)
  bitwise_and_result = cv.bitwise_and(mask_piece, tmp_mask)
  area = np.count_nonzero(bitwise_and_result)
  return area

def find_largest_quadrilateral(points, mask_piece):
    """
    input:
      points: a list of (x, y) points
      mask_piece: a binary mask where 1 represents the image, 0 the background
    from the given point list we iterate through all possible sets of 4 points,
    then this function returns the quadrilateral that most covers the mask piece given, and that closely resembles a square.
    """
    max_area = -1e10
    max_quad = points[:4]

    square_treshold = 100

    # iterate
    for quad in itertools.combinations(points, 4):
        quad = sort_points(quad)
        # get area shared
        area = area_with_mask(quad, mask_piece)
        # get how far the sides are from a sqaure 
        edges_error = edge_length_error(quad)
        area -= square_treshold * edges_error
        if area > max_area:
            max_area = area
            max_quad = quad

    return max_quad

def get_corner_points_2(border_mask, mask_piece):
  """
  This function is to get the four corners of the puzzle piece.
  """
  # get corners from border_mask using cornerHarris
  gray = np.float32(border_mask)
  gray = cv.GaussianBlur(gray, (7, 7), 0)
  gray = cv.GaussianBlur(gray, (7, 7), 0)
  dst = cv.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
  dst = cv.dilate(dst, None)
  corner_mask = dst >  0.03 * dst.max()

  corner_points = []
  for i in range(corner_mask.shape[0]):
    for j in range(corner_mask.shape[1]):
      if(corner_mask[i, j]):
        corner_points.append((i, j))

  # the corner points obatined are of large number, to folllowing code is to reduce the number of points by keeping only one in each 20 * 20 square.
  thresh = 20
  pot_points = {}
  for point in corner_points:
    x , y = point
    found = False
    keys_to_del = []
    for key, val in pot_points.items():
      if distance_px(point, key) <= thresh:
        if  dst[x, y] > val:
          keys_to_del.append(key)
          pot_points[point] = dst[x, y]
        found = True
        break
    for key in keys_to_del:
      del pot_points[key]
    if not found:
      pot_points[point] = dst[x, y]

  # gets the largest quadrilateral which represents the corner points.
  final_corner_points = find_largest_quadrilateral(list(pot_points.keys()), mask_piece)
  return final_corner_points

def get_borders(border_mask, mask_piece):
  """
  takes the border mask, identifies the four corner points of the puzzle piece
  nd then returns the points for each border.
  """
  # divide the border into four parts
  border_indicies = np.where(border_mask == 255)
  border_indicies = list(zip(*border_indicies))

  # get the four corner points
  corner_points = get_corner_points_2(border_mask, mask_piece)

  border_indicies = sort_points(border_indicies)

  border_indicies = np.array(border_indicies)
  border_corner_points = []
  for corner in corner_points:
    min_in = -1
    min_val = 10000
    for i, point in enumerate(border_indicies):
      dis = distance_px(point, corner)
      if(dis < min_val):
        min_val = dis
        min_in = i
    border_corner_points.append(min_in)

  borders = []
  border_corner_points.append(len(border_indicies)) #help for later
  for i in range(len(border_corner_points) - 1):
    cur_border = []
    for j in range(border_corner_points[i], border_corner_points[i+1]):
      cur_border.append(border_indicies[j])
    if(i == len(border_corner_points) - 2):
      for j in range(0, border_corner_points[0]):
        cur_border.append(border_indicies[j])
    cur_border = np.array(cur_border)

    borders.append(cur_border)
  return borders, border_indicies[border_corner_points[0]]

def get_border_color(border, puzzle_piece):
  """
  returns an array of the color of the border.
  """
  color = []
  for point in border:
    color.append(puzzle_piece[point[0], point[1]])
  return color

def keep_one_each_unit(border, axis):
  """
  this function removes excess points from the border, 
  by keeping only one "y" to each "x" or one "x" to each "y" depending if the border is horizontal or vertical
  returns the new border.
  """
  # axis = 0 => x-axis
  # axis = 1 => y_axis
  i = axis%2
  points = {}
  for point in border:
    key = point[i]
    val = point[1-i]
    if key in points.keys():
      points[key].append(val)
    else:
      points[key] = [val]
  new_border = []
  for key, val in points.items():
    new_point = np.zeros(2, dtype=int)
    new_point[i] = key
    tmp = 0
    if(axis == 0 or axis == 3):
      tmp = min(val)
    else:
      tmp = max(val)
    new_point[1-i] = tmp
    new_border.append(new_point)
  new_border = np.array(new_border)
  if(axis < 2):
    sorted_indices = np.argsort(new_border[:, i])[::-1]
  else:
    sorted_indices = np.argsort(new_border[:, i])

  new_border = new_border[sorted_indices]

  return new_border

def align_border(border):
    """
    alignt the points of the border on the x-axis, returns the aligned border.
    """
    border = np.array(border)
    # Calculate the angle to rotate
    dx = dy = 0
    dx = border[-1][0] - border[0][0]
    dy = border[-1][1] - border[0][1]
    angle = np.arctan2(dy, dx)

    # Rotation matrix
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])

    # Apply rotation
    aligned_border = np.dot(border - border[0], rotation_matrix)
    return aligned_border

def average_of_points(points):
    """
    takes a set of points and returns min, max, avg value on the y-axis.
    """
    points_array = np.array(points)
    y_values = points_array[:, 1]  # Extracting the y-values
    max_y = np.max(y_values)
    min_y = np.min(y_values)
    avg = np.mean(y_values)
    return min_y, max_y, avg

def edge_extraction(image_path):
  """
  main function here
  takes the image path 
  returns 
    pieces_borders: a dictionary that conatins a list of dictionaries, each dictionary represents a puzzle piece and contains the coordiantes for the 4 borders, the aligned borders, border type and the colors of the border.
    puzzle_pieces: a list of the puzzle pieces after seperating them from the image
    puzzle_pieces_top_left_corner: a list that contains the top left corner of each puzzle piece
  """
  contours, image, mask = extract_puzzle_pieces(image_path)

  pieces_borders = []
  puzzle_pieces = []
  puzzle_pieces_top_left_corner = []

  for i, contour in enumerate(contours):

    x, y, w, h = cv.boundingRect(contour)
    puzzle_piece = image[y-3:y+h+3, x-3:x+w+3]
    mask_piece = mask[y-3:y+h+3, x-3:x+w+3] * 255

    masked_piece = and_operation(image[y:y+h, x:x+w], mask[y:y+h, x:x+w])
    puzzle_pieces.append(masked_piece)

    border_mask = get_border_mask(mask_piece)
    tmp_borders, left_corner = get_borders(border_mask, mask_piece)
    puzzle_pieces_top_left_corner.append(left_corner)

    colors_border=[]
    for j in range(len(tmp_borders)):
      colors_border.append(get_border_color(tmp_borders[j], puzzle_piece))
      tmp_borders[j] = keep_one_each_unit(tmp_borders[j], j)

    borders = {}
    borders["border"] = tmp_borders
    borders["aligned_border"] = []
    borders["border_type"] = []
    borders["colors_border"] = colors_border


    # align border and try to detect if its straight or ..
    for i, border in enumerate(borders["border"]):

      aligned_border = align_border(border)
      borders["aligned_border"].append(aligned_border)

      min, max, _ = average_of_points(aligned_border)
      dif = max - min
      edge_type = 0 # 2 if straight, 0 if ntoo2, 1 if hole
      # classifying the edge based on minimum and maximum y value after aligning it to the x axis
      if(dif <= 10):
        edge_type = 2
      elif(min <= -25):
        edge_type = 1

      borders["border_type"].append(edge_type)
    pieces_borders.append(borders)

  return pieces_borders, puzzle_pieces, puzzle_pieces_top_left_corner