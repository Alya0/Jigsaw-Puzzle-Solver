import numpy as np
import cv2 as cv
dir_map = {"left":0, "bottom": 1, "right":2, "top":3}
type_map = {"head":0, "hole":1, "straight":2}

def get_classified_pieces(pieces_borders):
  classified_pieces=[[[], [], []], [[], [], []], [[], [], []], [[], [], []]]
  for i, piece in enumerate(pieces_borders):
    for j, btype in enumerate(piece["border_type"]):
      classified_pieces[j][btype].append(i)
  return classified_pieces

def find_puzzle_dimensions_piece(classified_pieces):
  width_top = len(classified_pieces[dir_map["top"]][type_map["straight"]])
  width_bottom = len(classified_pieces[dir_map["bottom"]][type_map["straight"]])
  height_right = len(classified_pieces[dir_map["right"]][type_map["straight"]])
  height_left = len(classified_pieces[dir_map["left"]][type_map["straight"]])

  return max(width_top, width_bottom), max(height_right, height_left)

def find_puzzle_corners(pieces_borders):
  corners = [0, 0, 0, 0]
  coordinates_straight = [(dir_map["left"], dir_map["top"]), (dir_map["right"], dir_map["top"]), (dir_map["left"], dir_map["bottom"]), (dir_map["bottom"], dir_map["right"])]
  for i, piece in enumerate(pieces_borders):
    for j, tpl in enumerate(coordinates_straight):
      if(piece["border_type"][tpl[0]] == type_map["straight"] and piece["border_type"][tpl[1]] == type_map["straight"]):
        corners[j] = i
  return tuple(corners)

def find_potential_pieces(first_dir, first_type, second_dir, second_type, classified_pieces, pieces_borders, pieces_used):
  """
  first_dir: direction of the first border we want to match(left, top, ..)
  first_type: type of the border we want for the first dir
  same for second

  output: list of pieces where the pieces match the input description
  """
  pot_pieces = []
  for piece in classified_pieces[first_dir][first_type]:
    if(pieces_borders[piece]["border_type"][second_dir] == second_type and not piece in pieces_used ):
      pot_pieces.append(piece)

  return pot_pieces

def normalize(vector):
    """ Normalize a vector to zero mean and unit standard deviation. """
    mean = np.mean(vector)
    std = np.std(vector)
    return (vector - mean) / std if std != 0 else vector - mean

def cross_correlation(vector1, vector2):
    """ Calculate the cross-correlation between two vectors. """
    return np.correlate(normalize(vector1), normalize(vector2), mode='full')

def convert_to_lab(bgr_colors):
    # Convert list to a NumPy array and reshape for OpenCV
    bgr_array = np.array(bgr_colors, dtype=np.uint8).reshape(1, -1, 3)
    # Convert from BGR to LAB
    lab_array = cv.cvtColor(bgr_array, cv.COLOR_BGR2LAB)
    return lab_array.reshape(-1, 3)

def get_color_score(border1, border2):

  border1 = convert_to_lab(border1)
  border2 = convert_to_lab(border2)
  r1, g1, b1 = zip(*border1)
  r2, g2, b2 = zip(*border2)
  # Convert to numpy arrays
  r1, g1, b1 = np.array(r1), np.array(g1), np.array(b1)
  r2, g2, b2 = np.array(r2), np.array(g2), np.array(b2)

  # Calculate cross-correlation for each color channel
  cc_r = cross_correlation(r1, r2)
  cc_g = cross_correlation(g1, g2)
  cc_b = cross_correlation(b1, b2)

  peak_r = np.max(cc_r)
  peak_g = np.max(cc_g)
  peak_b = np.max(cc_b)

  # Combine these peaks to get an overall similarity score
  similarity_score = (peak_r + peak_g + peak_b) / 3

  return similarity_score


def find_best_color_match(piece_wanted, border_wanted, potential_pieces, border_needed, pieces_borders):
  if len(potential_pieces) == 1: return potential_pieces[0]
  colored_border_wanted = pieces_borders[piece_wanted]["colors_border"][border_wanted]
  best_piece = 0
  best_score = -1
  for piece_num in potential_pieces:
    color_border_tmp=pieces_borders[piece_num]["colors_border"][border_needed]

    score = get_color_score(colored_border_wanted, color_border_tmp)
    if score > best_score:
      best_score = score
      best_piece = piece_num
  return best_piece

def build_borders(width, height, puzzle_list, pieces_borders, classified_pieces, pieces_used):
  # top border
  i = 0
  for j in range(1, width - 1):
    left_piece = puzzle_list[i][j-1]
    border_type_left = (1 - pieces_borders[left_piece]["border_type"][dir_map["right"]]) #1  - right border of the left piece
    border_type_top = type_map["straight"]
    pieces = find_potential_pieces(dir_map["top"], border_type_top, dir_map["left"], border_type_left, classified_pieces, pieces_borders, pieces_used)
    #i want to match the right border of the left piece with my left border
    piece = find_best_color_match(left_piece, dir_map["right"], pieces, dir_map["left"], pieces_borders)
    pieces_used.append(piece)
    puzzle_list[i][j] = piece


  # bottom border
  i = height - 1
  for j in range(1, width - 1):
    left_piece = puzzle_list[i][j-1]
    border_type_left = (1 - pieces_borders[left_piece]["border_type"][dir_map["right"]]) #1  - right border of the left piece
    border_type_bottom = type_map["straight"]
    pieces = find_potential_pieces(dir_map["bottom"], border_type_bottom, dir_map["left"], border_type_left, classified_pieces, pieces_borders, pieces_used)
    #i want to match the right border of the left piece with my left border
    piece = find_best_color_match(left_piece, dir_map["right"], pieces, dir_map["left"], pieces_borders)
    pieces_used.append(piece)
    puzzle_list[i][j] = piece

  #left_border
  j=0
  for i in range(1, height-1):
    top_piece = puzzle_list[i-1][j]
    border_type_top = (1 - pieces_borders[top_piece]["border_type"][dir_map["bottom"]]) #bottom border of the above piece
    border_type_left = type_map["straight"]
    pieces = find_potential_pieces(dir_map["left"], border_type_left, dir_map["top"], border_type_top, classified_pieces, pieces_borders, pieces_used)
    #i want to match the bottom border of the top piece with my top border
    piece = find_best_color_match(top_piece, dir_map["bottom"], pieces, dir_map["top"], pieces_borders)
    pieces_used.append(piece)
    puzzle_list[i][j] = piece

  #right_border
  j= width - 1
  for i in range(1, height-1):
    top_piece = puzzle_list[i-1][j]
    border_type_top = (1 - pieces_borders[top_piece]["border_type"][dir_map["bottom"]]) #bottom border of the above piece
    border_type_right = type_map["straight"]
    pieces = find_potential_pieces(dir_map["right"], border_type_right, dir_map["top"], border_type_top, classified_pieces, pieces_borders, pieces_used)
    #i want to match the bottom border of the top piece with my top border
    piece = find_best_color_match(top_piece, dir_map["bottom"], pieces, dir_map["top"], pieces_borders)
    pieces_used.append(piece)
    puzzle_list[i][j] = piece

  return puzzle_list, pieces_used

def find_best_color_match_inner(upper_piece, left_piece, potential_pieces, pieces_borders):
  best_score = -1
  best_piece = 0
  bottom_color_border_for_upper_piece = pieces_borders[upper_piece]["colors_border"][dir_map["bottom"]]
  right_color_border_for_left_piece = pieces_borders[left_piece]["colors_border"][dir_map["right"]]
  for piece_num in potential_pieces:
    upper_color_border = pieces_borders[piece_num]["colors_border"][dir_map["top"]]
    left_color_border = pieces_borders[piece_num]["colors_border"][dir_map["left"]]
    score1 = get_color_score(upper_color_border, bottom_color_border_for_upper_piece)
    score2 = get_color_score(left_color_border, right_color_border_for_left_piece)

    score=score1+score2
    if score > best_score:
      best_score = score
      best_piece = piece_num
  return best_piece

def build_inner_pieces(width, height, puzzle_list, pieces_borders, classified_pieces, pieces_used):
  for i in range(1, height-1):
    for j in range(1, width - 1):
      top_piece = puzzle_list[i-1][j]
      left_piece = puzzle_list[i][j-1]

      border_type_top = 1 - (pieces_borders[top_piece]["border_type"][dir_map["bottom"]])
      border_type_left = 1 - (pieces_borders[left_piece]["border_type"][dir_map["right"]])

      pieces = find_potential_pieces(dir_map["top"], border_type_top, dir_map["left"], border_type_left, classified_pieces, pieces_borders, pieces_used)
      piece = find_best_color_match_inner(upper_piece=top_piece, left_piece=left_piece, potential_pieces=pieces, pieces_borders=pieces_borders)

      pieces_used.append(piece)
      puzzle_list[i, j] = piece

  return puzzle_list

def find_puzzle_dimensions_pixels(classified_pieces, pieces_borders):
  width_top = width_bottom = height_right = height_left = 0
  for piece_num in classified_pieces[3][2]:
    width_top += len(pieces_borders[piece_num]["border"][dir_map["top"]])
  for piece_num in classified_pieces[1][2]:
    width_bottom += len(pieces_borders[piece_num]["border"][dir_map["bottom"]])
  for piece_num in classified_pieces[2][2]:
    height_right += len(pieces_borders[piece_num]["border"][dir_map["right"]])
  for piece_num in classified_pieces[0][2]:
    height_left += len(pieces_borders[piece_num]["border"][dir_map["left"]])

  return max(width_top, width_bottom), max(height_right, height_left)

def order_pieces(classified_pieces, pieces_borders):
  width, height = find_puzzle_dimensions_piece(classified_pieces)
  puzzle_list = np.zeros((height, width), dtype = int)

  top_left, top_right, bottom_left, bottom_right = find_puzzle_corners(pieces_borders)
  puzzle_list[0][0] = top_left
  puzzle_list[height - 1][0] = bottom_left
  puzzle_list[0][width-1] = top_right
  puzzle_list[height-1][width-1] = bottom_right

  pieces_used = [top_left, top_right, bottom_left, bottom_right]
  # print(pieces_used)
  puzzle_list, pieces_used = build_borders(width, height, puzzle_list, pieces_borders, classified_pieces, pieces_used)
  # print(pieces_used)
  puzzle_list = build_inner_pieces(width, height, puzzle_list, pieces_borders, classified_pieces, pieces_used)
  width_px , height_px = find_puzzle_dimensions_pixels(classified_pieces, pieces_borders)

  return puzzle_list, width, height, width_px, height_px
