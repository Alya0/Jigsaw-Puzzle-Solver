import numpy as np
import cv2 as cv

def convert_list_to_image(width, height, width_px, height_px, puzzle_list, puzzle_pieces, puzzle_pieces_top_left_corner):
    step_w = width_px // width
    step_h = height_px // height
 
    result_image = np.zeros((height + 5000, width+5000, 3), dtype=np.uint8)
    for i, lst in enumerate(puzzle_list):
        for j, piece in enumerate(lst):
            small_image = puzzle_pieces[piece]
            h, w = small_image.shape[:2]
            I = i * step_h
            J = j * step_w
            top_left = puzzle_pieces_top_left_corner[piece]
            I = max(0, I - top_left[0])
            J = max(0, J - top_left[1])
            section = result_image[I:I+h, J:J+w]
            section = section.astype(np.uint8)
            small_image = small_image.astype(np.uint8)
            or_section = cv.bitwise_or(section, small_image)
            result_image[I:I+h, J:J+w] = or_section
    resized_image = result_image[:height_px, :width_px]
    return resized_image