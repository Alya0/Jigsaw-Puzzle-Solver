# imports:
import cv2 as cv
import time
from edge_extraction_functions import edge_extraction
from match_and_order_with_sift import get_classified_pieces, order_pieces
from list_to_image import convert_list_to_image

# code 
image_path = './project_images/24pieces1.png'
hint_image_path = './project_images/24pieces1_solved.png'
hint_image = cv.imread(hint_image_path)
start_time = time.perf_counter()

pieces_borders, puzzle_pieces, puzzle_pieces_top_left_corner = edge_extraction(image_path)

classified_pieces = get_classified_pieces(pieces_borders)

puzzle_list, width, height, width_px, height_px= order_pieces(classified_pieces, pieces_borders, hint_image, puzzle_pieces)

final_image = convert_list_to_image(width, height, width_px, height_px, puzzle_list, puzzle_pieces, puzzle_pieces_top_left_corner)


finish_time = time.perf_counter()
execution_time = finish_time - start_time
print("Algorithm Execution Time: {:.2f} seconds".format(execution_time))
cv.imwrite('./project_images/ff.png', final_image)
# cv.imshow('final image', final_image)

cv.waitKey(0)
