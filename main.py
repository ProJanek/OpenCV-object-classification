import functions as f

# Select the number of the first displayed image.
image_number = 48

# Create images path list.
path = "shapes"
images_paths = f.create_image_path_list(path)

# Convert images to grey scale.
images_grey = f.convert_to_grey_scale(images_paths)
f.save_four_images(images_grey, image_number, "1_images_in grey_scale.png")

# Perform thresholding on images.
images_thresh = f.thresholding(images_grey)
f.save_four_images(images_thresh, image_number,
                   "2_images_after_thresholding.png")

# Perform morphological operations on images.
images_closing = f.closing(images_thresh, 5)
f.save_four_images(images_closing, image_number, "3_images_after_closing.png")

images_opening = f.opening(images_closing, 3)
f.save_four_images(images_opening, image_number, "4_images_after_opening.png")

images_ero = f.erosion(images_opening, 5, 4)
f.save_four_images(images_ero, image_number, "5_images_after_erosion.png")

images_dil = f.dilation(images_ero, 5, 2)
f.save_four_images(images_dil, image_number, "6_images_after_dilation.png")

# Determine the edges of objects 
edges = f.canny_edge(images_ero)
f.save_four_images(edges, image_number, "7_edges.png")

# Find centroid of each shape
centroids = f.find_centroid(edges)

# Calculate shape coeffisients for each shap
shape_coeffisients = f.find_shape_coefficients(edges, centroids)

# Assign the results and calculate the accuracy
result, accuracy = f.identification(shape_coeffisients, path)

print(f"Accuracy of classification: {accuracy}")


