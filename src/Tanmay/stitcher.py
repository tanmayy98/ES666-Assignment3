import cv2
import numpy as np
import glob
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PanaromaStitcher:
    def __init__(self):
        # Use SIFT for feature detection
        self.sift_detector = cv2.SIFT_create()

        # FLANN-based matcher for better performance
        flann_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)  # Specify how many times the tree should be traversed
        self.feature_matcher = cv2.FlannBasedMatcher(flann_params, search_params)

    def make_panaroma_for_images_in(self, input_path):
        image_file_paths = glob.glob('{}/*.*'.format(input_path))
        if len(image_file_paths) < 2:
            raise ValueError("Need at least two images to create a panorama")

        input_images = [cv2.imread(file_path) for file_path in image_file_paths]
        if any(image is None for image in input_images):
            raise ValueError("Error reading one or more images from the path")

        stitched_image_result = input_images[0]
        homography_matrix_collection = []

        for index in range(1, len(input_images)):
            keypoints1, descriptors1 = self.sift_detector.detectAndCompute(stitched_image_result, None)
            keypoints2, descriptors2 = self.sift_detector.detectAndCompute(input_images[index], None)

            if descriptors1 is None or descriptors2 is None:
                logger.warning(f"No descriptors found in image {index}. Skipping this pair.")
                continue

            knn_matches_list = self.feature_matcher.knnMatch(descriptors1, descriptors2, k=2)

            # Ratio test for matches without cross-checking
            valid_matches = []
            for match1, match2 in knn_matches_list:
                if match1.distance < 0.85 * match2.distance:
                    valid_matches.append(match1)

            if len(valid_matches) < 6:
                logger.warning(f"Not enough good matches between image {index} and image {index-1}. Skipping this pair.")
                continue

            points1 = np.float32([keypoints1[match.queryIdx].pt for match in valid_matches]).reshape(-1, 2)
            points2 = np.float32([keypoints2[match.trainIdx].pt for match in valid_matches]).reshape(-1, 2)

            homography_matrix = self.compute_homography(points2, points1)
            if homography_matrix is None:
                logger.warning(f"Failed to compute homography for image {index} and image {index-1}. Skipping this pair.")
                continue

            homography_matrix_collection.append(homography_matrix)
            stitched_image_result = self.inverse_warp(stitched_image_result, input_images[index], homography_matrix)

        return stitched_image_result, homography_matrix_collection

    def normalize_points(self, point_set):
        mean_point = np.mean(point_set, axis=0)
        std_point = np.std(point_set, axis=0)
        std_point[std_point < 1e-8] = 1e-8  # avoiding division by zero (adding a small epsilon)
        scale_factor = np.sqrt(2) / std_point
        translation_matrix = np.array([[scale_factor[0], 0, -scale_factor[0] * mean_point[0]],
                                        [0, scale_factor[1], -scale_factor[1] * mean_point[1]],
                                        [0, 0, 1]])
        point_set_homogeneous = np.hstack((point_set, np.ones((point_set.shape[0], 1))))
        normalized_points_result = (translation_matrix @ point_set_homogeneous.T).T
        return normalized_points_result[:, :2], translation_matrix

    def dlt(self, point_set1, point_set2):
        normalized_points1, transformation1 = self.normalize_points(point_set1)
        normalized_points2, transformation2 = self.normalize_points(point_set2)
        matrix_A = []
        for i in range(len(normalized_points1)):
            x, y = normalized_points1[i]
            x_prime, y_prime = normalized_points2[i]
            matrix_A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
            matrix_A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
        matrix_A = np.array(matrix_A)
        try:
            U, S, Vt = np.linalg.svd(matrix_A)
        except np.linalg.LinAlgError:
            logger.warning("SVD did not converge. Returning None for homography.")
            return None
        normalized_homography_matrix = Vt[-1].reshape(3, 3)
        denormalized_homography_matrix = np.linalg.inv(transformation2) @ normalized_homography_matrix @ transformation1  # Denormalizing
        return denormalized_homography_matrix / denormalized_homography_matrix[2, 2]

    def compute_homography(self, point_set1, point_set2):
        max_iterations = 2000  # Same as before
        error_threshold = 3.0
        best_homography_matrix = None
        max_inliers_count = 0
        best_inliers_indices = []

        if len(point_set1) < 4:
            return None

        for iteration in range(max_iterations):
            random_indices = np.random.choice(len(point_set1), 4, replace=False)
            sampled_points1 = point_set1[random_indices]
            sampled_points2 = point_set2[random_indices]

            candidate_homography_matrix = self.dlt(sampled_points1, sampled_points2)
            if candidate_homography_matrix is None:
                continue

            points1_homogeneous = np.hstack((point_set1, np.ones((point_set1.shape[0], 1))))
            projected_points2_homogeneous = (candidate_homography_matrix @ points1_homogeneous.T).T

            projected_points2_homogeneous[projected_points2_homogeneous[:, 2] == 0, 2] = 1e-10
            projected_points2 = projected_points2_homogeneous[:, :2] / projected_points2_homogeneous[:, 2, np.newaxis]

            error_values = np.linalg.norm(point_set2 - projected_points2, axis=1)
            inliers_indices = np.where(error_values < error_threshold)[0]

            if len(inliers_indices) > max_inliers_count:
                max_inliers_count = len(inliers_indices)
                best_homography_matrix = candidate_homography_matrix
                best_inliers_indices = inliers_indices

            # Early stopping if enough inliers are found
            if len(inliers_indices) > 0.8 * len(point_set1):
                break

        if best_homography_matrix is not None and len(best_inliers_indices) >= 10:
            best_homography_matrix = self.dlt(point_set1[best_inliers_indices], point_set2[best_inliers_indices])
        else:
            logger.warning("Not enough inliers after RANSAC.")
            return None

        return best_homography_matrix

    def apply_homography_to_points(self, homography_matrix, point_set):
        point_set_homogeneous = np.hstack([point_set, np.ones((point_set.shape[0], 1))])
        transformed_point_set = (homography_matrix @ point_set_homogeneous.T).T
        transformed_point_set[transformed_point_set[:, 2] == 0, 2] = 1e-10
        transformed_point_set = transformed_point_set[:, :2] / transformed_point_set[:, 2, np.newaxis]
        return transformed_point_set

    def warp_image(self, source_image, target_image, homography_matrix, output_image_shape):
        output_height, output_width = output_image_shape    # coordinate grid
        xx, yy = np.meshgrid(np.arange(output_width), np.arange(output_height))
        ones_array = np.ones_like(xx)
        coords_array = np.stack([xx, yy, ones_array], axis=-1).reshape(-1, 3)

        inverse_homography_matrix = np.linalg.inv(homography_matrix)
        transformed_coords_array = coords_array @ inverse_homography_matrix.T
        transformed_coords_array[transformed_coords_array[:, 2] == 0, 2] = 1e-10
        transformed_coords_array /= transformed_coords_array[:, 2, np.newaxis]

        x_src_coords = transformed_coords_array[:, 0]  #interpolate
        y_src_coords = transformed_coords_array[:, 1]

        valid_coord_indices = (
            (x_src_coords >= 0) & (x_src_coords < target_image.shape[1] - 1) &
            (y_src_coords >= 0) & (y_src_coords < target_image.shape[0] - 1)
        )

        x_src_coords = x_src_coords[valid_coord_indices]
        y_src_coords = y_src_coords[valid_coord_indices]
        x0_coords = np.floor(x_src_coords).astype(np.int32)
        y0_coords = np.floor(y_src_coords).astype(np.int32)
        x1_coords = x0_coords + 1
        y1_coords = y0_coords + 1

        wx_coords = x_src_coords - x0_coords
        wy_coords = y_src_coords - y0_coords

        # Perform bilinear interpolation
        interpolated_values = (
            (1 - wx_coords) * (1 - wy_coords) * target_image[y0_coords, x0_coords] +
            (wx_coords * (1 - wy_coords) * target_image[y0_coords, x1_coords]) +
            (wy_coords * (1 - wx_coords) * target_image[y1_coords, x0_coords]) +
            (wx_coords * wy_coords * target_image[y1_coords, x1_coords])
        )

        # Create output image
        output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        output_image[valid_coord_indices] = interpolated_values

        return output_image

    def inverse_warp(self, stitched_image, new_image, homography_matrix):
        # Calculate output size
        corners = np.array([[0, 0, 1], [new_image.shape[1], 0, 1], [new_image.shape[1], new_image.shape[0], 1], [0, new_image.shape[0], 1]])
        transformed_corners = self.apply_homography_to_points(homography_matrix, corners)
        
        # Calculate bounding box for the new image
        min_x = min(0, transformed_corners[:, 0].min())
        max_x = max(stitched_image.shape[1], transformed_corners[:, 0].max())
        min_y = min(0, transformed_corners[:, 1].min())
        max_y = max(stitched_image.shape[0], transformed_corners[:, 1].max())

        # Calculate size of the output image
        output_width = int(max_x - min_x)
        output_height = int(max_y - min_y)

        # Create translation matrix to shift the image
        translation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
        stitched_image_translated = cv2.warpPerspective(stitched_image, translation_matrix, (output_width, output_height))
        new_image_warped = self.warp_image(new_image, stitched_image, homography_matrix, (output_height, output_width))

        # Combine images
        stitched_image_translated[new_image_warped > 0] = new_image_warped[new_image_warped > 0]

        return stitched_image_translated

