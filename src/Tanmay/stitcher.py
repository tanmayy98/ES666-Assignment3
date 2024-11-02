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
        self.feature_detector = cv2.SIFT_create()
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
        self.feature_detector = cv2.SIFT_create()

        # FLANN-based matcher for better performance
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)  # Specify how many times the tree should be traversed
        self.feature_matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def make_panaroma_for_images_in(self, path):
        image_files = glob.glob('{}/*.*'.format(path))
        if len(image_files) < 2:
            raise ValueError("Need at least two images to create a panorama")

        input_images = [cv2.imread(file) for file in image_files]
        if any(img is None for img in input_images):
            raise ValueError("Error reading one or more images from the path")

        result_image = input_images[0]
        homography_matrices = []

        for i in range(1, len(input_images)):
            keypoints1, descriptors1 = self.feature_detector.detectAndCompute(result_image, None)
            keypoints2, descriptors2 = self.feature_detector.detectAndCompute(input_images[i], None)

            if descriptors1 is None or descriptors2 is None:
                logger.warning(f"No descriptors found in image {i}. Skipping this pair.")
                continue

            matches = self.feature_matcher.knnMatch(descriptors1, descriptors2, k=2)

            # Ratio test for matches without cross-checking
            good_matches = []
            for match1, match2 in matches:
                if match1.distance < 0.85 * match2.distance:
                    good_matches.append(match1)

            if len(good_matches) < 6:
                logger.warning(f"Not enough good matches between image {i} and image {i-1}. Skipping this pair.")
                continue

            points1 = np.float32([keypoints1[match.queryIdx].pt for match in good_matches]).reshape(-1, 2)
            points2 = np.float32([keypoints2[match.trainIdx].pt for match in good_matches]).reshape(-1, 2)

            homography_matrix = self.compute_homography(points2, points1)
            if homography_matrix is None:
                logger.warning(f"Failed to compute homography for image {i} and image {i-1}. Skipping this pair.")
                continue

            homography_matrices.append(homography_matrix)
            result_image = self.inverse_warp(result_image, input_images[i], homography_matrix)

        return result_image, homography_matrices

    def normalize_points(self, points):
        mean = np.mean(points, axis=0)
        std_dev = np.std(points, axis=0)
        std_dev[std_dev < 1e-8] = 1e-8  # avoiding division by zero (adding a small epsilon)
        scaling = np.sqrt(2) / std_dev
        transformation_matrix = np.array([
            [scaling[0], 0, -scaling[0] * mean[0]],
            [0, scaling[1], -scaling[1] * mean[1]],
            [0, 0, 1]
        ])
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        normalized_points = (transformation_matrix @ points_homogeneous.T).T
        return normalized_points[:, :2], transformation_matrix

    def dlt(self, points1, points2):
        points1_norm, transform1 = self.normalize_points(points1)
        points2_norm, transform2 = self.normalize_points(points2)
        matrix_A = []
        for i in range(len(points1_norm)):
            x, y = points1_norm[i]
            x_prime, y_prime = points2_norm[i]
            matrix_A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
            matrix_A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
        matrix_A = np.array(matrix_A)
        try:
            U, S, Vt = np.linalg.svd(matrix_A)
        except np.linalg.LinAlgError:
            logger.warning("SVD did not converge. Returning None for homography.")
            return None
        homography_normalized = Vt[-1].reshape(3, 3)
        homography_denormalized = np.linalg.inv(transform2) @ homography_normalized @ transform1
        return homography_denormalized / homography_denormalized[2, 2]

    def compute_homography(self, points1, points2):
        max_iterations = 2000
        inlier_threshold = 3.0
        best_homography = None
        max_inliers = 0
        optimal_inliers = []

        if len(points1) < 4:
            return None

        for _ in range(max_iterations):
            random_indices = np.random.choice(len(points1), 4, replace=False)
            sampled_points1 = points1[random_indices]
            sampled_points2 = points2[random_indices]

            candidate_homography = self.dlt(sampled_points1, sampled_points2)
            if candidate_homography is None:
                continue

            points1_homogeneous = np.hstack((points1, np.ones((points1.shape[0], 1))))
            projected_points2 = (candidate_homography @ points1_homogeneous.T).T

            projected_points2[projected_points2[:, 2] == 0, 2] = 1e-10
            projected_points2 /= projected_points2[:, 2, np.newaxis]
            projected_points2 = projected_points2[:, :2]

            errors = np.linalg.norm(points2 - projected_points2, axis=1)
            current_inliers = np.where(errors < inlier_threshold)[0]

            if len(current_inliers) > max_inliers:
                max_inliers = len(current_inliers)
                best_homography = candidate_homography
                optimal_inliers = current_inliers

            if len(current_inliers) > 0.8 * len(points1):
                break

        if best_homography is not None and len(optimal_inliers) >= 10:
            best_homography = self.dlt(points1[optimal_inliers], points2[optimal_inliers])
        else:
            logger.warning("Not enough inliers after RANSAC.")
            return None

        return best_homography

    def apply_homography_to_points(self, homography_matrix, points):
        points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed_points = (homography_matrix @ points_homogeneous.T).T
        transformed_points[transformed_points[:, 2] == 0, 2] = 1e-10
        transformed_points /= transformed_points[:, 2, np.newaxis]
        return transformed_points[:, :2]

    def warp_image(self, image1, image2, homography_matrix, output_shape):
        height_out, width_out = output_shape
        grid_x, grid_y = np.meshgrid(np.arange(width_out), np.arange(height_out))
        ones = np.ones_like(grid_x)
        grid_coords = np.stack([grid_x, grid_y, ones], axis=-1).reshape(-1, 3)

        homography_inverse = np.linalg.inv(homography_matrix)
        transformed_coords = grid_coords @ homography_inverse.T
        transformed_coords[transformed_coords[:, 2] == 0, 2] = 1e-10
        transformed_coords /= transformed_coords[:, 2, np.newaxis]

        x_src = transformed_coords[:, 0]
        y_src = transformed_coords[:, 1]

        valid = (
            (x_src >= 0) & (x_src < image2.shape[1] - 1) &
            (y_src >= 0) & (y_src < image2.shape[0] - 1)
        )

        x_src = x_src[valid]
        y_src = y_src[valid]
        x0 = np.floor(x_src).astype(np.int32)
        y0 = np.floor(y_src).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        weight_x = x_src - x0
        weight_y = y_src - y0

        flat_image2 = image2.reshape(-1, image2.shape[2])
        index = y0 * image2.shape[1] + x0
        Ia = flat_image2[index]
        Ib = flat_image2[y0 * image2.shape[1] + x1]
        Ic = flat_image2[y1 * image2.shape[1] + x0]
        Id = flat_image2[y1 * image2.shape[1] + x1]

        wa = (1 - weight_x) * (1 - weight_y)
        wb = weight_x * (1 - weight_y)
        wc = (1 - weight_x) * weight_y
        wd = weight_x * weight_y

        warped_image2 = np.zeros((height_out, width_out, image2.shape[2]), dtype=image2.dtype)
        warped_image2[grid_y.reshape(-1)[valid], grid_x.reshape(-1)[valid]] = (
            wa[:, None] * Ia + wb[:, None] * Ib + wc[:, None] * Ic + wd[:, None] * Id
        )

        combined_image = np.maximum(image1, warped_image2)
        return combined_image

    def inverse_warp(self, image1, image2, homography_matrix):
        width_combined = image1.shape[1] + image2.shape[1]
        height_combined = max(image1.shape[0], image2.shape[0])
        return self.warp_image(image1, image2, homography_matrix, (height_combined, width_combined))

        # FLANN-based matcher for better performance
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)  # Specify how many times the tree should be traversed
        self.feature_matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def make_panaroma_for_images_in(self, path):
        image_files = glob.glob('{}/*.*'.format(path))
        if len(image_files) < 2:
            raise ValueError("Need at least two images to create a panorama")

        input_images = [cv2.imread(file) for file in image_files]
        if any(img is None for img in input_images):
            raise ValueError("Error reading one or more images from the path")

        result_image = input_images[0]
        homography_matrices = []

        for i in range(1, len(input_images)):
            keypoints1, descriptors1 = self.feature_detector.detectAndCompute(result_image, None)
            keypoints2, descriptors2 = self.feature_detector.detectAndCompute(input_images[i], None)

            if descriptors1 is None or descriptors2 is None:
                logger.warning(f"No descriptors found in image {i}. Skipping this pair.")
                continue

            matches = self.feature_matcher.knnMatch(descriptors1, descriptors2, k=2)

            # Ratio test for matches without cross-checking
            good_matches = []
            for match1, match2 in matches:
                if match1.distance < 0.85 * match2.distance:
                    good_matches.append(match1)

            if len(good_matches) < 6:
                logger.warning(f"Not enough good matches between image {i} and image {i-1}. Skipping this pair.")
                continue

            points1 = np.float32([keypoints1[match.queryIdx].pt for match in good_matches]).reshape(-1, 2)
            points2 = np.float32([keypoints2[match.trainIdx].pt for match in good_matches]).reshape(-1, 2)

            homography_matrix = self.compute_homography(points2, points1)
            if homography_matrix is None:
                logger.warning(f"Failed to compute homography for image {i} and image {i-1}. Skipping this pair.")
                continue

            homography_matrices.append(homography_matrix)
            result_image = self.inverse_warp(result_image, input_images[i], homography_matrix)

        return result_image, homography_matrices

    def normalize_points(self, points):
        mean = np.mean(points, axis=0)
        std_dev = np.std(points, axis=0)
        std_dev[std_dev < 1e-8] = 1e-8  # avoiding division by zero (adding a small epsilon)
        scaling = np.sqrt(2) / std_dev
        transformation_matrix = np.array([
            [scaling[0], 0, -scaling[0] * mean[0]],
            [0, scaling[1], -scaling[1] * mean[1]],
            [0, 0, 1]
        ])
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        normalized_points = (transformation_matrix @ points_homogeneous.T).T
        return normalized_points[:, :2], transformation_matrix

    def dlt(self, points1, points2):
        points1_norm, transform1 = self.normalize_points(points1)
        points2_norm, transform2 = self.normalize_points(points2)
        matrix_A = []
        for i in range(len(points1_norm)):
            x, y = points1_norm[i]
            x_prime, y_prime = points2_norm[i]
            matrix_A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
            matrix_A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
        matrix_A = np.array(matrix_A)
        try:
            U, S, Vt = np.linalg.svd(matrix_A)
        except np.linalg.LinAlgError:
            logger.warning("SVD did not converge. Returning None for homography.")
            return None
        homography_normalized = Vt[-1].reshape(3, 3)
        homography_denormalized = np.linalg.inv(transform2) @ homography_normalized @ transform1
        return homography_denormalized / homography_denormalized[2, 2]

    def compute_homography(self, points1, points2):
        max_iterations = 2000
        inlier_threshold = 3.0
        best_homography = None
        max_inliers = 0
        optimal_inliers = []

        if len(points1) < 4:
            return None

        for _ in range(max_iterations):
            random_indices = np.random.choice(len(points1), 4, replace=False)
            sampled_points1 = points1[random_indices]
            sampled_points2 = points2[random_indices]

            candidate_homography = self.dlt(sampled_points1, sampled_points2)
            if candidate_homography is None:
                continue

            points1_homogeneous = np.hstack((points1, np.ones((points1.shape[0], 1))))
            projected_points2 = (candidate_homography @ points1_homogeneous.T).T

            projected_points2[projected_points2[:, 2] == 0, 2] = 1e-10
            projected_points2 /= projected_points2[:, 2, np.newaxis]
            projected_points2 = projected_points2[:, :2]

            errors = np.linalg.norm(points2 - projected_points2, axis=1)
            current_inliers = np.where(errors < inlier_threshold)[0]

            if len(current_inliers) > max_inliers:
                max_inliers = len(current_inliers)
                best_homography = candidate_homography
                optimal_inliers = current_inliers

            if len(current_inliers) > 0.8 * len(points1):
                break

        if best_homography is not None and len(optimal_inliers) >= 10:
            best_homography = self.dlt(points1[optimal_inliers], points2[optimal_inliers])
        else:
            logger.warning("Not enough inliers after RANSAC.")
            return None

        return best_homography

    def apply_homography_to_points(self, homography_matrix, points):
        points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed_points = (homography_matrix @ points_homogeneous.T).T
        transformed_points[transformed_points[:, 2] == 0, 2] = 1e-10
        transformed_points /= transformed_points[:, 2, np.newaxis]
        return transformed_points[:, :2]

    def warp_image(self, image1, image2, homography_matrix, output_shape):
        height_out, width_out = output_shape
        grid_x, grid_y = np.meshgrid(np.arange(width_out), np.arange(height_out))
        ones = np.ones_like(grid_x)
        grid_coords = np.stack([grid_x, grid_y, ones], axis=-1).reshape(-1, 3)

        homography_inverse = np.linalg.inv(homography_matrix)
        transformed_coords = grid_coords @ homography_inverse.T
        transformed_coords[transformed_coords[:, 2] == 0, 2] = 1e-10
        transformed_coords /= transformed_coords[:, 2, np.newaxis]

        x_src = transformed_coords[:, 0]
        y_src = transformed_coords[:, 1]

        valid = (
            (x_src >= 0) & (x_src < image2.shape[1] - 1) &
            (y_src >= 0) & (y_src < image2.shape[0] - 1)
        )

        x_src = x_src[valid]
        y_src = y_src[valid]
        x0 = np.floor(x_src).astype(np.int32)
        y0 = np.floor(y_src).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        weight_x = x_src - x0
        weight_y = y_src - y0

        flat_image2 = image2.reshape(-1, image2.shape[2])
        index = y0 * image2.shape[1] + x0
        Ia = flat_image2[index]
        Ib = flat_image2[y0 * image2.shape[1] + x1]
        Ic = flat_image2[y1 * image2.shape[1] + x0]
        Id = flat_image2[y1 * image2.shape[1] + x1]

        wa = (1 - weight_x) * (1 - weight_y)
        wb = weight_x * (1 - weight_y)
        wc = (1 - weight_x) * weight_y
        wd = weight_x * weight_y

        warped_image2 = np.zeros((height_out, width_out, image2.shape[2]), dtype=image2.dtype)
        warped_image2[grid_y.reshape(-1)[valid], grid_x.reshape(-1)[valid]] = (
            wa[:, None] * Ia + wb[:, None] * Ib + wc[:, None] * Ic + wd[:, None] * Id
        )

        combined_image = np.maximum(image1, warped_image2)
        return combined_image

    def inverse_warp(self, image1, image2, homography_matrix):
        width_combined = image1.shape[1] + image2.shape[1]
        height_combined = max(image1.shape[0], image2.shape[0])
        return self.warp_image(image1, image2, homography_matrix, (height_combined, width_combined))
