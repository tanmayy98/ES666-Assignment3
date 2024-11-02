import cv2
import numpy as np
import glob
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PanaromaStitcher:
    def __init__(self):
        # Feature detector setup with SIFT
        self.feature_detector = cv2.SIFT_create()

        # Configure FLANN-based matcher
        flann_index_params = dict(algorithm=1, trees=5)
        flann_search_params = dict(checks=50)
        self.flann_matcher = cv2.FlannBasedMatcher(flann_index_params, flann_search_params)

    def make_panaroma_for_images_in(self, directory_path):
        img_paths = glob.glob('{}/*.*'.format(directory_path))
        if len(img_paths) < 2:
            raise ValueError("Panorama creation requires at least two images.")

        images = [cv2.imread(img) for img in img_paths]
        if any(image is None for image in images):
            raise ValueError("Error loading images from the specified path.")

        panorama_image = images[0]
        homography_matrices = []

        for i in range(1, len(images)):
            keypoints_1, descriptors_1 = self.feature_detector.detectAndCompute(panorama_image, None)
            keypoints_2, descriptors_2 = self.feature_detector.detectAndCompute(images[i], None)

            if descriptors_1 is None or descriptors_2 is None:
                logger.warning(f"No descriptors in image {i}; skipping.")
                continue

            # KNN matches with Lowe's ratio test
            knn_matches = self.flann_matcher.knnMatch(descriptors_1, descriptors_2, k=2)
            valid_matches = [m for m, n in knn_matches if m.distance < 0.85 * n.distance]

            if len(valid_matches) < 6:
                logger.warning(f"Insufficient good matches between images {i} and {i-1}. Skipping.")
                continue

            pts_img1 = np.float32([keypoints_1[m.queryIdx].pt for m in valid_matches]).reshape(-1, 2)
            pts_img2 = np.float32([keypoints_2[m.trainIdx].pt for m in valid_matches]).reshape(-1, 2)

            H_matrix = self.compute_homography(pts_img2, pts_img1)
            if H_matrix is None:
                logger.warning(f"Homography calculation failed for images {i} and {i-1}. Skipping.")
                continue

            homography_matrices.append(H_matrix)
            panorama_image = self.inverse_warp(panorama_image, images[i], H_matrix)

        return panorama_image, homography_matrices

    def normalize_points(self, points):
        mean, std_dev = np.mean(points, axis=0), np.std(points, axis=0)
        std_dev[std_dev < 1e-8] = 1e-8  # Avoid zero division
        scale_factor = np.sqrt(2) / std_dev
        transform_matrix = np.array([[scale_factor[0], 0, -scale_factor[0] * mean[0]],
                                     [0, scale_factor[1], -scale_factor[1] * mean[1]],
                                     [0, 0, 1]])
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
        normalized_points = (transform_matrix @ homogeneous_points.T).T
        return normalized_points[:, :2], transform_matrix

    def dlt(self, points_1, points_2):
        norm_pts1, trans_1 = self.normalize_points(points_1)
        norm_pts2, trans_2 = self.normalize_points(points_2)
        A_matrix = []
        for j in range(len(norm_pts1)):
            x1, y1 = norm_pts1[j]
            x2, y2 = norm_pts2[j]
            A_matrix.extend([[-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2],
                             [0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2]])

        A_matrix = np.array(A_matrix)
        try:
            _, _, Vt = np.linalg.svd(A_matrix)
        except np.linalg.LinAlgError:
            logger.warning("SVD did not converge. Returning None for homography.")
            return None
        norm_homography = Vt[-1].reshape(3, 3)
        homography = np.linalg.inv(trans_2) @ norm_homography @ trans_1
        return homography / homography[2, 2]

    def compute_homography(self, pts_1, pts_2):
        ransac_iterations = 2000
        distance_threshold = 3.0
        optimal_H = None
        max_inliers_count = 0
        best_inliers = []

        if len(pts_1) < 4:
            return None

        for _ in range(ransac_iterations):
            sample_indices = np.random.choice(len(pts_1), 4, replace=False)
            sample_pts1, sample_pts2 = pts_1[sample_indices], pts_2[sample_indices]

            candidate_H = self.dlt(sample_pts1, sample_pts2)
            if candidate_H is None:
                continue

            projected_pts = (candidate_H @ np.hstack((pts_1, np.ones((pts_1.shape[0], 1)))).T).T
            projected_pts /= projected_pts[:, 2, np.newaxis]
            reprojection_errors = np.linalg.norm(pts_2 - projected_pts[:, :2], axis=1)
            inliers = np.where(reprojection_errors < distance_threshold)[0]

            if len(inliers) > max_inliers_count:
                max_inliers_count, optimal_H, best_inliers = len(inliers), candidate_H, inliers

            if len(inliers) > 0.8 * len(pts_1):
                break

        return self.dlt(pts_1[best_inliers], pts_2[best_inliers]) if optimal_H is not None and len(best_inliers) >= 10 else None

    def apply_homography_to_points(self, H_matrix, points):
        homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed_points = (H_matrix @ homogeneous_points.T).T
        transformed_points /= transformed_points[:, 2, np.newaxis]
        return transformed_points[:, :2]

    def warp_image(self, base_img, overlay_img, H_matrix, out_shape):
        h, w = out_shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        homogeneous_coords = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1).reshape(-1, 3)

        inv_H_matrix = np.linalg.inv(H_matrix)
        transformed_coords = homogeneous_coords @ inv_H_matrix.T
        transformed_coords /= transformed_coords[:, 2, np.newaxis]
        src_x, src_y = transformed_coords[:, 0], transformed_coords[:, 1]

        valid_coords = (0 <= src_x) & (src_x < overlay_img.shape[1] - 1) & (0 <= src_y) & (src_y < overlay_img.shape[0] - 1)
        flat_overlay_img = overlay_img.reshape(-1, overlay_img.shape[2])

        return self.bilinear_interpolation(src_x, src_y, valid_coords, flat_overlay_img, overlay_img.shape, out_shape)

    def inverse_warp(self, base_img, overlay_img, H_matrix):
        base_height, base_width = base_img.shape[:2]
        overlay_height, overlay_width = overlay_img.shape[:2]
        overlay_corners = np.array([[0, 0], [overlay_width, 0], [overlay_width, overlay_height], [0, overlay_height]])
        transformed_corners = self.apply_homography_to_points(H_matrix, overlay_corners)
        all_corners = np.vstack((transformed_corners, [[0, 0], [base_width, 0], [base_width, base_height], [0, base_height]]))
        min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
        max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)

        output_size = (max_y - min_y, max_x - min_x)
        warped_overlay = self.warp_image(base_img, overlay_img, H_matrix, output_size)
        return self.blend_images(base_img, warped_overlay, min_x, min_y, output_size)

    def blend_images(self, base_img, overlay_img, offset_x, offset_y, output_size):
        blended_image = np.zeros((output_size[0], output_size[1], 3), dtype=base_img.dtype)
        blended_image[-offset_y:-offset_y + base_img.shape[0], -offset_x:-offset_x + base_img.shape[1]] = base_img
        mask_base = (blended_image > 0).astype(np.float32)
        mask_overlay = (overlay_img > 0).astype(np.float32)
        combined_mask = mask_base + mask_overlay
        safe_mask = np.where(combined_mask == 0, 1, combined_mask)
        return (blended_image * mask_base + overlay_img * mask_overlay) / safe_mask

    def bilinear_interpolation(self, src_x, src_y, valid_coords, flat_overlay_img, img_shape, out_shape):
        int_x, int_y = src_x[valid_coords].astype(int), src_y[valid_coords].astype(int)
        valid_src = (0 <= int_x) & (int_x < img_shape[1] - 1) & (0 <= int_y) & (int_y < img_shape[0] - 1)
        if not valid_src.any():
            return np.zeros(out_shape + (3,), dtype=flat_overlay_img.dtype)

        valid_x, valid_y = int_x[valid_src], int_y[valid_src]
        base_img_pts = (1 - (src_x[valid_coords] - valid_x)) * (1 - (src_y[valid_coords] - valid_y))
        combined_pts = base_img_pts[:, np.newaxis] * flat_overlay_img[valid_y * img_shape[1] + valid_x]
        combined_img = np.zeros(out_shape + (3,), dtype=flat_overlay_img.dtype)
        combined_img[valid_coords[valid_src], :] = combined_pts
        return combined_img
