import cv2
import numpy as np
from PIL import Image


class PrescriptionPreprocessor:
    """
    Preprocessing and line segmentation for handwritten prescriptions.

    Pipeline:
        Original image
            ↓
        Upscaling
            ↓
        Grayscale
            ↓
        Noise reduction
            ↓
        Contrast enhancement
            ↓
        Sharpening
            ↓
        Handwriting/line detection
    """

    def __init__(self):
        self.target_min_width = 1400
        self.max_width = 1800

    # ============================================================
    # MAIN PREPROCESSING
    # ============================================================

    def preprocess(self, pil_image):
        """
        Improve the prescription image for OCR.

        Small images are enlarged because handwritten OCR performs
        poorly when the original image is very small.
        """

        if pil_image is None:
            raise ValueError("Input image is None.")

        # PIL -> RGB numpy
        image = np.array(pil_image.convert("RGB"))

        height, width = image.shape[:2]

        # --------------------------------------------------------
        # UPSCALE SMALL IMAGES
        # --------------------------------------------------------

        if width < self.target_min_width:
            scale = self.target_min_width / width

            new_width = int(width * scale)
            new_height = int(height * scale)

            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # --------------------------------------------------------
        # DOWNSCALE VERY LARGE IMAGES
        # --------------------------------------------------------

        elif width > self.max_width:
            scale = self.max_width / width

            new_width = int(width * scale)
            new_height = int(height * scale)

            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # --------------------------------------------------------
        # RGB -> GRAYSCALE
        # --------------------------------------------------------

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # --------------------------------------------------------
        # NOISE REDUCTION
        # --------------------------------------------------------

        denoised = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)

        # --------------------------------------------------------
        # CLAHE CONTRAST ENHANCEMENT
        # --------------------------------------------------------

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        enhanced = clahe.apply(denoised)

        # --------------------------------------------------------
        # GENTLE SHARPENING
        # --------------------------------------------------------

        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.2)

        sharpened = cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)

        # --------------------------------------------------------
        # GRAY -> RGB
        # --------------------------------------------------------

        cleaned_rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)

        return Image.fromarray(cleaned_rgb)

    # ============================================================
    # BINARY IMAGE
    # ============================================================

    def create_binary_image(self, pil_image):
        """
        Create a binary image used for detecting handwriting.

        This image should NOT be sent directly to TrOCR.
        """

        if pil_image is None:
            raise ValueError("Input image is None.")

        image = np.array(pil_image.convert("RGB"))

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Small blur removes camera noise.
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)

        # Remove tiny isolated noise.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return binary

    # ============================================================
    # BLUE INK MASK
    # ============================================================

    def create_blue_ink_mask(self, pil_image):
        """
        Detect blue handwriting.

        Many prescriptions contain blue pen handwriting.
        This mask helps reduce interference from printed
        hospital text and the paper background.
        """

        image = np.array(pil_image.convert("RGB"))

        r = image[:, :, 0].astype(np.int16)
        g = image[:, :, 1].astype(np.int16)
        b = image[:, :, 2].astype(np.int16)

        blue_difference = b - r

        mask = (blue_difference > 5) & (b > g)

        mask = mask.astype(np.uint8) * 255

        # Remove isolated pixels.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    # ============================================================
    # TEXT REGION VALIDATION
    # ============================================================

    def _is_probable_text_region(self, binary_crop, min_ink_density=0.008, max_ink_density=0.50):
        """
        Check whether a crop contains a reasonable amount
        of handwriting/ink.
        """

        if binary_crop is None:
            return False

        if binary_crop.size == 0:
            return False

        density = np.count_nonzero(binary_crop) / binary_crop.size

        return min_ink_density <= density <= max_ink_density

    # ============================================================
    # MERGE NEARBY LINE BOXES
    # ============================================================

    def _merge_close_boxes(self, boxes, vertical_gap=18):
        """
        Merge bounding boxes that belong to the same
        handwritten prescription line.
        """

        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda b: b[1])

        merged = []

        for box in boxes:
            x1, y1, x2, y2 = box

            if not merged:
                merged.append([x1, y1, x2, y2])

                continue

            previous = merged[-1]

            px1, py1, px2, py2 = previous

            # Check whether this box overlaps or is close
            # vertically to the previous line.
            if y1 <= py2 + vertical_gap:
                previous[0] = min(px1, x1)

                previous[1] = min(py1, y1)

                previous[2] = max(px2, x2)

                previous[3] = max(py2, y2)

            else:
                merged.append([x1, y1, x2, y2])

        return merged

    # ============================================================
    # LINE SEGMENTATION
    # ============================================================

    def extract_text_lines(self, cleaned_image, debug=False):
        """
        Detect handwritten prescription lines.

        Returns:
            list[PIL.Image]

        Each returned image represents one possible
        handwritten line.
        """

        if cleaned_image is None:
            raise ValueError("cleaned_image is None.")

        image_np = np.array(cleaned_image.convert("RGB"))

        height, width = image_np.shape[:2]

        # --------------------------------------------------------
        # CREATE MASKS
        # --------------------------------------------------------

        binary = self.create_binary_image(cleaned_image)

        blue_mask = self.create_blue_ink_mask(cleaned_image)

        # --------------------------------------------------------
        # PRESCRIPTION CONTENT AREA
        #
        # Ignore:
        # - hospital header
        # - patient information area
        # - bottom signature area
        # --------------------------------------------------------

        content_start = int(height * 0.22)

        content_end = int(height * 0.94)

        binary[:content_start, :] = 0
        binary[content_end:, :] = 0

        blue_mask[:content_start, :] = 0
        blue_mask[content_end:, :] = 0

        # --------------------------------------------------------
        # CHOOSE MASK
        # --------------------------------------------------------

        blue_pixels = np.count_nonzero(blue_mask)

        if blue_pixels > 100:
            working_mask = blue_mask.copy()

            if debug:
                print(f"[PREPROCESSOR] Blue ink detected: {blue_pixels} pixels")

        else:
            working_mask = binary.copy()

            if debug:
                print("[PREPROCESSOR] Blue ink not reliable. Using grayscale mask.")

        # --------------------------------------------------------
        # CONNECT CHARACTERS IN SAME LINE
        #
        # IMPORTANT:
        # This kernel is deliberately small.
        #
        # Your previous code used:
        #
        #     width // 18
        #
        # which can connect multiple prescription rows.
        # --------------------------------------------------------

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 2))

        connected = cv2.morphologyEx(working_mask, cv2.MORPH_CLOSE, horizontal_kernel)

        # --------------------------------------------------------
        # CONNECTED COMPONENTS
        # --------------------------------------------------------

        num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(connected, connectivity=8)

        components = []

        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])

            y = int(stats[i, cv2.CC_STAT_TOP])

            w = int(stats[i, cv2.CC_STAT_WIDTH])

            h = int(stats[i, cv2.CC_STAT_HEIGHT])

            area = int(stats[i, cv2.CC_STAT_AREA])

            # ----------------------------------------------------
            # FILTER TINY NOISE
            # ----------------------------------------------------

            if area < 15:
                continue

            if w < 12:
                continue

            if h < 3:
                continue

            # Reject something covering almost the whole image.
            if w > width * 0.95:
                continue

            # Reject extremely tall regions.
            if h > height * 0.15:
                continue

            center_y = y + h / 2

            components.append({"x": x, "y": y, "w": w, "h": h, "area": area, "center_y": center_y})

        if debug:
            print(f"[PREPROCESSOR] Components after filtering: {len(components)}")

        # --------------------------------------------------------
        # GROUP COMPONENTS BY VERTICAL POSITION
        # --------------------------------------------------------

        components.sort(key=lambda item: item["center_y"])

        line_groups = []

        for component in components:
            center_y = component["center_y"]

            placed = False

            for group in line_groups:
                group_center = np.mean([item["center_y"] for item in group])

                # Adaptive vertical tolerance.
                tolerance = max(12, int(height * 0.018))

                if abs(center_y - group_center) <= tolerance:
                    group.append(component)

                    placed = True
                    break

            if not placed:
                line_groups.append([component])

        # --------------------------------------------------------
        # CREATE LINE BOXES
        # --------------------------------------------------------

        line_boxes = []

        for group in line_groups:
            if not group:
                continue

            x1 = min(item["x"] for item in group)

            y1 = min(item["y"] for item in group)

            x2 = max(item["x"] + item["w"] for item in group)

            y2 = max(item["y"] + item["h"] for item in group)

            w = x2 - x1
            h = y2 - y1

            # Minimum useful line width.
            if w < width * 0.08:
                continue

            if h < 5:
                continue

            if h > height * 0.20:
                continue

            line_boxes.append([x1, y1, x2, y2])

        # --------------------------------------------------------
        # MERGE VERY CLOSE BOXES
        # --------------------------------------------------------

        line_boxes = self._merge_close_boxes(line_boxes, vertical_gap=16)

        # --------------------------------------------------------
        # SORT TOP TO BOTTOM
        # --------------------------------------------------------

        line_boxes.sort(key=lambda box: box[1])

        if debug:
            print(f"[PREPROCESSOR] Detected lines: {len(line_boxes)}")

            for index, box in enumerate(line_boxes, start=1):
                print(f"  Line {index}: x={box[0]}, y={box[1]}, x2={box[2]}, y2={box[3]}")

        # --------------------------------------------------------
        # CROP LINES FROM ENHANCED IMAGE
        # --------------------------------------------------------

        line_images = []

        for index, box in enumerate(line_boxes, start=1):
            x1, y1, x2, y2 = box

            padding_x = 30
            padding_y = 18

            x1 = max(0, x1 - padding_x)

            y1 = max(0, y1 - padding_y)

            x2 = min(width, x2 + padding_x)

            y2 = min(height, y2 + padding_y)

            crop = image_np[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # ----------------------------------------------------
            # SANITY CHECK
            # ----------------------------------------------------

            gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

            binary_crop = cv2.adaptiveThreshold(
                gray_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
            )

            if not self._is_probable_text_region(binary_crop):
                if debug:
                    density = np.count_nonzero(binary_crop) / binary_crop.size

                    print(f"[PREPROCESSOR] Rejected line {index}; density={density:.4f}")

                continue

            line_images.append(Image.fromarray(crop))

        return line_images
