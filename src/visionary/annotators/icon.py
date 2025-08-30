
import cv2
import numpy as np

class IconAnnotator:
    """Annotator for placing icons/symbols on images."""

    def __init__(self, icon_size=(32, 32)):
        """
        Args:
            icon_size (tuple): Default size for icons (width, height)
        """
        self.icon_size = icon_size

    def _create_basic_icon(self, icon_type, size, color):
        """Create basic geometric icons."""
        icon = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        center_x, center_y = size[0] // 2, size[1] // 2

        if icon_type == 'circle':
            radius = min(size) // 3
            cv2.circle(icon, (center_x, center_y), radius, color, -1)

        elif icon_type == 'square':
            side = min(size) // 2
            cv2.rectangle(icon, (center_x - side//2, center_y - side//2), 
                         (center_x + side//2, center_y + side//2), color, -1)

        elif icon_type == 'triangle':
            points = np.array([
                [center_x, center_y - size[1]//3],
                [center_x - size[0]//3, center_y + size[1]//4],
                [center_x + size[0]//3, center_y + size[1]//4]
            ], dtype=np.int32)
            cv2.fillPoly(icon, [points], color)

        elif icon_type == 'cross':
            thickness = max(2, min(size) // 8)
            cv2.line(icon, (center_x - size[0]//3, center_y), 
                    (center_x + size[0]//3, center_y), color, thickness)
            cv2.line(icon, (center_x, center_y - size[1]//3), 
                    (center_x, center_y + size[1]//3), color, thickness)

        elif icon_type == 'arrow_right':
            points = np.array([
                [center_x - size[0]//3, center_y - size[1]//4],
                [center_x + size[0]//4, center_y],
                [center_x - size[0]//3, center_y + size[1]//4]
            ], dtype=np.int32)
            cv2.fillPoly(icon, [points], color)

        return icon

    def annotate(self, image, icons):
        """
        Place icons on the image.

        Args:
            image (np.ndarray): Input image
            icons (list): List of icons to place. Each should be dict with:
                        - 'position': (x, y) position for icon center
                        - 'type': icon type ('circle', 'square', 'triangle', 'cross', 'arrow_right')
                          OR 'image': numpy array of icon image
                        - 'color': color for basic icons (ignored if 'image' provided)
                        - 'size': size override (optional)

        Returns:
            np.ndarray: Annotated image
        """
        annotated_image = image.copy()

        for icon_info in icons:
            x, y = map(int, icon_info['position'])
            size = icon_info.get('size', self.icon_size)

            if 'image' in icon_info:
                # Use provided icon image
                icon = icon_info['image']
                # Resize if needed
                if icon.shape[:2] != (size[1], size[0]):
                    icon = cv2.resize(icon, size)

            elif 'type' in icon_info:
                # Create basic geometric icon
                icon_type = icon_info['type']
                color = icon_info.get('color', (255, 255, 255))
                icon = self._create_basic_icon(icon_type, size, color)

            else:
                continue

            # Calculate placement bounds
            half_width, half_height = size[0] // 2, size[1] // 2
            x1 = max(0, x - half_width)
            y1 = max(0, y - half_height)
            x2 = min(annotated_image.shape[1], x + half_width)
            y2 = min(annotated_image.shape[0], y + half_height)

            # Adjust icon size if it goes out of bounds
            icon_x1 = half_width - (x - x1)
            icon_y1 = half_height - (y - y1)
            icon_x2 = icon_x1 + (x2 - x1)
            icon_y2 = icon_y1 + (y2 - y1)

            if icon_x2 > icon.shape[1]:
                icon_x2 = icon.shape[1]
            if icon_y2 > icon.shape[0]:
                icon_y2 = icon.shape[0]

            # Place icon (simple overlay - could add alpha blending)
            roi = annotated_image[y1:y2, x1:x2]
            icon_roi = icon[icon_y1:icon_y2, icon_x1:icon_x2]

            # Create mask where icon is not black (for transparency effect)
            mask = np.any(icon_roi != [0, 0, 0], axis=2)
            roi[mask] = icon_roi[mask]

        return annotated_image
