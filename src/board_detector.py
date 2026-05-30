"""
Detect a chessboard in a photo, warp it to a top-down square,
and split it into the 64 individual squares.
"""

import cv2
import numpy as np
from pathlib import Path


class BoardDetector:
    """Find and extract a chessboard from an image."""

    def __init__(self, warp_size=800):
        # Final size of the warped board image.
        self.warp_size = warp_size
        # Size of each square in the warped board.
        self.square_size = warp_size // 8
    def detect_corners_auto(self, image):
        """
        Try to detect the four board corners automatically.

        Returns the corner points, or None if detection fails.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Smooth the image before edge detection.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Thicken edges so the board outline is easier to detect.
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        # Find the outer contours in the image.
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # The board should be the largest four-sided contour.
        best_quad = None
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Skip small shapes that are unlikely to be the board.
            if area < 0.2 * image.shape[0] * image.shape[1]:
                continue
            # Simplify the contour to a polygon.
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # Keep the largest polygon with exactly four corners.
            if len(approx) == 4 and area > max_area:
                max_area = area
                best_quad = approx

        if best_quad is not None:
            print("Board detected via contour method")
            return best_quad.reshape(4, 2).astype(np.float32)

        print("Auto-detection failed")
        return None
    
    
    def select_corners_manual(self, image):
        """
        Let the user click the four board corners if automatic detection fails.

        Clicks should be entered in this order: top-left, top-right,
        bottom-right, bottom-left.

        Press 'u' to undo the last click, or ESC to cancel.
        """
        corners = []
        win = "Click 4 corners: TL, TR, BR, BL  (u=undo)"

        # Scale down large images so they fit on screen.
        scale = 1.0
        display_base = image.copy()
        if display_base.shape[1] > 1000:
            scale = 1000 / display_base.shape[1]
            display_base = cv2.resize(display_base, None, fx=scale, fy=scale)

        def redraw():
            """Refresh the preview with the selected corner points."""
            img = display_base.copy()
            labels = ["TL", "TR", "BR", "BL"]
            for i, (x, y) in enumerate(corners):
                cv2.circle(img, (x, y), 7, (0, 255, 0), -1)
                cv2.putText(img, labels[i], (x + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Show the next expected click.
            remaining = 4 - len(corners)
            msg = f"Click {labels[len(corners)]}" if remaining > 0 else "Done! Any key."
            cv2.putText(img, msg, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow(win, img)

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
                corners.append((x, y))
                redraw()

        cv2.namedWindow(win)
        cv2.setMouseCallback(win, on_click)
        redraw()

        print("Click the four corners in order: TL, TR, BR, BL")
        print("Press 'u' to undo, ESC to cancel.")

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # ESC
                corners = []
                break
            if key == ord('u') and corners:  # Undo the last click.
                corners.pop()
                redraw()
            if len(corners) == 4:
                cv2.waitKey(500)  # Brief pause so the final selection is visible.
                break
            # Stop if the window was closed manually.
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                corners = []
                break

        cv2.destroyAllWindows()

        if len(corners) < 4:
            print("Corner selection cancelled.")
            return None

        # Convert the clicks back to the original image size.
        corners = np.array(corners, dtype=np.float32) / scale
        return corners
    
    
    def _order_corners(self, pts):
        """Order four points as top-left, top-right, bottom-right, bottom-left."""
        pts = np.array(pts, dtype=np.float32)
        ordered = np.zeros((4, 2), dtype=np.float32)

        # Top-left has the smallest sum, bottom-right the largest.
        s = pts.sum(axis=1)
        ordered[0] = pts[np.argmin(s)]
        ordered[2] = pts[np.argmax(s)]

        # Top-right has the smallest difference, bottom-left the largest.
        diff = np.diff(pts, axis=1)
        ordered[1] = pts[np.argmin(diff)]
        ordered[3] = pts[np.argmax(diff)]

        return ordered

    def warp_to_square(self, image, corners):
        """Warp the board into a square using the detected corners."""
        ordered = self._order_corners(corners)

        # Destination coordinates for the warped board.
        dst = np.array([
            [0, 0],
            [self.warp_size, 0],
            [self.warp_size, self.warp_size],
            [0, self.warp_size]
        ], dtype=np.float32)

        # Build the perspective transform and apply it.
        matrix = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(image, matrix,
                                     (self.warp_size, self.warp_size))
        return warped
    def split_into_squares(self, warped):
        """
        Split the warped board into its 64 squares.

        Returns a list of dictionaries with row, col, and image keys.
        Row 0 is the top row and column 0 is the leftmost column.
        """
        squares = []
        s = self.square_size

        for row in range(8):
            for col in range(8):
                y1, y2 = row * s, (row + 1) * s
                x1, x2 = col * s, (col + 1) * s
                square_img = warped[y1:y2, x1:x2]
                squares.append({
                    'row': row,
                    'col': col,
                    'image': square_img
                })
        return squares
    
# Simple test block.
if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    TEST_PHOTO = PROJECT_ROOT / "data" / "test_photos" / "test_board.jpg"

    print(f"Loading photo: {TEST_PHOTO}\n")
    image = cv2.imread(str(TEST_PHOTO))

    if image is None:
        print("Could not load the photo. Check the path.")
        exit()

    print(f"Photo loaded: {image.shape[1]}x{image.shape[0]} pixels\n")

    detector = BoardDetector(warp_size=800)

    # Try auto-detection first
    corners = detector.detect_corners_auto(image)

    # Fallback to manual if auto fails
    if corners is None:
        print("Falling back to manual corner selection...")
        corners = detector.select_corners_manual(image)

    if corners is None:
        print("No corners found. Aborting.")
        exit()

    # Warp and split
    warped = detector.warp_to_square(image, corners)
    squares = detector.split_into_squares(warped)

    print(f"\nBoard warped and split into {len(squares)} squares")

    # Show the warped board
    cv2.imshow("Warped Board", warped)
    print("Showing warped board. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the warped board image for later use.
    out_path = SCRIPT_DIR.parent / "data" / "test_photos" / "warped_board.jpg"
    cv2.imwrite(str(out_path), warped)
    print(f"Warped board saved to: {out_path}")
