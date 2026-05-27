"""
board_detector.py
Detects a chess board in a photo, warps it to a top-down square,
and splits it into 64 individual squares.
"""

import cv2
import numpy as np
from pathlib import Path


class BoardDetector:
    """Detects and extracts a chess board from a photo."""

    def __init__(self, warp_size=800):
        # The warped board will be warp_size x warp_size pixels
        self.warp_size = warp_size
        # Each of the 64 squares will be this many pixels
        self.square_size = warp_size // 8
    def detect_corners_auto(self, image):
        """
        Try to auto-detect the 4 board corners.
        Returns 4 corner points, or None if detection fails.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # --- METHOD B: Contour-based quadrilateral detection ---
        # Blur to reduce noise, then find edges
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Close gaps in edges
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        # Find contours (closed shapes)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Look for the biggest 4-sided shape (the board)
        best_quad = None
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Ignore tiny shapes
            if area < 0.2 * image.shape[0] * image.shape[1]:
                continue
            # Approximate the contour to a polygon
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # We want a 4-corner shape with the largest area
            if len(approx) == 4 and area > max_area:
                max_area = area
                best_quad = approx

        if best_quad is not None:
            print("✅ Board detected via contour method")
            return best_quad.reshape(4, 2).astype(np.float32)

        print("⚠️ Auto-detection failed")
        return None
    
    
    def select_corners_manual(self, image):
        """
        Fallback: user clicks the 4 board corners.
        Order: top-left, top-right, bottom-right, bottom-left.
        Press 'u' to undo last click, ESC to abort.
        """
        corners = []
        win = "Click 4 corners: TL, TR, BR, BL  (u=undo)"

        # Resize big images for display
        scale = 1.0
        display_base = image.copy()
        if display_base.shape[1] > 1000:
            scale = 1000 / display_base.shape[1]
            display_base = cv2.resize(display_base, None, fx=scale, fy=scale)

        def redraw():
            """Redraw the image with current corner dots."""
            img = display_base.copy()
            labels = ["TL", "TR", "BR", "BL"]
            for i, (x, y) in enumerate(corners):
                cv2.circle(img, (x, y), 7, (0, 255, 0), -1)
                cv2.putText(img, labels[i], (x + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Instruction text
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

        print("🖱️  Click 4 corners in order: TL, TR, BR, BL")
        print("    Press 'u' to undo, ESC to abort.")

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # ESC
                corners = []
                break
            if key == ord('u') and corners:  # undo
                corners.pop()
                redraw()
            if len(corners) == 4:
                cv2.waitKey(500)  # brief pause to see result
                break
            # Detect if window was closed manually
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                corners = []
                break

        cv2.destroyAllWindows()

        if len(corners) < 4:
            print("❌ Corner selection cancelled.")
            return None

        # Scale corners back to original image size
        corners = np.array(corners, dtype=np.float32) / scale
        return corners
    
    
    def _order_corners(self, pts):
        """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
        pts = np.array(pts, dtype=np.float32)
        ordered = np.zeros((4, 2), dtype=np.float32)

        # Sum of coords: TL has smallest, BR has largest
        s = pts.sum(axis=1)
        ordered[0] = pts[np.argmin(s)]  # top-left
        ordered[2] = pts[np.argmax(s)]  # bottom-right

        # Diff of coords: TR has smallest, BL has largest
        diff = np.diff(pts, axis=1)
        ordered[1] = pts[np.argmin(diff)]  # top-right
        ordered[3] = pts[np.argmax(diff)]  # bottom-left

        return ordered

    def warp_to_square(self, image, corners):
        """Warp the board (from 4 corners) into a perfect square."""
        ordered = self._order_corners(corners)

        # Destination: a perfect square
        dst = np.array([
            [0, 0],
            [self.warp_size, 0],
            [self.warp_size, self.warp_size],
            [0, self.warp_size]
        ], dtype=np.float32)

        # Compute homography matrix and apply it
        matrix = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(image, matrix,
                                     (self.warp_size, self.warp_size))
        return warped
    def split_into_squares(self, warped):
        """
        Split the warped board into 64 squares.
        Returns a list of 64 dicts: {row, col, image}.
        Row 0 = top, Col 0 = left.
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
    
# Test block
if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    TEST_PHOTO = PROJECT_ROOT / "data" / "test_photos" / "test_board.jpg"

    print(f"📂 Loading photo: {TEST_PHOTO}\n")
    image = cv2.imread(str(TEST_PHOTO))

    if image is None:
        print("❌ Could not load photo! Check the path.")
        exit()

    print(f"✅ Photo loaded: {image.shape[1]}x{image.shape[0]} pixels\n")

    detector = BoardDetector(warp_size=800)

    # Try auto-detection first
    corners = detector.detect_corners_auto(image)

    # Fallback to manual if auto fails
    if corners is None:
        print("→ Falling back to manual corner selection...")
        corners = detector.select_corners_manual(image)

    if corners is None:
        print("❌ No corners. Aborting.")
        exit()

    # Warp and split
    warped = detector.warp_to_square(image, corners)
    squares = detector.split_into_squares(warped)

    print(f"\n✅ Board warped and split into {len(squares)} squares")

    # Show the warped board
    cv2.imshow("Warped Board", warped)
    print("👁️  Showing warped board. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the warped board for the report
    out_path = SCRIPT_DIR.parent / "data" / "test_photos" / "warped_board.jpg"
    cv2.imwrite(str(out_path), warped)
    print(f"💾 Warped board saved to: {out_path}") 