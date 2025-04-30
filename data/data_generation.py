import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.animation as animation
import argparse

def find_coordinate_range(drawing):
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    for stroke in drawing:
        if len(stroke[0]) > 0:
            x_coords = stroke[0]
            y_coords = stroke[1]
            min_x = min(min_x, min(x_coords))
            max_x = max(max_x, max(x_coords))
            min_y = min(min_y, min(y_coords))
            max_y = max(max_y, max(y_coords))

    x_margin = (max_x - min_x) * 0.05 if max_x > min_x else 10
    y_margin = (max_y - min_y) * 0.05 if max_y > min_y else 10
    
    return (
        min_x - x_margin,
        max_x + x_margin,
        min_y - y_margin,
        max_y + y_margin
    )

def drawing_to_vector_frames(drawing, num_frames):
    total_points = sum(len(stroke[0]) for stroke in drawing)
    points_per_frame = total_points / num_frames if num_frames > 1 else total_points
    
    frames = []
    
    for frame_idx in range(num_frames):
        visible_point_count = min(int(round(points_per_frame * (frame_idx + 1))), total_points)
        frame = []
        points_so_far = 0
        
        for stroke_idx, stroke in enumerate(drawing):
            x_coords = stroke[0]
            y_coords = stroke[1]
            stroke_points = len(x_coords)
            
            # Determine how many points of this stroke to show
            if points_so_far >= visible_point_count:
                continue
            
            points_remaining = visible_point_count - points_so_far
            points_to_show = min(stroke_points, points_remaining)
            
            # Add visible part of the stroke to the frame
            if points_to_show > 0:
                visible_stroke = {
                    'stroke_idx': stroke_idx,
                    'x': x_coords[:points_to_show],
                    'y': y_coords[:points_to_show]
                }
                frame.append(visible_stroke)
            
            points_so_far += stroke_points
        
        frames.append(frame)
    
    return frames

def render_to_pixel_array(frame, coordinate_range, canvas_size=(256, 256), line_width=3, dpi=100):
    fig_width = canvas_size[0] / dpi
    fig_height = canvas_size[1] / dpi
    fig = plt.Figure(figsize=(fig_width, fig_height), dpi=dpi)
    
    # Create canvas for rendering
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    
    # Set background to white (1 in binary)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Set limits based on the coordinate range of the entire drawing
    min_x, max_x, min_y, max_y = coordinate_range
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    
    # Plot each stroke in black (0 in binary)
    for stroke in frame:
        # Check if the stroke has points
        if len(stroke['x']) > 0:
            ax.plot(stroke['x'], stroke['y'], color='black', linewidth=line_width)
    
    ax.invert_yaxis()  # QuickDraw has origin at top-left
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Render to buffer
    fig.tight_layout(pad=0)
    canvas.draw()
    
    # Convert to numpy array
    buf = canvas.buffer_rgba()
    pixel_array = np.asarray(buf).copy()
    
    # Convert to binary (0 for black, 1 for white)
    # First convert to grayscale
    gray = 0.299 * pixel_array[:, :, 0] + 0.587 * pixel_array[:, :, 1] + 0.114 * pixel_array[:, :, 2]
    
    # Threshold to binary (0 and 1)
    binary = (gray < 128).astype(np.uint8)
    
    plt.close(fig)
    
    return binary
    
def drawing_to_pixel_frames(drawing, num_frames, canvas_size=(256, 256), line_width=3):
    vector_frames = drawing_to_vector_frames(drawing, num_frames)
    coordinate_range = find_coordinate_range(drawing)
    
    # Convert each frame to pixels
    pixel_frames = []
    for frame in vector_frames:
        pixel_frame = render_to_pixel_array(
            frame, 
            coordinate_range=coordinate_range,
            canvas_size=canvas_size, 
            line_width=line_width
        )
        pixel_frames.append(pixel_frame)
    
    return pixel_frames
        
def create_animation_from_pixel_frames(pixel_frames, output_file="animation.mp4", fps=10):
    height, width = pixel_frames[0].shape
    dpi = 100
    fig = plt.Figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    
    def update(frame_idx):
        ax.clear()
        ax.axis('off')
        ax.imshow(pixel_frames[frame_idx], cmap='binary')
        return ax,

    ani = animation.FuncAnimation(fig, update, frames=len(pixel_frames), interval=1000/fps, blit=True)
    
    ani.save(output_file, writer='ffmpeg', fps=fps)
    plt.close(fig)

def generate_npz(categories, size=10, num_frames=60, image_size=(256, 256), line_width=3):
    train_images = []
    train_labels = []
    valid_images = []
    valid_labels = []
    test_images = []
    test_labels = []

    for category_idx, category in enumerate(categories):
        file_path = f"{category}.ndjson"

        count = 0
        max_per_category = 1.2 * size
        
        with open(file_path, 'r') as f:
            for line in f:
                if count >= max_per_category:
                    break
                    
                data = json.loads(line)
                
                if data['recognized'] and len(data['drawing']) > 0:
                    drawing = data['drawing']
                    pixel_image = drawing_to_pixel_frames(
                                    drawing, 
                                    num_frames=num_frames, 
                                    canvas_size=image_size, 
                                    line_width=line_width
                                )

                    if count < size:
                        train_images.append(pixel_image)
                        train_labels.append(category_idx)
                    elif count < 1.1 * size:
                        valid_images.append(pixel_image)
                        valid_labels.append(category_idx)
                    else:
                        test_images.append(pixel_image)
                        test_labels.append(category_idx)
                        
                    count += 1
        
        print(f"Finished processing {category} with {count} samples.")

    train_images = np.array(train_images, dtype=np.uint8)
    train_labels = np.array(train_labels, dtype=np.uint8)
    test_images = np.array(test_images, dtype=np.uint8)
    test_labels = np.array(test_labels, dtype=np.uint8)

    np.savez('quickdraw_10k.npz', 
            train_images=train_images, train_labels=train_labels,
            valid_images=valid_images, valid_labels=valid_labels,
            test_images=test_images, test_labels=test_labels)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate QuickDraw dataset')
    parser.add_argument('--categories', nargs='+', default=['apple', 'cat', 'cloud', 'computer', 'diamond', 'moon', 'mushroom', 'snowflake', 'star'], help='List of categories')
    parser.add_argument('--size', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--frames', type=int, default=60, help='Number of frames per animation')
    parser.add_argument('--width', type=int, default=256, help='Image width')
    parser.add_argument('--height', type=int, default=256, help='Image height')
    parser.add_argument('--line-width', type=int, default=3, help='Line width for drawing')
    
    args = parser.parse_args()
    
    generate_npz(
        args.categories, 
        size=args.size, 
        num_frames=args.frames, 
        image_size=(args.width, args.height),
        line_width=args.line_width
    )
    
    print("NPZ file generated successfully.")
