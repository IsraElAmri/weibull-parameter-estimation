import os
from pathlib import Path
from PIL import Image

def combine_images(image_paths, output_path, grid_size=(2, 2)):
    """Combines multiple images into a grid."""
    images = []
    for path in image_paths:
        if path.exists():
            images.append(Image.open(path))
        else:
            print(f"Warning: {path.name} not found.")
            return

    if not images:
        return

    # Assuming all images are the same size (from our R plots, they're 3000x2400)
    img_width, img_height = images[0].size
    
    cols, rows = grid_size
    grid_width = cols * img_width
    grid_height = rows * img_height

    # Create a new white image for the background
    grid_img = Image.new('RGB', (grid_width, grid_height), color='white')

    for index, img in enumerate(images):
        # Resize if somehow different, though they shouldn't be
        if img.size != (img_width, img_height):
            img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
            
        row = index // cols
        col = index % cols
        x_offset = col * img_width
        y_offset = row * img_height
        
        grid_img.paste(img, (x_offset, y_offset))

    # Save with high quality
    grid_img.save(output_path, quality=95)
    print(f"Saved: {output_path.name}")

def main():
    root_dir = Path(__file__).resolve().parent.parent
    plots_dir = root_dir / "R" / "results" / "plots"
    combined_dir = root_dir / "R" / "results" / "plots_combined"
    
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    # Models to process (must match R cleanly stripped names)
    models = ["TOSHIBA_MG07ACA14TA", "TOSHIBA_MG08ACA16TA", "WDC_WUH722222ALE6L4"]
    
    for model in models:
        # Group 1: Probability, CDF, Survival, PDF
        group1 = [
            plots_dir / f"plot1_weibull_probability_{model}.png",
            plots_dir / f"plot2_cdf_{model}.png",
            plots_dir / f"plot3_survival_{model}.png",
            plots_dir / f"plot4_pdf_{model}.png"
        ]
        
        # Group 2: Hazard, GoF Heatmap, Composite Rank, Parameters
        group2 = [
            plots_dir / f"plot5_hazard_{model}.png",
            plots_dir / f"plot6_gof_heatmap_{model}.png",
            plots_dir / f"plot7_composite_rank_{model}.png",
            plots_dir / f"plot8_parameters_{model}.png"
        ]
        
        combine_images(group1, combined_dir / f"combined_grid1_distributions_{model}.png")
        combine_images(group2, combined_dir / f"combined_grid2_metrics_{model}.png")

if __name__ == "__main__":
    main()
