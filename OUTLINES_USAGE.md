# Voronoi Outlines Feature

The Genetic Art Algorithm now supports adding customizable outlines to Voronoi regions, which serve as "solder" to bind regions together visually. These outlines are drawn after the final generation is complete.

## Parameters

### `OUTLINE_WIDTH` (integer, default: 0)
- Width of the outline in pixels at the output scale (1x)
- Set to 0 to disable outlines
- Example: `OUTLINE_WIDTH=2` creates 2-pixel outlines at 1x scale, 4-pixel at 2x scale

### `OUTLINE_COLOR` (RGB tuple, default: "0,0,0" for black)
- Color of the outlines as comma-separated RGB values (0-255)
- Format: `R,G,B` (no spaces)
- Example: `OUTLINE_COLOR=255,255,255` for white outlines
- Example: `OUTLINE_COLOR=128,0,128` for purple outlines

## Usage Examples

### Single-generation with white 2-pixel outlines:
```bash
OUTLINE_WIDTH=2 OUTLINE_COLOR=255,255,255 python evolve_voronoi.py
```

### Divide-and-conquer with black 1-pixel outlines:
```bash
USE_DIVIDE_AND_CONQUER=1 OUTLINE_WIDTH=1 python evolve_voronoi.py
```

### Custom outline color (dark blue):
```bash
OUTLINE_WIDTH=3 OUTLINE_COLOR=0,51,102 python evolve_voronoi.py
```

## Technical Details

- Outlines are drawn **after** the final image is rendered
- Outline width scales with the `OUTPUT_SCALE` parameter
- Outlines are applied to all Voronoi regions uniformly
- The outline color persists through all evolution generations
- Works with both regular and divide-and-conquer evolution modes
- Outlines are cached along with the rendered image

## Visual Effect

The outlines create a "solder joint" or border effect between adjacent Voronoi regions:
- **Thin outlines (1-2px)** subtly define region boundaries
- **Thick outlines (3-5px)** create a bold, prominent separation
- **Dark outlines** provide high contrast and clear definition
- **Light outlines** integrate smoothly with bright color palettes
- **Colored outlines** can complement or contrast with the overall image tone

## Notes

- For best results, choose outline colors that provide contrast with your target image
- Adjust `OUTLINE_WIDTH` based on the desired visual prominence
- Outlines scale proportionally with `OUTPUT_SCALE`, so plan accordingly
- Outline styling is inherited by child paintings during evolution, ensuring consistency
